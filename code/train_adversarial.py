"""
train_adversarial.py — Adversarial training against payload padding.

v2 showed every detector learned "zero payload => scan": all CIC-IDS2017
PortScan flows carry 0–2 bytes, so a padded probe looks benign. This retrains
the same three detectors on the same training split, plus payload-padded
copies of the training-split PortScan flows.

Designs (--design), see docs/03_diversity_experiment_design.md:
  full  (v3)  every detector hardened on 1–1460 B               -> models_adv/
  cov1  (v4)  each payload band hardened in exactly 1 detector  -> models_cov1/
  cov2  (v4)  each payload band hardened in exactly 2 detectors -> models_cov2/
  Bands: A = 1–90 B, B = 91–362 B, C = 363–1460 B.

Controlled comparison with the original detectors:
  - identical split (random_state=42), scaler and label encoder (copied)
  - identical model hyperparameters (train_models.py / train_cnn.py)
  - the only change is the added padded scan flows: one per eligible
    training-split PortScan flow (71,381) for every detector in every design
  - padding sizes drawn uniformly within the detector's band(s), never at the
    agent's discrete sizes, so the defence isn't trained on the test attack
  - padded features computed with flow_features.set_forward_payload, the
    same validated CICFlowMeter arithmetic the simulator uses
  - test split and probe bank (held-out) are untouched

One deliberate difference: the CNN's best epoch is chosen on a validation
slice of the training data, not on the test split as train_cnn.py does.

Run in two processes (xgboost and torch each bundle libomp on macOS); the
trees stage also writes augmentation.json:
    python train_adversarial.py --design cov1 --stage trees
    python train_adversarial.py --design cov1 --stage cnn

--mask-feature NAME [NAME ...] retrains with those columns removed: the scaler
saved in <out>_nopsh/ is a masked_scaler.MaskedScaler that outputs 0 for them
at training and at every later inference call (docs/09_psh_diagnostic_design.md).

--model-seed S retrains the same design with a different training seed
(RF/XGBoost random_state, CNN torch seed) into <out>_msS/. The training data,
padded rows and splits stay identical, so only training randomness changes.
Without it the original seed (42) is used.
"""

import argparse
import json
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from build_probe_bank import DATA_PATH, MODEL_DIR, load_and_clean_data
from flow_features import MAX_PAYLOAD, is_single_probe, set_forward_payload
from masked_scaler import MaskedScaler
from project_log import log_event

CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
AUG_COPIES = 1      # padded copies per eligible training-split PortScan flow
AUG_SEED   = 2026
MODEL_SEED = 42     # RF/XGBoost random_state and CNN torch seed, as in train_models.py / v3
DETECTORS  = ("xgboost", "rf", "cnn")

BANDS = {"full": (1, MAX_PAYLOAD), "A": (1, 90), "B": (91, 362), "C": (363, MAX_PAYLOAD)}
DESIGNS = {
    "full": {"out": "models_adv",  "bands": {d: ["full"] for d in DETECTORS}},
    "cov1": {"out": "models_cov1", "bands": {"xgboost": ["A"], "rf": ["B"], "cnn": ["C"]}},
    "cov2": {"out": "models_cov2", "bands": {"xgboost": ["A", "B"], "rf": ["B", "C"], "cnn": ["C", "A"]}},
}


def sample_payloads(rng, n, band_names):
    """Equal share of rows per band, uniform within each band."""
    if len(band_names) == 1:
        lo, hi = BANDS[band_names[0]]
        return rng.integers(lo, hi + 1, size=n)
    which = rng.permutation(np.arange(n) % len(band_names))
    lo = np.array([BANDS[b][0] for b in band_names])[which]
    hi = np.array([BANDS[b][1] for b in band_names])[which]
    return rng.integers(lo, hi + 1)


def payload_plan(design, n):
    """{detector: (n,) payload bytes}."""
    bands = DESIGNS[design]["bands"]
    if design == "full":
        # v3: one draw shared by all detectors (reproduces models_adv/ exactly)
        p = sample_payloads(np.random.default_rng(AUG_SEED), n, bands["xgboost"])
        return {d: p for d in DETECTORS}
    return {d: sample_payloads(np.random.default_rng([AUG_SEED, i]), n, bands[d])
            for i, d in enumerate(DETECTORS)}


class AugmentedData:
    """Scaled original training split + per-detector padded PortScan rows."""

    def __init__(self, design, mask_features=()):
        df = load_and_clean_data(DATA_PATH)
        self.le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        y = self.le.transform(df["Label"])
        X = df.drop(columns=["Label"]).astype(np.float32)
        del df
        X_train, _, self.y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train = X_train.to_numpy(np.float32)
        del X

        self.ps_idx = list(self.le.classes_).index("PortScan")
        eligible = X_train[(self.y_train == self.ps_idx) & is_single_probe(X_train)]
        self.base = np.repeat(eligible, AUG_COPIES, axis=0)
        self.plan = payload_plan(design, len(self.base))

        self.scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        if mask_features:
            self.scaler = MaskedScaler.from_fitted(self.scaler, mask_features)
        # Scaling is per row, so scaling the split once and the padded rows per detector
        # gives exactly the same matrix as scaling their concatenation.
        self.X_train_s = self._scale(X_train)
        self.meta = {
            "design": design,
            "bands_bytes": {b: list(BANDS[b]) for b in sorted({b for bs in DESIGNS[design]["bands"].values() for b in bs})},
            "detector_bands": DESIGNS[design]["bands"],
            "train_rows_original": int(len(X_train)),
            "portscan_train_rows_single_probe": int(len(eligible)),
            "padded_rows_added_per_detector": int(len(self.base)),
            "payload_bytes_per_detector": {
                d: {"min": int(p.min()), "max": int(p.max()), "mean": round(float(p.mean()), 1),
                    "rows_per_band": {b: int(((p >= BANDS[b][0]) & (p <= BANDS[b][1])).sum())
                                      for b in DESIGNS[design]["bands"][d]}}
                for d, p in self.plan.items()},
            "aug_copies": AUG_COPIES, "aug_seed": AUG_SEED,
            "masked_features": list(mask_features),
        }

    def _scale(self, A):
        return self.scaler.transform(pd.DataFrame(A, columns=self.scaler.feature_names_in_)).astype(np.float32)

    def for_detector(self, name):
        padded = self._scale(set_forward_payload(self.base, self.plan[name]))
        return (np.concatenate([self.X_train_s, padded]),
                np.concatenate([self.y_train, np.full(len(padded), self.ps_idx)]))


def copy_shared_artifacts(out_dir, mask_features=()):
    os.makedirs(out_dir, exist_ok=True)
    for name in ("scaler.pkl", "label_encoder.pkl", "cnn_arch.json"):
        shutil.copy2(os.path.join(MODEL_DIR, name), os.path.join(out_dir, name))
    if mask_features:
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(MaskedScaler.from_fitted(scaler, mask_features), os.path.join(out_dir, "scaler.pkl"))


def train_trees(data, out_dir, design, model_seed):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    with open(os.path.join(out_dir, "augmentation.json"), "w") as f:
        json.dump({**data.meta, "model_seed": model_seed}, f, indent=2)

    t = time.time()
    X, y = data.for_detector("rf")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=model_seed)
    rf.fit(X, y)
    joblib.dump(rf, os.path.join(out_dir, "rf_model.pkl"))
    del rf, X, y
    rf_min = (time.time() - t) / 60
    print(f"RF trained in {rf_min:.1f} min", flush=True)

    t = time.time()
    X, y = data.for_detector("xgboost")
    xgb = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=model_seed)
    xgb.fit(X, y)
    joblib.dump(xgb, os.path.join(out_dir, "xgboost_model.pkl"))
    xgb_min = (time.time() - t) / 60
    print(f"XGBoost trained in {xgb_min:.1f} min", flush=True)

    m = data.meta
    log_event(f"Adversarial training ({design}, model seed {model_seed}): trees finished (train_adversarial.py)",
              [f"bands per detector: {m['detector_bands']}  ({m['bands_bytes']} bytes)",
               f"padded PortScan rows per detector: {m['padded_rows_added_per_detector']:,} "
               f"(seed {AUG_SEED}) added to {m['train_rows_original']:,} train rows",
               f"RF {rf_min:.1f} min, XGBoost {xgb_min:.1f} min; saved to {out_dir}"])


def train_cnn_stage(data, out_dir, design, model_seed):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import accuracy_score
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from cnn_arch import IDS1DCNN
    from train_cnn import BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, FC_DIMS, HIDDEN_DIMS, LR

    n_classes = len(data.le.classes_)
    X, y = data.for_detector("cnn")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.05, stratify=y, random_state=0)
    del X, y

    # Same class-balanced sampling as train_cnn.py
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    weights = torch.tensor((1.0 / (counts + 1e-8))[y_tr], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))),
                          batch_size=BATCH_SIZE, sampler=sampler)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
                        batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(model_seed)  # weight init and the class-balanced sampler's draws
    model = IDS1DCNN(n_features=X_tr.shape[1], n_classes=n_classes, hidden_dims=HIDDEN_DIMS,
                     fc_dims=FC_DIMS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    save_path = os.path.join(out_dir, "cnn_model.pth")

    best, t0 = 0.0, time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        preds = []
        with torch.no_grad():
            for Xb, _ in val_dl:
                preds.append(model(Xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
        val_acc = accuracy_score(y_val, np.concatenate(preds))
        scheduler.step(val_acc)
        marker = ""
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), save_path)
            marker = "  <- best"
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={total / len(train_dl):.4f}  "
              f"val_acc={val_acc:.4f}  ({(time.time() - t0) / 60:.1f} min){marker}", flush=True)

    log_event(f"Adversarial training ({design}, model seed {model_seed}): CNN finished (train_adversarial.py)",
              [f"bands: {DESIGNS[design]['bands']['cnn']}; {EPOCHS} epochs on {DEVICE}, "
               f"{(time.time() - t0) / 60:.1f} min; best validation accuracy {best:.4f} "
               f"(5% slice of augmented train split)",
               f"saved {save_path}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", choices=list(DESIGNS), default="full")
    parser.add_argument("--stage", choices=["trees", "cnn"], required=True)
    parser.add_argument("--model-seed", type=int, default=None,
                        help=f"Training seed (default {MODEL_SEED}); writes to <out>_ms<seed>/")
    parser.add_argument("--mask-feature", nargs="+", default=[], metavar="NAME",
                        help="Feature column(s) to remove (scaler outputs 0); writes to <out>_nopsh[_msS]/")
    args = parser.parse_args()

    model_seed = MODEL_SEED if args.model_seed is None else args.model_seed
    mask = tuple(args.mask_feature)
    out_name = (DESIGNS[args.design]["out"] + ("_nopsh" if mask else "")
                + ("" if args.model_seed is None else f"_ms{args.model_seed}"))
    out_dir = os.path.join(CODE_DIR, out_name)
    copy_shared_artifacts(out_dir, mask)
    data = AugmentedData(args.design, mask)
    print(json.dumps(data.meta, indent=2), flush=True)
    log_event(f"Adversarial training ({args.design}, model seed {model_seed}) started "
              f"(train_adversarial.py --stage {args.stage})",
              [f"bands per detector: {data.meta['detector_bands']}",
               f"{data.meta['padded_rows_added_per_detector']:,} padded PortScan rows per detector "
               f"+ {data.meta['train_rows_original']:,} original train rows; same split, scaler, "
               f"hyperparameters; output {out_dir}"])
    if args.stage == "trees":
        train_trees(data, out_dir, args.design, model_seed)
    else:
        train_cnn_stage(data, out_dir, args.design, model_seed)
