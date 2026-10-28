"""
train_v5.py — detectors under the corrected protocol (docs/07_split_plan.md, splits.py).

Replaces train_models.py / train_cnn.py / train_adversarial.py for every reported number. Those
scripts stay as the record of the original pipeline (their test-split use is documented).

  - rows come from splits.py; detectors see the train partition only
  - StandardScaler fitted on train rows only, wrapped in MaskedScaler (PSH Flag Count removed)
  - padded copies are generated after the split from train-partition probe flows only, and keep
    their source flow's inner-validation side
  - CNN: best epoch AND the ReduceLROnPlateau scheduler use the inner validation rows (5% of
    train, by group); trees train on every train row, since they have no epoch selection
  - nothing here reads the validation or final-test partitions

Designs (same hardening bands as train_adversarial.py):
  cov0 unhardened   cov1 one band per detector   cov2 two bands   cov3 full 1-1460 B
Output: models_v5/<design>[_ms<seed>]/    (xgboost_model.pkl, rf_model.pkl, cnn_model.pth, scaler.pkl, ...)

    python train_v5.py --design cov1 --stage trees [--model-seed 1]
    python train_v5.py --design cov1 --stage cnn   [--model-seed 1]
"""

import argparse
import json
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import splits
from flow_features import set_forward_payload
from masked_scaler import MaskedScaler
from project_log import log_event
from train_adversarial import AUG_SEED, BANDS, DETECTORS, DESIGNS, MODEL_SEED, payload_plan

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(CODE_DIR, "models_v5")
PSH      = (splits.PSH_COLUMN,)
# v5 name -> train_adversarial design (None = no padding)
V5_DESIGNS = {"cov0": None, "cov1": "cov1", "cov2": "cov2", "cov3": "full"}


class V5Data:
    """Scaled train partition + per-detector padded PortScan rows, with the inner-validation side."""

    def __init__(self, design):
        self.ds = splits.load_dataset()
        self.sp = splits.load_splits(self.ds)
        ds, sp = self.ds, self.sp
        raw = splits.fit_scaler(ds, sp)
        self.scaler = MaskedScaler.from_fitted(raw, PSH)
        self.classes = ds.classes
        self.ps_idx = ds.classes.index("PortScan")

        train = sp.part == 0
        self.X = self._scale(ds.X[train])
        self.y = ds.y[train].astype(np.int64)
        self.inner = sp.inner_val[train]

        self.key = V5_DESIGNS[design]
        bank_rows = np.where(sp.is_bank & train)[0]
        self.base = ds.X[bank_rows]
        self.base_inner = sp.inner_val[bank_rows]
        self.plan = payload_plan(self.key, len(self.base)) if self.key else None
        self.meta = {
            "design": design, "protocol": "v5", "masked_features": list(PSH),
            "train_rows": int(train.sum()), "inner_validation_rows": int(self.inner.sum()),
            "padded_rows_per_detector": int(len(self.base)) if self.plan else 0,
            "padded_rows_inner_validation": int(self.base_inner.sum()) if self.plan else 0,
            "detector_bands": DESIGNS[self.key]["bands"] if self.key else None,
            "aug_seed": AUG_SEED, "fingerprint": sp.meta["fingerprint"],
            "scaler_fit_rows": "train partition only",
        }
        if self.plan:
            self.meta["payload_bytes_per_detector"] = {
                d: {"min": int(p.min()), "max": int(p.max()), "mean": round(float(p.mean()), 1)}
                for d, p in self.plan.items()}

    def _scale(self, A):
        return self.scaler.transform(pd.DataFrame(A, columns=self.ds.columns)).astype(np.float32)

    def for_detector(self, name):
        """(X, y, inner_flag): train rows, then this detector's padded copies of train probe flows."""
        if not self.plan:
            return self.X, self.y, self.inner
        padded = self._scale(set_forward_payload(self.base, self.plan[name]))
        return (np.concatenate([self.X, padded]),
                np.concatenate([self.y, np.full(len(padded), self.ps_idx, np.int64)]),
                np.concatenate([self.inner, self.base_inner]))


def prepare_dir(out_dir, data):
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(os.path.join(CODE_DIR, "models", "cnn_arch.json"), os.path.join(out_dir, "cnn_arch.json"))
    joblib.dump(data.scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(LabelEncoder().fit(data.classes), os.path.join(out_dir, "label_encoder.pkl"))


def train_trees(data, out_dir, tag, model_seed):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    with open(os.path.join(out_dir, "augmentation.json"), "w") as f:
        json.dump({**data.meta, "model_seed": model_seed}, f, indent=2)
    t = time.time()
    X, y, _ = data.for_detector("rf")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=model_seed).fit(X, y)
    joblib.dump(rf, os.path.join(out_dir, "rf_model.pkl"))
    del rf, X, y
    rf_min = (time.time() - t) / 60
    t = time.time()
    X, y, _ = data.for_detector("xgboost")
    xgb = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=model_seed).fit(X, y)
    joblib.dump(xgb, os.path.join(out_dir, "xgboost_model.pkl"))
    xgb_min = (time.time() - t) / 60
    print(f"RF {rf_min:.1f} min, XGBoost {xgb_min:.1f} min", flush=True)
    log_event(f"v5 trees finished ({tag}, model seed {model_seed}) (train_v5.py)",
              [f"{data.meta['train_rows']:,} train rows + {data.meta['padded_rows_per_detector']:,} padded; "
               f"RF {rf_min:.1f} min, XGBoost {xgb_min:.1f} min; saved to {out_dir}"])


def train_cnn_stage(data, out_dir, tag, model_seed, epochs):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import accuracy_score
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from cnn_arch import IDS1DCNN
    from train_cnn import BATCH_SIZE, DEVICE, DROPOUT, FC_DIMS, HIDDEN_DIMS, LR

    X, y, inner = data.for_detector("cnn")
    X_tr, y_tr, X_val, y_val = X[~inner], y[~inner], X[inner], y[inner]
    del X, y
    n_classes = len(data.classes)

    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    weights = torch.tensor((1.0 / (counts + 1e-8))[y_tr], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                          batch_size=BATCH_SIZE, sampler=sampler)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                        batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(model_seed)
    model = IDS1DCNN(n_features=X_tr.shape[1], n_classes=n_classes, hidden_dims=HIDDEN_DIMS,
                     fc_dims=FC_DIMS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    save_path = os.path.join(out_dir, "cnn_model.pth")
    print(f"CNN {tag}: {len(y_tr):,} train rows, {len(y_val):,} inner-validation rows, {epochs} epochs on {DEVICE}",
          flush=True)

    best, best_epoch, t0 = 0.0, 0, time.time()
    for epoch in range(1, epochs + 1):
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
        scheduler.step(val_acc)                        # inner validation, never the test split
        marker = ""
        if val_acc > best:
            best, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), save_path)
            marker = "  <- best"
        print(f"Epoch {epoch:2d}/{epochs}  loss={total / len(train_dl):.4f}  val_acc={val_acc:.4f}  "
              f"({(time.time() - t0) / 60:.1f} min){marker}", flush=True)

    with open(os.path.join(out_dir, "cnn_training.json"), "w") as f:
        json.dump({"best_epoch": best_epoch, "best_inner_val_acc": best, "epochs": epochs, "model_seed": model_seed,
                   "inner_validation_rows": int(len(y_val)), "train_rows": int(len(y_tr))}, f, indent=2)
    log_event(f"v5 CNN finished ({tag}, model seed {model_seed}) (train_v5.py)",
              [f"best epoch {best_epoch}, inner-validation accuracy {best:.4f}; {epochs} epochs on {DEVICE}, "
               f"{(time.time() - t0) / 60:.1f} min; saved {save_path}"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=list(V5_DESIGNS), required=True)
    ap.add_argument("--stage", choices=["trees", "cnn"], required=True)
    ap.add_argument("--model-seed", type=int, default=None, help=f"default {MODEL_SEED}; writes <design>_ms<seed>/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out-root", default=OUT_ROOT)
    args = ap.parse_args()

    seed = MODEL_SEED if args.model_seed is None else args.model_seed
    tag = args.design + ("" if args.model_seed is None else f"_ms{args.model_seed}")
    out_dir = os.path.join(args.out_root, tag)
    data = V5Data(args.design)
    prepare_dir(out_dir, data)
    print(json.dumps(data.meta, indent=2), flush=True)
    if args.stage == "trees":
        train_trees(data, out_dir, tag, seed)
    else:
        train_cnn_stage(data, out_dir, tag, seed, args.epochs)
