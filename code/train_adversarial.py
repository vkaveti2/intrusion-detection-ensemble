"""
train_adversarial.py — Adversarial training against payload padding.

v2 showed every detector learned "zero payload => scan": all CIC-IDS2017
PortScan flows carry 0–2 bytes, so a padded probe looks benign. This retrains
the same three detectors on the same training split, plus payload-padded
copies of the training-split PortScan flows.

Controlled comparison with the original detectors:
  - identical split (random_state=42), scaler and label encoder (copied)
  - identical model hyperparameters (train_models.py / train_cnn.py)
  - the only change is the added padded scan flows
  - padding sizes drawn uniformly from 1..1460 bytes (TCP MSS), NOT the
    agent's discrete sizes, so the defence isn't trained on the test attack
  - padded features computed with flow_features.set_forward_payload, the
    same validated CICFlowMeter arithmetic the simulator uses
  - test split and probe bank (held-out) are untouched

One deliberate difference: the CNN's best epoch is chosen on a validation
slice of the training data, not on the test split as train_cnn.py does.

Run in two processes (xgboost and torch each bundle libomp on macOS):
    python train_adversarial.py --stage trees
    python train_adversarial.py --stage cnn
Outputs: models_adv/
"""

import argparse
import json
import os
import shutil
import time

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from build_probe_bank import DATA_PATH, MODEL_DIR, load_and_clean_data
from flow_features import MAX_PAYLOAD, is_single_probe, set_forward_payload
from project_log import log_event

ADV_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_adv")
AUG_COPIES = 1      # padded copies per eligible training-split PortScan flow
AUG_SEED   = 2026


def build_training_set():
    """Original training split + padded PortScan copies, scaled with the original scaler."""
    df = load_and_clean_data(DATA_PATH)
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    y = le.transform(df["Label"])
    X = df.drop(columns=["Label"]).astype(np.float32)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train = X_train.to_numpy(np.float32)

    ps_idx = list(le.classes_).index("PortScan")
    eligible = X_train[(y_train == ps_idx) & is_single_probe(X_train)]
    rng = np.random.default_rng(AUG_SEED)
    base = np.repeat(eligible, AUG_COPIES, axis=0)
    payload = rng.integers(1, MAX_PAYLOAD + 1, size=len(base))
    padded = set_forward_payload(base, payload)

    X_aug = np.concatenate([X_train, padded])
    y_aug = np.concatenate([y_train, np.full(len(padded), ps_idx)])

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    import pandas as pd
    X_aug = scaler.transform(pd.DataFrame(X_aug, columns=scaler.feature_names_in_)).astype(np.float32)
    meta = {
        "train_rows_original": int(len(X_train)),
        "portscan_train_rows_single_probe": int(len(eligible)),
        "padded_rows_added": int(len(padded)),
        "payload_range_bytes": [1, MAX_PAYLOAD],
        "aug_copies": AUG_COPIES, "aug_seed": AUG_SEED,
    }
    return X_aug, y_aug, le, meta


def copy_shared_artifacts():
    os.makedirs(ADV_DIR, exist_ok=True)
    for name in ("scaler.pkl", "label_encoder.pkl", "cnn_arch.json"):
        shutil.copy2(os.path.join(MODEL_DIR, name), os.path.join(ADV_DIR, name))


def train_trees(X, y, meta):
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    t = time.time()
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    joblib.dump(rf, os.path.join(ADV_DIR, "rf_model.pkl"))
    rf_min = (time.time() - t) / 60
    print(f"RF trained in {rf_min:.1f} min", flush=True)

    t = time.time()
    xgb = XGBClassifier(tree_method="hist", n_jobs=-1, random_state=42)
    xgb.fit(X, y)
    joblib.dump(xgb, os.path.join(ADV_DIR, "xgboost_model.pkl"))
    xgb_min = (time.time() - t) / 60
    print(f"XGBoost trained in {xgb_min:.1f} min", flush=True)

    with open(os.path.join(ADV_DIR, "augmentation.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log_event("Adversarial training: trees finished (train_adversarial.py --stage trees)",
              [f"padded PortScan rows added: {meta['padded_rows_added']:,} "
               f"(payload 1–{MAX_PAYLOAD} B, seed {AUG_SEED}) to {meta['train_rows_original']:,} train rows",
               f"RF {rf_min:.1f} min, XGBoost {xgb_min:.1f} min; saved to {ADV_DIR}"])


def train_cnn_stage(X, y, n_classes):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import accuracy_score
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    from cnn_arch import IDS1DCNN
    from train_cnn import BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, FC_DIMS, HIDDEN_DIMS, LR

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.05, stratify=y, random_state=0)

    # Same class-balanced sampling as train_cnn.py
    counts = np.bincount(y_tr, minlength=n_classes).astype(np.float64)
    weights = torch.tensor((1.0 / (counts + 1e-8))[y_tr], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.int64))),
                          batch_size=BATCH_SIZE, sampler=sampler)
    val_dl = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.int64))),
                        batch_size=BATCH_SIZE, shuffle=False)

    torch.manual_seed(42)
    model = IDS1DCNN(n_features=X.shape[1], n_classes=n_classes, hidden_dims=HIDDEN_DIMS,
                     fc_dims=FC_DIMS, dropout=DROPOUT).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    save_path = os.path.join(ADV_DIR, "cnn_model.pth")

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

    log_event("Adversarial training: CNN finished (train_adversarial.py --stage cnn)",
              [f"{EPOCHS} epochs on {DEVICE}, {(time.time() - t0) / 60:.1f} min; "
               f"best validation accuracy {best:.4f} (5% slice of augmented train split)",
               f"saved {save_path}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["trees", "cnn"], required=True)
    args = parser.parse_args()

    copy_shared_artifacts()
    X, y, le, meta = build_training_set()
    print(json.dumps(meta, indent=2), flush=True)
    if args.stage == "trees":
        log_event("Adversarial training started (train_adversarial.py --stage trees)",
                  [f"{meta['padded_rows_added']:,} padded PortScan rows (payload 1–{MAX_PAYLOAD} B) "
                   f"+ {meta['train_rows_original']:,} original train rows; same split, scaler, hyperparameters"])
        train_trees(X, y, meta)
    else:
        log_event("Adversarial training started (train_adversarial.py --stage cnn)")
        train_cnn_stage(X, y, len(le.classes_))
