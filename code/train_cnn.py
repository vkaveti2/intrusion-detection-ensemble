"""
train_cnn.py — 1D-CNN Detector Training on CIC-IDS 2017

Trains a 1D Convolutional Neural Network on the CIC-IDS 2017 dataset as
the third constituent detector for the thesis ensemble IDS.

Design decisions:
  - Reuses the *same* scaler.pkl and label_encoder.pkl already fit by
    train_models.py, guaranteeing that XGBoost, RF, and CNN all operate
    in an identical feature space.
  - WeightedRandomSampler corrects the severe class imbalance in CICIDS-2017
    (BENIGN accounts for ~80% of samples) so the CNN does not collapse to
    always predicting BENIGN.
  - MPS (Metal Performance Shaders) is used automatically if available
    (Mac M-series), otherwise falls back to CPU.
  - Best model (highest val accuracy) is checkpointed; training does not
    save overfit end-of-run weights.

Outputs:
  models/cnn_model.pth   — PyTorch state dict (best checkpoint)
  models/cnn_arch.json   — Architecture hyperparameters for reconstruction
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from cnn_arch import IDS1DCNN

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "data", "*.csv")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPOCHS      = 30
BATCH_SIZE  = 512
LR          = 1e-3
HIDDEN_DIMS = [64, 128, 128]    # Conv1d channel sizes across 3 blocks
FC_DIMS     = [128, 64]         # FC hidden sizes before output layer
DROPOUT     = 0.3

# Mac M-series MPS GPU → else CPU (no CUDA on Mac)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ── Data Loading ───────────────────────────────────────────────────────────────
def load_and_clean_data(file_pattern: str) -> pd.DataFrame:
    """
    Loads and cleans all CICIDS-2017 CSVs.
    Mirrors the exact preprocessing in train_models.py for consistency.
    """
    print("Loading CICIDS-2017 CSVs (this may take a minute)...")
    all_files = glob.glob(file_pattern)
    if not all_files:
        raise FileNotFoundError(f"No CSV files found at: {file_pattern}")

    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

    # Strip whitespace from column names (Kaggle version quirk)
    df.columns = df.columns.str.strip()

    # Drop duplicate Fwd Header Length column present in some Kaggle exports
    if "Fwd Header Length.1" in df.columns:
        df.drop(columns=["Fwd Header Length.1"], inplace=True)

    # Replace infinities, then drop NaN rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {before - len(df)} duplicates. Rows remaining: {len(df):,}")
    return df


def main():
    df = load_and_clean_data(DATA_PATH)

    # ── Reuse the saved LabelEncoder (must match RF/XGBoost class mapping) ────
    le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    if not os.path.exists(le_path):
        raise FileNotFoundError(
            "label_encoder.pkl not found. Run train_models.py first."
        )
    le = joblib.load(le_path)
    df["Label"] = le.transform(df["Label"])

    X = df.drop(columns=["Label"]).astype(np.float32)
    y = df["Label"].values

    n_features = X.shape[1]
    n_classes  = len(le.classes_)
    print(f"\nFeatures: {n_features}  |  Classes: {n_classes}  |  Device: {DEVICE}")
    benign_idx = list(le.classes_).index("BENIGN")
    print(f"BENIGN class index: {benign_idx}  (p_scanning = 1 - probs[{benign_idx}])")

    # ── Reuse the saved StandardScaler (same feature space as RF/XGBoost) ─────
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            "scaler.pkl not found. Run train_models.py first."
        )
    scaler = joblib.load(scaler_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Transform via DataFrame so scaler's feature_names_in_ validation passes
    if hasattr(scaler, "feature_names_in_"):
        cols = scaler.feature_names_in_
        X_train = scaler.transform(
            pd.DataFrame(X_train.values, columns=cols)
        ).astype(np.float32)
        X_test = scaler.transform(
            pd.DataFrame(X_test.values, columns=cols)
        ).astype(np.float32)
    else:
        X_train = scaler.transform(X_train).astype(np.float32)
        X_test  = scaler.transform(X_test).astype(np.float32)

    # ── Weighted sampler to handle class imbalance ────────────────────────────
    class_counts   = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    class_weights  = 1.0 / (class_counts + 1e-8)
    sample_weights = torch.tensor(
        [class_weights[c] for c in y_train], dtype=torch.float32
    )
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_ds = TensorDataset(
        torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test), torch.tensor(y_test, dtype=torch.long)
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

    # ── Build model ───────────────────────────────────────────────────────────
    model = IDS1DCNN(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dims=HIDDEN_DIMS,
        fc_dims=FC_DIMS,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=4, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    model_save_path = os.path.join(MODEL_DIR, "cnn_model.pth")

    print(f"\n{'='*65}")
    print(f" Training IDS1DCNN  |  {EPOCHS} epochs  |  batch={BATCH_SIZE}")
    print(f"{'='*65}")

    for epoch in range(1, EPOCHS + 1):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for Xb, yb in train_dl:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for Xb, yb in test_dl:
                preds = model(Xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(yb.numpy())

        val_acc   = accuracy_score(all_true, all_preds)
        avg_loss  = total_loss / len(train_dl)
        scheduler.step(val_acc)

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            marker = "  ← best ✓"

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"loss={avg_loss:.4f}  "
            f"val_acc={val_acc:.4f}{marker}"
        )

    # ── Save architecture config (needed for reconstruction in ensemble.py) ───
    arch_config = {
        "n_features":  n_features,
        "n_classes":   n_classes,
        "hidden_dims": HIDDEN_DIMS,
        "fc_dims":     FC_DIMS,
        "dropout":     DROPOUT,
    }
    arch_path = os.path.join(MODEL_DIR, "cnn_arch.json")
    with open(arch_path, "w") as f:
        json.dump(arch_config, f, indent=2)

    # ── Final evaluation ──────────────────────────────────────────────────────
    # Reload best checkpoint for report
    model.load_state_dict(
        torch.load(model_save_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    final_preds, final_true = [], []
    with torch.no_grad():
        for Xb, yb in test_dl:
            preds = model(Xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            final_preds.extend(preds)
            final_true.extend(yb.numpy())

    print(f"\n{'='*65}")
    print(f" Training complete.  Best val accuracy: {best_val_acc:.4f}")
    print(f"{'='*65}")
    print(classification_report(
        final_true, final_preds,
        target_names=le.classes_,
        zero_division=0,
    ))
    print(f"\nSaved: {model_save_path}")
    print(f"Saved: {arch_path}")
    print("\nStep 1.5 (CNN training) COMPLETE.")
    print("Run train_ensemble.py to launch the full DDQN experiment matrix.")


if __name__ == "__main__":
    main()
