"""
build_probe_bank.py — Real CIC-IDS2017 PortScan flows for the simulated probe.

Reproduces the exact load/clean/split from train_models.py and keeps the
PortScan flows from the *held-out test split* (never seen by any detector)
that have one forward packet and at most one backward packet, i.e. the
single-probe shape that ids_env._simulate_probe models.

Output: models/probe_bank.npz  (flows: (N, 77) raw float32, feature_names)
"""

import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "data", "*.csv")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "models")


def load_and_clean_data(file_pattern):
    # Identical to train_models.py so the split matches the detectors' split.
    df = pd.concat([pd.read_csv(f) for f in glob.glob(file_pattern)], ignore_index=True)
    df.columns = df.columns.str.strip()
    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


if __name__ == "__main__":
    df = load_and_clean_data(DATA_PATH)
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    y = le.transform(df["Label"])
    X = df.drop(columns=["Label"]).astype(np.float32)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Guard: the scaler must have been fit on this exact feature order.
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    assert list(scaler.feature_names_in_) == list(X.columns), "feature order mismatch"

    scans = X_test[y_test == list(le.classes_).index("PortScan")]
    single = scans[(scans["Total Fwd Packets"] == 1) & (scans["Total Backward Packets"] <= 1)]

    out = os.path.join(MODEL_DIR, "probe_bank.npz")
    np.savez(out, flows=single.to_numpy(np.float32), feature_names=np.array(X.columns))
    print(f"Held-out PortScan flows: {len(scans):,}  |  single-probe shape: {len(single):,}")
    print(f"Saved → {out}")
