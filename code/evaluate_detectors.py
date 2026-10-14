"""
evaluate_detectors.py — Held-out performance of each standalone detector.

Reproduces train_models.py's exact split and evaluates XGBoost, Random Forest
and the 1D-CNN on the 20% test set, both as 15-class classifiers and as the
binary attack-vs-BENIGN decision ensemble.py actually uses.

Also measures robustness to payload padding without any RL agent: the
held-out PortScan probe bank padded to each of the agent's payload sizes and
to random sizes, scored by each detector.

Detector set: $IDS_MODEL_DIR (default code/models).
Outputs (in that directory):
  detector_eval.json     — metrics per detector (thesis table)
  ensemble_weights.json  — confidence-weighted voting weights ∝ held-out binary F1
"""

import json
import os
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from build_probe_bank import DATA_PATH, load_and_clean_data
from cnn_arch import IDS1DCNN
from ensemble import MODEL_DIR
from flow_features import MAX_PAYLOAD, set_forward_payload
from ids_env import PKTSIZE_OPTIONS, get_probe_bank
from project_log import log_event

# GPU for the CNN pass: on CPU it must stay single-threaded (libomp clash with xgboost)
CNN_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def cnn_proba(Xs):
    with open(os.path.join(MODEL_DIR, "cnn_arch.json")) as f:
        net = IDS1DCNN(**json.load(f))
    net.load_state_dict(torch.load(os.path.join(MODEL_DIR, "cnn_model.pth"),
                                   map_location="cpu", weights_only=True))
    net.to(CNN_DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            x = torch.from_numpy(Xs[i:i + 8192]).to(CNN_DEVICE)
            out.append(torch.softmax(net(x), dim=1).cpu().numpy())
    return np.concatenate(out)


def padded_probe_sets(seed=0):
    """Held-out PortScan probes at each agent payload size, plus random sizes."""
    bank = get_probe_bank()
    sets = {f"{s}B": set_forward_payload(bank, s) for s in PKTSIZE_OPTIONS}
    rng = np.random.default_rng(seed)
    sets[f"random 1-{MAX_PAYLOAD}B"] = set_forward_payload(
        bank, rng.integers(1, MAX_PAYLOAD + 1, size=len(bank)))
    return sets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--update-weights", action="store_true",
                        help="Overwrite an existing ensemble_weights.json")
    args = parser.parse_args()

    df = load_and_clean_data(DATA_PATH)
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    classes = list(le.classes_)
    y = le.transform(df["Label"])
    X = df.drop(columns=["Label"]).astype(np.float32)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    Xs = scaler.transform(X_test).astype(np.float32)
    portscan = y_test == classes.index("PortScan")
    attack_true = y_test != classes.index("BENIGN")

    padded = {k: scaler.transform(pd.DataFrame(v, columns=X.columns)).astype(np.float32)
              for k, v in padded_probe_sets().items()}

    results = {}
    for name in ["xgboost", "rf", "cnn"]:
        if name == "cnn":
            proba = cnn_proba
        else:
            model = joblib.load(os.path.join(MODEL_DIR, f"{name}_model.pkl"))
            model.set_params(n_jobs=-1)
            proba = model.predict_proba
        P = proba(Xs)

        pred = P.argmax(axis=1)
        attack_pred = P[:, 0] < 0.5  # ensemble.py: p_scanning = 1 - P(BENIGN) >= 0.5
        tp = int((attack_pred & attack_true).sum())
        fp = int((attack_pred & ~attack_true).sum())
        fn = int((~attack_pred & attack_true).sum())
        tn = int((~attack_pred & ~attack_true).sum())
        precision, recall = tp / (tp + fp), tp / (tp + fn)
        results[name] = {
            "multiclass_accuracy": float((pred == y_test).mean()),
            "binary_precision": precision,
            "binary_recall": recall,
            "binary_f1": 2 * precision * recall / (precision + recall),
            "binary_false_positive_rate": fp / (fp + tn),
            "portscan_detected": float(attack_pred[portscan].mean()),
            "padded_probe_detected": {k: float((proba(v)[:, 0] < 0.5).mean())
                                      for k, v in padded.items()},
        }
        print(f"\n── {name} ──")
        print(json.dumps(results[name], indent=2))
        print(classification_report(y_test, pred, labels=list(range(len(classes))),
                                    target_names=classes, digits=4, zero_division=0))

    with open(os.path.join(MODEL_DIR, "detector_eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    f1 = np.array([results[n]["binary_f1"] for n in ["xgboost", "rf", "cnn"]])
    weights = dict(zip(["xgboost", "rf", "cnn"], (f1 / f1.sum()).round(6).tolist()))
    weights_path = os.path.join(MODEL_DIR, "ensemble_weights.json")
    if os.path.exists(weights_path) and not args.update_weights:
        # Keep the weights trained agents were run against (reproducibility)
        with open(weights_path) as f:
            kept = json.load(f)
        print(f"\nKept existing weights {kept} (recomputed: {weights}); --update-weights to replace")
        weights = kept
    else:
        with open(weights_path, "w") as f:
            json.dump(weights, f, indent=2)
        print(f"\nWeighted-vote weights → {weights}")

    lines = [f"detector set: {MODEL_DIR}",
             f"held-out test split: {len(y_test):,} rows",
             "detector  accuracy  attack-recall  false-pos  PortScan-caught"]
    for n, r in results.items():
        lines.append(f"{n:8s}  {r['multiclass_accuracy']:7.2%}   {r['binary_recall']:7.2%}      "
                     f"{r['binary_false_positive_rate']:6.2%}    {r['portscan_detected']:7.2%}")
    lines.append(f"padded held-out scan probes caught ({len(get_probe_bank()):,} flows each):")
    lines.append("detector  " + "  ".join(f"{k:>10s}" for k in padded))
    for n, r in results.items():
        lines.append(f"{n:8s}  " + "  ".join(f"{v:10.1%}" for v in r["padded_probe_detected"].values()))
    lines.append(f"weighted-vote weights: {weights}")
    log_event("Detector evaluation (evaluate_detectors.py)", lines)
    print("\n" + "\n".join(lines))
