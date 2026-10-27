"""
psh_sensitivity.py — Do the detectors rely on the PSH flag columns?

Design and decision rule (fixed before running): docs/09_psh_diagnostic_design.md.

Inference-only interventions, no retraining, for the detector set in $IDS_MODEL_DIR:
  attack  — padded held-out PortScan probes (simulator ports), PSH column set to 0
  benign  — held-out BENIGN rows, PSH column forced to 1 (the value scans carry)
  ref     — both unchanged; must match rule_eval.json / detector_eval.json
Columns tested: "PSH Flag Count" and "Fwd PSH Flags".

Output: psh_sensitivity.json in the detector-set directory.
"""

import json
import os
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from build_probe_bank import DATA_PATH, load_and_clean_data
from ensemble import MODEL_DIR, VOTING_RULES, load_weights
from evaluate_detectors import (CNN_DEVICE, DETECTORS, cnn_proba, padded_probe_sets,
                                rule_flags)  # noqa: F401  (CNN_DEVICE kept for parity)
from project_log import log_event

COLUMNS = ["PSH Flag Count", "Fwd PSH Flags"]
ATTACK_THRESHOLD_PP = 5.0    # detection drop that counts as substantial dependence
BENIGN_THRESHOLD_PP = 0.5    # false-positive rise that counts as substantial dependence


def attack_probability(name, X_scaled):
    if name == "cnn":
        return 1.0 - cnn_proba(X_scaled)[:, 0]
    model = joblib.load(os.path.join(MODEL_DIR, f"{name}_model.pkl"))
    model.set_params(n_jobs=1 if name == "rf" else -1)
    return 1.0 - model.predict_proba(X_scaled)[:, 0]


def rates(sets, weights):
    """sets: {label: {detector: p_attack}} -> {label: {rule: flag rate}}."""
    return {k: {r: float(f.mean()) for r, f in
                rule_flags(np.stack([p[n] for n in DETECTORS]), weights).items()}
            for k, p in sets.items()}


if __name__ == "__main__":
    df = load_and_clean_data(DATA_PATH)
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    classes = list(le.classes_)
    y = le.transform(df["Label"])
    X = df.drop(columns=["Label"]).astype(np.float32)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    cols = list(X.columns)
    weights = load_weights()

    benign = X_test[y_test == classes.index("BENIGN")].to_numpy(np.float32)
    probes = padded_probe_sets(sim_ports=True)           # {"64B": (n, 77) ...}
    scale = lambda A: scaler.transform(pd.DataFrame(A, columns=cols)).astype(np.float32)

    report = {"detector_set": MODEL_DIR, "weights": weights.tolist(), "columns": {}}
    for col in COLUMNS:
        j = cols.index(col)
        attack_cases = {"ref": 0, "psh0": 1}      # 1 = apply the intervention
        benign_cases = {"ref": 0, "psh1": 1}
        p_attack = {k: {} for k in ("ref", "psh0")}
        for size, A in probes.items():
            for case, on in attack_cases.items():
                B = A.copy()
                if on:
                    B[:, j] = 0.0
                p_attack[case][size] = scale(B)
        p_benign = {}
        for case, on in benign_cases.items():
            B = benign.copy()
            if on:
                B[:, j] = 1.0
            p_benign[case] = scale(B)

        probs_attack = {c: {s: {} for s in probes} for c in p_attack}
        probs_benign = {c: {} for c in p_benign}
        for name in DETECTORS:
            for c in p_attack:
                for s in probes:
                    probs_attack[c][s][name] = attack_probability(name, p_attack[c][s])
            for c in p_benign:
                probs_benign[c][name] = attack_probability(name, p_benign[c])

        attack = {c: rates(probs_attack[c], weights) for c in probs_attack}
        benign_fp = rates(probs_benign, weights)
        per_rule = {}
        for r in VOTING_RULES:
            det_drop = {s: 100 * (attack["ref"][s][r] - attack["psh0"][s][r]) for s in probes}
            fp_rise = 100 * (benign_fp["psh1"][r] - benign_fp["ref"][r])
            nonzero_drop = max(v for s, v in det_drop.items() if s != "0B")
            per_rule[r] = {
                "detection_ref": {s: attack["ref"][s][r] for s in probes},
                "detection_psh0": {s: attack["psh0"][s][r] for s in probes},
                "detection_drop_pp": det_drop,
                "benign_fp_ref": benign_fp["ref"][r],
                "benign_fp_psh1": benign_fp["psh1"][r],
                "benign_fp_rise_pp": fp_rise,
                "substantial": bool(nonzero_drop > ATTACK_THRESHOLD_PP or fp_rise > BENIGN_THRESHOLD_PP),
            }
        report["columns"][col] = per_rule

    out = os.path.join(MODEL_DIR, "psh_sensitivity.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    lines = [f"detector set: {MODEL_DIR}  ({len(benign):,} benign rows, "
             f"{len(next(iter(probes.values()))):,} probes per size)",
             f"substantial = padded-probe detection drop > {ATTACK_THRESHOLD_PP}pp (non-zero payloads) "
             f"or benign FP rise > {BENIGN_THRESHOLD_PP}pp"]
    for col, per_rule in report["columns"].items():
        lines.append(f"[{col}]  rule       worst det. drop (pp)   benign FP ref -> PSH=1   substantial")
        for r, m in per_rule.items():
            worst = max(v for s, v in m["detection_drop_pp"].items() if s != "0B")
            lines.append(f"  {r:9s}  {worst:8.2f}              {m['benign_fp_ref']:.3%} -> "
                         f"{m['benign_fp_psh1']:.3%}      {'YES' if m['substantial'] else 'no'}")
    print("\n".join(lines))
    log_event("PSH sensitivity (psh_sensitivity.py)", lines + [f"output: {out}"])
