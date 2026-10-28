"""
evaluate_v5.py — detector and decision-rule evaluation under the corrected protocol.

Two stages, so no choice is ever made on final-test data:
  --stage weights   reads the VALIDATION partition (and the dev probe bank) only. Computes each
                    detector's binary F1, writes the weighted-vote weights (∝ F1) to
                    ensemble_weights.json, and writes validation_eval.json.
  --stage final     requires the frozen protocol (config/protocol_v5.json "frozen": true). Reads
                    the weights written above and reports every rule on the FINAL-TEST partition
                    and final-test probe bank, with exact counts. Writes final_eval.json.

Padded-probe detection is reported as k/n counts at the simulator's ports (1001-1100), for the
agent's payload sizes and for random sizes 1-1460 B. These are detection rates, a different
quantity from trained-agent evasion rates.

Detector set: $IDS_MODEL_DIR (default code/models_v5/cov1).
    IDS_MODEL_DIR=.../models_v5/cov1 python evaluate_v5.py --stage weights
"""

import argparse
import json
import os
os.environ["OMP_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd

import splits
from ensemble import MODEL_DIR, VOTING_RULES
from evaluate_detectors import DETECTORS, binary_metrics, cnn_proba, flagged, pairwise_diversity, rule_flags
from flow_features import F_DST_PORT, MAX_PAYLOAD, set_forward_payload
from ids_env import EPISODE_LENGTH, PKTSIZE_OPTIONS
from project_log import log_event


def attack_probabilities(X_scaled):
    """{detector: P(attack)} for scaled rows; RF single-threaded to match per-row inference."""
    out = {}
    for name in DETECTORS:
        if name == "cnn":
            out[name] = 1.0 - cnn_proba(X_scaled)[:, 0]
        else:
            model = joblib.load(os.path.join(MODEL_DIR, f"{name}_model.pkl"))
            model.set_params(n_jobs=1 if name == "rf" else -1)
            out[name] = 1.0 - model.predict_proba(X_scaled)[:, 0]
    return out


def padded_sets(bank, seed=0):
    """Bank flows at the simulator's ports, padded to each agent payload size and to random sizes."""
    bank = bank.copy()
    bank[:, F_DST_PORT] = 1001 + np.arange(len(bank)) % EPISODE_LENGTH
    sets = {f"{s}B": set_forward_payload(bank, s) for s in PKTSIZE_OPTIONS}
    rng = np.random.default_rng(seed)
    sets[f"random 1-{MAX_PAYLOAD}B"] = set_forward_payload(bank, rng.integers(1, MAX_PAYLOAD + 1, size=len(bank)))
    return sets


def counts(flags):
    return {"flagged": int(flags.sum()), "n": int(len(flags)), "rate": float(flags.mean())}


def evaluate(ds, scaler, rows, bank, weights):
    """All detectors and rules on `rows` (held-out partition) and on the padded bank."""
    X = scaler.transform(pd.DataFrame(ds.X[rows], columns=ds.columns)).astype(np.float32)
    y = ds.y[rows]
    attack = y != ds.classes.index("BENIGN")
    p = attack_probabilities(X)
    rule = rule_flags(np.stack([p[n] for n in DETECTORS]), weights)

    det = {}
    for n in DETECTORS:
        m = binary_metrics(flagged(p[n]), attack)
        m["portscan"] = counts(flagged(p[n])[y == ds.classes.index("PortScan")])
        det[n] = m
    rules = {r: binary_metrics(rule[r], attack) for r in VOTING_RULES}

    pad_p = {}
    for size, A in padded_sets(bank).items():
        pad_p[size] = attack_probabilities(scaler.transform(pd.DataFrame(A, columns=ds.columns)).astype(np.float32))
    pad_rule = {size: rule_flags(np.stack([pp[n] for n in DETECTORS]), weights) for size, pp in pad_p.items()}
    padded = {r: {size: counts(pad_rule[size][r]) for size in pad_rule} for r in VOTING_RULES}
    nonzero = [s for s in pad_p if s != "0B"]
    diversity = {"test_rows": pairwise_diversity({n: flagged(p[n]) == attack for n in DETECTORS}),
                 "padded_nonzero": pairwise_diversity(
                     {n: np.concatenate([flagged(pad_p[s][n]) for s in nonzero]) for n in DETECTORS})}
    return {"rows": int(len(rows)), "bank_flows": int(len(bank)), "detectors": det, "rules": rules,
            "padded_probe_detection": padded, "diversity": diversity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["weights", "final"], required=True)
    args = ap.parse_args()

    ds = splits.load_dataset()
    sp = splits.load_splits(ds)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    weights_path = os.path.join(MODEL_DIR, "ensemble_weights.json")

    if args.stage == "weights":
        rows = splits.partition_rows(sp, "validation")
        bank = splits.load_bank("dev")
        # F1 first, from the validation rows alone: the weights cannot depend on anything reported later
        X = scaler.transform(pd.DataFrame(ds.X[rows], columns=ds.columns)).astype(np.float32)
        attack = ds.y[rows] != ds.classes.index("BENIGN")
        p = attack_probabilities(X)
        f1 = np.array([binary_metrics(flagged(p[n]), attack)["binary_f1"] for n in DETECTORS])
        weights = f1 / f1.sum()
        with open(weights_path, "w") as f:
            json.dump({n: round(float(w), 6) for n, w in zip(DETECTORS, weights)}, f, indent=2)
        out = evaluate(ds, scaler, rows, bank, weights)
        out.update({"stage": "weights", "split": "validation", "bank": "dev",
                    "weights": dict(zip(DETECTORS, map(float, weights))),
                    "validation_binary_f1": dict(zip(DETECTORS, map(float, f1)))})
        path = os.path.join(MODEL_DIR, "validation_eval.json")
        title = "validation partition + dev bank"
    else:
        splits.require_frozen("final-test evaluation")
        with open(weights_path) as f:
            w = json.load(f)
        weights = np.array([w[n] for n in DETECTORS], dtype=np.float64)
        rows = splits.partition_rows(sp, "final_test")
        bank = splits.load_bank("final_test")
        out = evaluate(ds, scaler, rows, bank, weights)
        out.update({"stage": "final", "split": "final_test", "bank": "final_test",
                    "weights": dict(zip(DETECTORS, map(float, weights)))})
        path = os.path.join(MODEL_DIR, "final_eval.json")
        title = "FINAL-TEST partition + final-test bank"

    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    lines = [f"detector set: {MODEL_DIR}", f"{title}: {out['rows']:,} rows, {out['bank_flows']:,} probe flows",
             f"weights (validation F1): {out['weights']}",
             "rule       attack-recall  false-pos     F1    | padded probes caught at 64 B / 128 B / 256 B / 512 B"]
    for r in VOTING_RULES:
        m, pr = out["rules"][r], out["padded_probe_detection"][r]
        cells = "  ".join(f"{pr[s]['flagged']}/{pr[s]['n']}" for s in ("64B", "128B", "256B", "512B"))
        lines.append(f"{r:9s}  {m['binary_recall']:8.2%}     {m['binary_false_positive_rate']:7.3%}  "
                     f"{m['binary_f1']:.4f}  | {cells}")
    print("\n".join(lines))
    log_event(f"v5 evaluation ({args.stage}) (evaluate_v5.py)", lines + [f"output: {path}"])


if __name__ == "__main__":
    main()
