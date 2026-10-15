"""
evaluate_detectors.py — Held-out performance of each detector and each decision rule.

Reproduces train_models.py's exact split and evaluates XGBoost, Random Forest
and the 1D-CNN on the 20% test set, both as 15-class classifiers and as the
binary attack-vs-BENIGN decision ensemble.py actually uses.

Also measures robustness to payload padding without any RL agent: the
held-out PortScan probe bank padded to each of the agent's payload sizes and
to random sizes, scored by each detector. Two port settings:
  bank ports — each flow keeps the destination port it was recorded with
  sim ports  — destination port 1001–1100, as ids_env assigns it (port = 1000 + step),
               i.e. the probes the DDQN agent actually sends

Every decision rule in ensemble.py (standalone + voting) is then scored on
the same detector outputs (paired), and pairwise diversity between the
detectors is measured (disagreement, double-fault; Kuncheva & Whitaker 2003).

Detector set: $IDS_MODEL_DIR (default code/models).
Outputs (in that directory):
  detector_eval.json     — metrics per detector (thesis table)
  rule_eval.json         — metrics per decision rule + detector diversity
  ensemble_weights.json  — confidence-weighted voting weights ∝ held-out binary F1
"""

import itertools
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
from ensemble import MODEL_DIR, VOTING_RULES
from flow_features import F_DST_PORT, MAX_PAYLOAD, set_forward_payload
from ids_env import EPISODE_LENGTH, PKTSIZE_OPTIONS, get_probe_bank
from project_log import log_event

# GPU for the CNN pass: on CPU it must stay single-threaded (libomp clash with xgboost)
CNN_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DETECTORS = ["xgboost", "rf", "cnn"]  # same order as ensemble.load_weights()
PORT_MODES = {"bank_ports": "padded_probe_detected",
              "sim_ports": "padded_probe_detected_sim_ports"}


def flagged(p_attack):
    """ensemble.py: label "scanning" when p_scanning = 1 - P(BENIGN) >= 0.5."""
    return p_attack >= 0.5


def rule_flags(p, weights, threshold=0.5):
    """p: (3, n) attack probabilities in DETECTORS order -> {rule: (n,) bool}, as EnsembleIDS.decide."""
    votes = flagged(p)
    n = votes.sum(axis=0)
    out = {name: votes[i] for i, name in enumerate(DETECTORS)}
    out["unanimous"] = n == 3
    out["majority"] = n >= 2
    out["any"] = n >= 1
    # explicit sum: numpy 2.0 matmul on macOS Accelerate raises spurious FP warnings
    out["weighted"] = (weights[:, None] * p).sum(axis=0) >= threshold
    return out


def binary_metrics(flags, attack_true):
    tp = int((flags & attack_true).sum())
    fp = int((flags & ~attack_true).sum())
    fn = int((~flags & attack_true).sum())
    tn = int((~flags & ~attack_true).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return {
        "binary_precision": precision,
        "binary_recall": recall,
        "binary_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "binary_false_positive_rate": fp / (fp + tn),
        "false_positives": fp,
        "missed_attacks": fn,
    }


def pairwise_diversity(correct):
    """
    correct: {detector: (n,) bool}. Per pair: share of rows where they disagree,
    split by which one is right, and where both are wrong (double fault).
    On padded probes every row is an attack, so "correct" means "flagged".
    """
    out = {}
    for a, b in itertools.combinations(DETECTORS, 2):
        ca, cb = correct[a], correct[b]
        out[f"{a}-{b}"] = {"disagreement": float((ca != cb).mean()),
                           f"only_{a}_correct": float((ca & ~cb).mean()),
                           f"only_{b}_correct": float((~ca & cb).mean()),
                           "double_fault": float((~ca & ~cb).mean())}
    return out


def rule_evaluation(p_test, p_pad, y_test, classes, weights):
    attack_true = y_test != classes.index("BENIGN")
    test_flags = rule_flags(np.stack([p_test[n] for n in DETECTORS]), weights)
    pad_flags = {mode: {k: rule_flags(np.stack([p_pad[n][mode][k] for n in DETECTORS]), weights)
                        for k in p_pad[DETECTORS[0]][mode]}
                 for mode in PORT_MODES}
    rules = {}
    for r in VOTING_RULES:
        f = test_flags[r]
        rules[r] = {
            **binary_metrics(f, attack_true),
            "per_class_flag_rate": {c: float(f[y_test == i].mean()) for i, c in enumerate(classes)},
            **{key: {k: float(v[r].mean()) for k, v in pad_flags[mode].items()}
               for mode, key in PORT_MODES.items()},
        }
    diversity = {"held_out_test_split": pairwise_diversity(
        {n: flagged(p_test[n]) == attack_true for n in DETECTORS})}
    for mode in PORT_MODES:
        nonzero = [k for k in p_pad[DETECTORS[0]][mode] if k != "0B"]
        diversity[f"padded_probes_nonzero_payload_{mode}"] = pairwise_diversity(
            {n: np.concatenate([flagged(p_pad[n][mode][k]) for k in nonzero]) for n in DETECTORS})
    return {"weights": dict(zip(DETECTORS, map(float, weights))), "rules": rules, "diversity": diversity}


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


def padded_probe_sets(seed=0, sim_ports=False):
    """Held-out PortScan probes at each agent payload size, plus random sizes."""
    bank = get_probe_bank().copy()
    if sim_ports:
        bank[:, F_DST_PORT] = 1001 + np.arange(len(bank)) % EPISODE_LENGTH
    sets = {f"{s}B": set_forward_payload(bank, s) for s in PKTSIZE_OPTIONS}
    rng = np.random.default_rng(seed)
    sets[f"random 1-{MAX_PAYLOAD}B"] = set_forward_payload(
        bank, rng.integers(1, MAX_PAYLOAD + 1, size=len(bank)))
    return sets


def padded_table(rows, label):
    """rows: {name: {size: rate}} -> log lines."""
    sizes = list(next(iter(rows.values())))
    lines = [f"{label:9s} " + "  ".join(f"{k:>8s}" for k in sizes)]
    for n, r in rows.items():
        lines.append(f"{n:9s} " + "  ".join(f"{r[k]:8.1%}" for k in sizes))
    return lines


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

    padded = {mode: {k: scaler.transform(pd.DataFrame(v, columns=X.columns)).astype(np.float32)
                     for k, v in padded_probe_sets(sim_ports=(mode == "sim_ports")).items()}
              for mode in PORT_MODES}

    results, p_test, p_pad = {}, {}, {}
    for name in DETECTORS:
        if name == "cnn":
            proba = cnn_proba
        else:
            model = joblib.load(os.path.join(MODEL_DIR, f"{name}_model.pkl"))
            # RF: multi-threaded predict_proba sums tree votes in a varying order, so
            # 50/50 votes land either side of 0.5 run to run. Single-threaded matches
            # ensemble.py's per-row inference exactly.
            model.set_params(n_jobs=1 if name == "rf" else -1)
            proba = model.predict_proba
        P = proba(Xs)
        p_test[name] = 1.0 - P[:, 0]
        p_pad[name] = {mode: {k: 1.0 - proba(v)[:, 0] for k, v in sets.items()}
                       for mode, sets in padded.items()}

        pred = P.argmax(axis=1)
        attack_pred = flagged(p_test[name])
        m = binary_metrics(attack_pred, attack_true)
        results[name] = {
            "multiclass_accuracy": float((pred == y_test).mean()),
            "binary_precision": m["binary_precision"],
            "binary_recall": m["binary_recall"],
            "binary_f1": m["binary_f1"],
            "binary_false_positive_rate": m["binary_false_positive_rate"],
            "portscan_detected": float(attack_pred[portscan].mean()),
            **{key: {k: float(flagged(v).mean()) for k, v in p_pad[name][mode].items()}
               for mode, key in PORT_MODES.items()},
        }
        print(f"\n── {name} ──")
        print(json.dumps(results[name], indent=2))
        print(classification_report(y_test, pred, labels=list(range(len(classes))),
                                    target_names=classes, digits=4, zero_division=0))

    with open(os.path.join(MODEL_DIR, "detector_eval.json"), "w") as f:
        json.dump(results, f, indent=2)

    f1 = np.array([results[n]["binary_f1"] for n in DETECTORS])
    weights = dict(zip(DETECTORS, (f1 / f1.sum()).round(6).tolist()))
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

    rule_eval = rule_evaluation(p_test, p_pad, y_test, classes,
                                np.array([weights[n] for n in DETECTORS], dtype=np.float64))
    with open(os.path.join(MODEL_DIR, "rule_eval.json"), "w") as f:
        json.dump(rule_eval, f, indent=2)

    n_bank = len(get_probe_bank())
    lines = [f"detector set: {MODEL_DIR}",
             f"held-out test split: {len(y_test):,} rows",
             "detector  accuracy  attack-recall  false-pos  PortScan-caught"]
    for n, r in results.items():
        lines.append(f"{n:8s}  {r['multiclass_accuracy']:7.2%}   {r['binary_recall']:7.2%}      "
                     f"{r['binary_false_positive_rate']:6.2%}    {r['portscan_detected']:7.2%}")
    lines.append(f"weighted-vote weights: {weights}")
    lines.append("decision rules on the held-out test split (paired, same detector outputs):")
    lines.append("rule       attack-recall  false-pos   F1")
    for r, m in rule_eval["rules"].items():
        lines.append(f"{r:9s}  {m['binary_recall']:8.2%}      {m['binary_false_positive_rate']:6.2%}   "
                     f"{m['binary_f1']:.4f}")
    for mode, key in PORT_MODES.items():
        port_note = ("recorded destination ports" if mode == "bank_ports"
                     else "destination ports 1001-1100, as in the simulator")
        lines.append(f"padded held-out scan probes caught ({n_bank:,} flows per size, {port_note}):")
        lines += padded_table({r: m[key] for r, m in rule_eval["rules"].items()}, "rule")
    lines.append("detector pairs: disagreement / double-fault "
                 "(test split | padded non-zero payload, sim ports)")
    for pair, d in rule_eval["diversity"]["held_out_test_split"].items():
        p = rule_eval["diversity"]["padded_probes_nonzero_payload_sim_ports"][pair]
        lines.append(f"{pair:12s} {d['disagreement']:6.2%} / {d['double_fault']:6.2%}   |   "
                     f"{p['disagreement']:6.2%} / {p['double_fault']:6.2%}")
    log_event("Detector evaluation (evaluate_detectors.py)", lines)
    print("\n" + "\n".join(lines))
