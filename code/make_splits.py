"""
make_splits.py — validation / agent-train / dev / final-test partitions (docs/07_split_plan.md).

Detectors are trained on the 80% training split and never change. The 20% held-out
split (random_state=42, as everywhere) is divided in two:
  validation side — voting weights, any threshold choice, agent-train + dev probe banks
  final-test side — every reported number
Non-bank rows are split 50/50 stratified by class. Probe-bank flows (single-probe
PortScan, as build_probe_bank.py) are split by near-duplicate group, so flows that are
identical in every feature the simulator does not modify never straddle a partition.
No source IP or timestamp exists in these CSVs, so this is a proxy for host separation.

Output (code/models/): splits.npz, splits.json.
  heldout_side  (n_heldout,)  0 = validation, 1 = final test
  bank_part     (n_bank,)     0 = agent_train, 1 = dev, 2 = final_test   (probe_bank.npz order)
  bank_group    (n_bank,)     near-duplicate group id
    python make_splits.py [--check]
"""

import argparse
import hashlib
import json
import os

import joblib
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from build_probe_bank import DATA_PATH, MODEL_DIR, load_and_clean_data
from flow_features import (F_ACT_DATA_PKT_FWD, F_AVG_FWD_SEG, F_AVG_PKT_SIZE, F_DST_PORT,
                           F_FLOW_BYTES_S, F_FWD_LEN_MAX, F_FWD_LEN_MEAN, F_FWD_LEN_MIN,
                           F_FWD_LEN_STD, F_MAX_PKT_LEN, F_MIN_PKT_LEN, F_PKT_LEN_MEAN,
                           F_PKT_LEN_STD, F_PKT_LEN_VAR, F_SUBFLOW_FWD_BYTES, F_TOT_LEN_FWD)

SEED = 2026
AGENT_TRAIN_SHARE = 0.7        # of the validation-side bank groups; the rest is dev
PART_NAMES = ["agent_train", "dev", "final_test"]
# Columns the simulator overwrites (destination port + set_forward_payload outputs)
MODIFIED = {F_DST_PORT, F_TOT_LEN_FWD, F_FWD_LEN_MAX, F_FWD_LEN_MIN, F_FWD_LEN_MEAN, F_AVG_FWD_SEG,
            F_SUBFLOW_FWD_BYTES, F_FWD_LEN_STD, F_ACT_DATA_PKT_FWD, F_MIN_PKT_LEN, F_MAX_PKT_LEN,
            F_PKT_LEN_MEAN, F_PKT_LEN_VAR, F_PKT_LEN_STD, F_AVG_PKT_SIZE, F_FLOW_BYTES_S}
KEEP = [i for i in range(77) if i not in MODIFIED]


def group_ids(flows):
    """Equal id = identical in every feature the simulator leaves unchanged."""
    keys = [hashlib.md5(np.ascontiguousarray(r[KEEP]).tobytes()).hexdigest() for r in flows]
    return np.unique(keys, return_inverse=True)[1]


def build():
    df = load_and_clean_data(DATA_PATH)
    le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    y = le.transform(df["Label"])
    X = df.drop(columns=["Label"]).astype(np.float32)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test = X_test.to_numpy(np.float32)

    ps = list(le.classes_).index("PortScan")
    in_bank = (y_test == ps) & (X_test[:, 2] == 1) & (X_test[:, 3] <= 1)   # build_probe_bank.py
    bank = X_test[in_bank]
    saved = np.load(os.path.join(MODEL_DIR, "probe_bank.npz"), allow_pickle=True)["flows"]
    assert np.array_equal(bank, saved), "reconstructed bank differs from probe_bank.npz"

    side = np.zeros(len(y_test), dtype=np.int8)
    rest = np.where(~in_bank)[0]
    _, test_rest = train_test_split(rest, test_size=0.5, stratify=y_test[rest], random_state=SEED)
    side[test_rest] = 1

    groups = group_ids(bank)
    bank_idx = np.where(in_bank)[0]
    to_val, to_test = next(GroupShuffleSplit(1, test_size=0.5, random_state=SEED).split(bank, groups=groups))
    side[bank_idx[to_test]] = 1
    part = np.full(len(bank), 2, dtype=np.int8)
    tr, dev = next(GroupShuffleSplit(1, test_size=1 - AGENT_TRAIN_SHARE, random_state=SEED)
                   .split(bank[to_val], groups=groups[to_val]))
    part[to_val[tr]], part[to_val[dev]] = 0, 1
    return y_test, classes_of(le), in_bank, side, part, groups


def classes_of(le):
    return list(le.classes_)


def check(y_test, classes, in_bank, side, part, groups):
    assert set(np.unique(part)) == {0, 1, 2}
    # no near-duplicate group on both sides of any partition boundary
    for g in np.unique(groups):
        assert len(set(part[groups == g])) == 1, f"group {g} spans partitions"
    # bank partitions agree with the held-out side
    bank_side = side[np.where(in_bank)[0]]
    assert np.array_equal(bank_side == 1, part == 2), "bank partition and held-out side disagree"
    # every class present on both sides (required for F1 and per-class reporting)
    for i, c in enumerate(classes):
        n = (y_test == i)
        assert n.sum() == 0 or {0, 1} <= set(side[n]) or n.sum() < 2, f"class {c} on one side only"
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="Verify the saved splits instead of rebuilding")
    args = ap.parse_args()
    y_test, classes, in_bank, side, part, groups = build()
    if args.check:
        z = np.load(os.path.join(MODEL_DIR, "splits.npz"))
        assert np.array_equal(z["heldout_side"], side) and np.array_equal(z["bank_part"], part), \
            "saved splits differ from a fresh build"
    check(y_test, classes, in_bank, side, part, groups)

    sizes = np.bincount(groups)
    info = {
        "seed": SEED, "heldout_rows": int(len(y_test)),
        "validation_side_rows": int((side == 0).sum()), "final_test_side_rows": int((side == 1).sum()),
        "bank_flows": int(len(part)), "bank_groups": int(len(sizes)),
        "largest_group": int(sizes.max()), "singleton_groups": int((sizes == 1).sum()),
        "bank_partition_flows": {n: int((part == i).sum()) for i, n in enumerate(PART_NAMES)},
        "bank_partition_groups": {n: int(len(np.unique(groups[part == i]))) for i, n in enumerate(PART_NAMES)},
        "per_class_rows": {c: {"validation": int(((y_test == i) & (side == 0)).sum()),
                               "final_test": int(((y_test == i) & (side == 1)).sum())}
                           for i, c in enumerate(classes)},
    }
    print(json.dumps({k: v for k, v in info.items() if k != "per_class_rows"}, indent=2))
    print("checks passed: partitions disjoint, no group spans partitions, bank matches probe_bank.npz")
    if not args.check:
        np.savez(os.path.join(MODEL_DIR, "splits.npz"), heldout_side=side, bank_part=part, bank_group=groups)
        with open(os.path.join(MODEL_DIR, "splits.json"), "w") as f:
            json.dump(info, f, indent=2)
        print("Saved → models/splits.npz, models/splits.json")
