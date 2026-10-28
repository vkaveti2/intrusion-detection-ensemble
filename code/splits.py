"""
splits.py — the one train / validation / final-test protocol (docs/07_split_plan.md).

Every script that touches data goes through this module instead of calling
train_test_split on its own. Separation is by *invariant-feature groups*: rows that are
identical in every feature the simulator does not modify (so, not the destination port,
not the payload-dependent length/byte columns, and not PSH Flag Count, which the masked
detectors cannot see) share a group signature and always land in the same partition.
This does NOT establish independence by host, capture session or time: the
MachineLearningCVE CSVs carry no source IP or timestamp.

Group scope. Every single-probe PortScan flow (1 forward packet, <=1 backward packet; the
shape the simulator models) defines a group by its signature. Rows of any other class that
share one of those signatures follow the group. All remaining rows are split stratified
by class. The guarantee covers probe flows and signature-sharing rows only.

Partitions (flow-weighted, seeded):
  train       70%  detector training and adversarial augmentation (CNN inner validation 5%)
  validation  15%  voting weights, thresholds, fixed-payload selection; its probe flows are
                   split 70/30 by group into agent_train and dev
  final_test  15%  every reported number; unreadable until config/protocol_v5.json is frozen

Padded (augmented) copies are generated from train source flows only, after the split,
and inherit their source flow's partition and inner-validation side.

    python splits.py build    # write code/splits_v5/ (splits.npz, splits.json, probe_bank.npz)
    python splits.py info     # print the counts
"""

import glob
import hashlib
import json
import os
import sys
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flow_features import (F_ACT_DATA_PKT_FWD, F_AVG_FWD_SEG, F_AVG_PKT_SIZE, F_DST_PORT,
                           F_FLOW_BYTES_S, F_FWD_LEN_MAX, F_FWD_LEN_MEAN, F_FWD_LEN_MIN,
                           F_FWD_LEN_STD, F_MAX_PKT_LEN, F_MIN_PKT_LEN, F_PKT_LEN_MEAN,
                           F_PKT_LEN_STD, F_PKT_LEN_VAR, F_SUBFLOW_FWD_BYTES, F_TOT_BWD_PKTS,
                           F_TOT_FWD_PKTS, F_TOT_LEN_FWD)

CODE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(CODE_DIR, "data", "*.csv")
SPLIT_DIR   = os.path.join(CODE_DIR, "splits_v5")
FROZEN_PATH = os.path.join(os.path.dirname(CODE_DIR), "config", "protocol_v5.json")

SEED            = 2026
SHARES          = (0.70, 0.15, 0.15)   # train, validation, final test
AGENT_SHARE     = 0.70                 # of validation-side probe flows -> agent_train; rest dev
INNER_VAL_SHARE = 0.05                 # of train, for CNN epoch / scheduler selection
PART_NAMES      = ("train", "validation", "final_test")
BANK_LABELS     = ("detector_train", "agent_train", "dev", "final_test")
PSH_COLUMN      = "PSH Flag Count"

# Features the simulator overwrites (destination port + set_forward_payload outputs)
MODIFIED = {F_DST_PORT, F_TOT_LEN_FWD, F_FWD_LEN_MAX, F_FWD_LEN_MIN, F_FWD_LEN_MEAN, F_AVG_FWD_SEG,
            F_SUBFLOW_FWD_BYTES, F_FWD_LEN_STD, F_ACT_DATA_PKT_FWD, F_MIN_PKT_LEN, F_MAX_PKT_LEN,
            F_PKT_LEN_MEAN, F_PKT_LEN_VAR, F_PKT_LEN_STD, F_AVG_PKT_SIZE, F_FLOW_BYTES_S}

Dataset = namedtuple("Dataset", "X y classes columns")
Splits  = namedtuple("Splits", "part group is_bank agent_split inner_val meta")


# ── Data ───────────────────────────────────────────────────────────────────────
def _read_clean():
    files = sorted(glob.glob(DATA_PATH))            # sorted: row order must not depend on the OS
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()
    if "Fwd Header Length.1" in df.columns:
        df = df.drop(columns=["Fwd Header Length.1"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df.reset_index(drop=True)


def load_dataset():
    """Cleaned CIC-IDS2017 (2,520,798 rows x 77 features), cached next to the splits."""
    cache = os.path.join(SPLIT_DIR, "dataset_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        # plain str, not np.str_: scikit-learn only records feature_names_in_ for exact str names
        return Dataset(z["X"], z["y"], [str(c) for c in z["classes"]], [str(c) for c in z["columns"]])
    df = _read_clean()
    classes = sorted(df["Label"].unique())          # == LabelEncoder order (alphabetical)
    y = df["Label"].map({c: i for i, c in enumerate(classes)}).to_numpy(np.int16)
    X = df.drop(columns=["Label"]).astype(np.float32)
    columns = list(X.columns)
    X = X.to_numpy(np.float32)
    os.makedirs(SPLIT_DIR, exist_ok=True)
    np.savez(cache, X=X, y=y, classes=np.array(classes), columns=np.array(columns))
    return Dataset(X, y, classes, columns)


def fingerprint(ds):
    """sha256 over every row's hash and label: detects any change in content or order."""
    h = pd.util.hash_pandas_object(pd.DataFrame(ds.X), index=False).to_numpy()
    return hashlib.sha256(h.tobytes() + ds.y.astype(np.int16).tobytes()).hexdigest()


def sig_columns(columns):
    """Indices of the features in the group signature."""
    psh = columns.index(PSH_COLUMN)
    return [i for i in range(len(columns)) if i not in MODIFIED and i != psh]


def signatures(X, columns):
    """64-bit signature per row over the retained, unmodified features."""
    return pd.util.hash_pandas_object(pd.DataFrame(X[:, sig_columns(columns)]), index=False).to_numpy()


def is_single_probe_scan(ds):
    ps = ds.classes.index("PortScan")
    return (ds.y == ps) & (ds.X[:, F_TOT_FWD_PKTS] == 1) & (ds.X[:, F_TOT_BWD_PKTS] <= 1)


# ── Building the split ─────────────────────────────────────────────────────────
def _split_groups(sizes, fractions, rng):
    """Flow-weighted random assignment of groups to buckets (bucket index per group)."""
    order = rng.permutation(len(sizes))
    mid = (np.cumsum(sizes[order]) - sizes[order] / 2) / sizes.sum()
    bucket = np.empty(len(sizes), np.int8)
    bucket[order] = np.searchsorted(np.cumsum(fractions)[:-1], mid, side="right")
    return bucket


def build(ds):
    n = len(ds.y)
    sig = signatures(ds.X, ds.columns)
    bank = is_single_probe_scan(ds)

    uni = np.unique(sig[bank])                       # one signature per group
    idx = np.minimum(np.searchsorted(uni, sig), len(uni) - 1)
    hit = uni[idx] == sig
    group = np.full(n, -1, np.int32)
    group[hit] = idx[hit]
    sizes = np.bincount(group[bank], minlength=len(uni))   # bank flows per group

    part = np.full(n, -1, np.int8)
    g_part = _split_groups(sizes, SHARES, np.random.default_rng([SEED, 0]))
    part[group >= 0] = g_part[group[group >= 0]]

    loose = np.where(group < 0)[0]                   # everything outside the group guarantee
    t_idx, rest = train_test_split(loose, test_size=1 - SHARES[0], stratify=ds.y[loose], random_state=SEED)
    v_idx, f_idx = train_test_split(rest, test_size=SHARES[2] / (SHARES[1] + SHARES[2]),
                                    stratify=ds.y[rest], random_state=SEED)
    part[t_idx], part[v_idx], part[f_idx] = 0, 1, 2

    agent_split = np.full(n, -1, np.int8)            # bank flows on the validation side only
    v_groups = np.where(g_part == 1)[0]
    g_agent = _split_groups(sizes[v_groups], (AGENT_SHARE, 1 - AGENT_SHARE), np.random.default_rng([SEED, 1]))
    lookup = np.full(len(uni), -1, np.int8)
    lookup[v_groups] = g_agent
    agent_split[bank & (part == 1)] = lookup[group[bank & (part == 1)]]

    inner_val = np.zeros(n, bool)                    # train side: by group, else stratified by class
    t_groups = np.where(g_part == 0)[0]
    g_inner = _split_groups(sizes[t_groups], (1 - INNER_VAL_SHARE, INNER_VAL_SHARE), np.random.default_rng([SEED, 2]))
    in_lookup = np.zeros(len(uni), bool)
    in_lookup[t_groups] = g_inner == 1
    grouped_train = (part == 0) & (group >= 0)
    inner_val[grouped_train] = in_lookup[group[grouped_train]]
    loose_train = np.where((part == 0) & (group < 0))[0]
    _, val_idx = train_test_split(loose_train, test_size=INNER_VAL_SHARE, stratify=ds.y[loose_train],
                                  random_state=SEED)
    inner_val[val_idx] = True

    meta = counts(ds, part, group, bank, agent_split, inner_val, sizes)
    meta.update({"seed": SEED, "shares": SHARES, "agent_share": AGENT_SHARE,
                 "inner_val_share": INNER_VAL_SHARE, "fingerprint": fingerprint(ds),
                 "signature_columns": len(sig_columns(ds.columns)),
                 "excluded_from_signature": sorted(ds.columns[i] for i in MODIFIED | {ds.columns.index(PSH_COLUMN)})})
    return Splits(part, group, bank, agent_split, inner_val, meta)


def counts(ds, part, group, bank, agent_split, inner_val, sizes):
    def groups_in(mask):
        return int(len(np.unique(group[mask & (group >= 0)])))
    out = {"rows": int(len(part)), "groups_total": int(len(sizes)), "largest_group_flows": int(sizes.max()),
           "singleton_groups": int((sizes == 1).sum()),
           "partitions": {}, "bank": {}, "per_class_rows": {}}
    for i, name in enumerate(PART_NAMES):
        m = part == i
        out["partitions"][name] = {"rows": int(m.sum()), "grouped_rows": int((m & (group >= 0)).sum()),
                                   "groups": groups_in(m), "probe_flows": int((m & bank).sum())}
    out["partitions"]["train"]["inner_validation_rows"] = int(inner_val.sum())
    out["partitions"]["train"]["inner_validation_groups"] = groups_in(inner_val)
    for label, m in (("detector_train", bank & (part == 0)), ("agent_train", bank & (agent_split == 0)),
                     ("dev", bank & (agent_split == 1)), ("final_test", bank & (part == 2))):
        out["bank"][label] = {"flows": int(m.sum()), "groups": groups_in(m)}
    for i, c in enumerate(ds.classes):
        out["per_class_rows"][c] = {nm: int(((ds.y == i) & (part == k)).sum()) for k, nm in enumerate(PART_NAMES)}
    return out


def save(ds, sp):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(SPLIT_DIR, "splits.npz"), part=sp.part, group=sp.group,
                        is_bank=sp.is_bank, agent_split=sp.agent_split, inner_val=sp.inner_val)
    with open(os.path.join(SPLIT_DIR, "splits.json"), "w") as f:
        json.dump(sp.meta, f, indent=2)
    rows = np.where(sp.is_bank)[0]
    label = np.where(sp.part[rows] == 0, 0, np.where(sp.part[rows] == 2, 3, 1 + sp.agent_split[rows]))
    np.savez_compressed(os.path.join(SPLIT_DIR, "probe_bank.npz"), flows=ds.X[rows], row_index=rows,
                        group=sp.group[rows], label=label.astype(np.int8),
                        label_names=np.array(BANK_LABELS), feature_names=np.array(ds.columns))


def load_splits(ds, verify=True):
    z = np.load(os.path.join(SPLIT_DIR, "splits.npz"))
    with open(os.path.join(SPLIT_DIR, "splits.json")) as f:
        meta = json.load(f)
    if verify and fingerprint(ds) != meta["fingerprint"]:
        raise RuntimeError("dataset differs from the one the splits were built on (fingerprint mismatch)")
    return Splits(z["part"], z["group"], z["is_bank"], z["agent_split"], z["inner_val"], meta)


# ── Access with the final-test gate ────────────────────────────────────────────
def is_frozen():
    if not os.path.exists(FROZEN_PATH):
        return False
    with open(FROZEN_PATH) as f:
        return bool(json.load(f).get("frozen"))


def require_frozen(what):
    if not is_frozen():
        raise PermissionError(f"{what} is final-test data: freeze the protocol first "
                              f"(set \"frozen\": true in {FROZEN_PATH}, docs/12_frozen_protocol.md)")


_BANK_CACHE = {}


def load_bank(name):
    """Probe flows of one bank: detector_train | agent_train | dev | final_test."""
    if name == "final_test":
        require_frozen("the final-test probe bank")      # checked on every call, cached or not
    if name not in _BANK_CACHE:                          # the simulator asks once per step
        z = np.load(os.path.join(SPLIT_DIR, "probe_bank.npz"), allow_pickle=False)
        names = [str(n) for n in z["label_names"]]
        _BANK_CACHE[name] = z["flows"][z["label"] == names.index(name)]
    return _BANK_CACHE[name]


def partition_rows(sp, name):
    """Row indices of a partition; final_test is gated."""
    if name == "final_test":
        require_frozen("the final-test rows")
    return np.where(sp.part == PART_NAMES.index(name))[0]


def fit_scaler(ds, sp):
    """StandardScaler fitted on the train partition only."""
    cols = ds.columns
    return StandardScaler().fit(pd.DataFrame(ds.X[sp.part == 0], columns=cols))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "info"
    ds = load_dataset()
    if cmd == "build":
        sp = build(ds)
        save(ds, sp)
        print("saved →", SPLIT_DIR)
    else:
        sp = load_splits(ds)
    m = sp.meta
    print(json.dumps({k: m[k] for k in ("rows", "groups_total", "largest_group_flows", "singleton_groups",
                                        "partitions", "bank", "fingerprint")}, indent=2))
