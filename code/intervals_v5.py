"""
intervals_v5.py — cluster-bootstrap 95% intervals for the headline rates (frozen protocol v5).

Why not plain binomial intervals: the final-test probe bank has 13,569 flows but only 72 invariant-feature
groups, and flows within a group are near-identical, so the effective sample is closer to 72 than 13,569.
Here whole clusters are resampled with replacement (percentile interval, 2,000 resamples, seed 0):

  padded-probe detection (no agent)   clusters = the 72 probe groups of the final-test bank
  clean-traffic false-positive and    clusters = invariant-feature signatures of the final-test rows
  missed-attack rates                 (rows outside the probe groups have no group; the signature is the
                                      conservative stand-in, see docs/split_audit.md)

The evasion rate of an agent that always sends payload size s equals 1 - detection at s, so the detection
intervals also bound that fixed-payload evasion. A trained agent's 2,000-probe count adds sampling noise on top
and varies with the agent seed; Part A therefore reports ranges over five seeds, not an interval.

Reads the frozen detector set in $IDS_MODEL_DIR and requires the frozen protocol (final-test data).
Writes results_v5/intervals/<set>.json.     IDS_MODEL_DIR=.../models_v5/cov3 python intervals_v5.py
"""

import json
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import joblib

import splits
from ensemble import MODEL_DIR, VOTING_RULES
from evaluate_detectors import DETECTORS, rule_flags
from evaluate_v5 import attack_probabilities, padded_sets

B = 2000
SEED = 0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results_v5", "intervals")
SIZES = ["0B", "64B", "128B", "256B", "512B", "random 1-1460B"]


def cluster_sums(cid, vectors):
    """vectors: list of per-item arrays -> (G, S) per-cluster sums."""
    g = int(cid.max()) + 1
    return np.stack([np.bincount(cid, weights=v.astype(np.float64), minlength=g) for v in vectors], axis=1)


def bootstrap_totals(C, rng):
    """(B, S) bootstrap totals of the per-cluster sums C (G, S)."""
    G = len(C)
    if G <= 2000:                                    # classic cluster bootstrap
        idx = rng.integers(0, G, size=(B, G))
        return np.stack([C[i].sum(axis=0) for i in idx])
    out = np.empty((B, C.shape[1]))                  # many tiny clusters: Poisson(1) weights, in chunks
    for s in range(0, B, 100):
        W = rng.poisson(1.0, size=(G, min(100, B - s))).astype(np.float64)
        # explicit sums: numpy 2.0 matmul on macOS Accelerate raises spurious floating-point warnings
        out[s:s + W.shape[1]] = np.stack([(C * W[:, j:j + 1]).sum(axis=0) for j in range(W.shape[1])])
    return out


def intervals(num_vecs, den_vecs, cid, rng):
    """Rates num/den per statistic with percentile 95% cluster-bootstrap intervals."""
    S = len(num_vecs)
    C = cluster_sums(cid, list(num_vecs) + list(den_vecs))
    T = bootstrap_totals(C, rng)
    num, den = C[:, :S].sum(axis=0), C[:, S:].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = T[:, :S] / T[:, S:]
    lo, hi = np.nanpercentile(rates, [2.5, 97.5], axis=0)
    return [{"k": int(n), "n": int(d), "rate": float(n / d) if d else None, "lo": float(a), "hi": float(b)}
            for n, d, a, b in zip(num, den, lo, hi)]


def main():
    splits.require_frozen("interval estimation on final-test data")
    ds = splits.load_dataset()
    sp = splits.load_splits(ds)
    import pandas as pd
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "ensemble_weights.json")) as f:
        w = json.load(f)
    weights = np.array([w[n] for n in DETECTORS], dtype=np.float64)
    rng = np.random.default_rng(SEED)

    # padded probes: final-test bank, clusters = probe groups
    z = np.load(os.path.join(splits.SPLIT_DIR, "probe_bank.npz"), allow_pickle=False)
    names = [str(n) for n in z["label_names"]]
    sel = z["label"] == names.index("final_test")
    bank, groups = z["flows"][sel], z["group"][sel]
    assert np.array_equal(bank, splits.load_bank("final_test"))
    _, gid = np.unique(groups, return_inverse=True)
    det_flags = {}
    for size, A in padded_sets(bank).items():
        X = scaler.transform(pd.DataFrame(A, columns=ds.columns)).astype(np.float32)
        p = attack_probabilities(X)
        det_flags[size] = rule_flags(np.stack([p[n] for n in DETECTORS]), weights)
    keys = [(r, s) for r in VOTING_RULES for s in SIZES]
    ones = np.ones(len(bank))
    res = intervals([det_flags[s][r] for r, s in keys], [ones] * len(keys), gid, rng)
    detection = {r: {s: res[i] for i, (rr, s) in enumerate(keys) if rr == r} for r in VOTING_RULES}

    # clean traffic: final-test rows, clusters = signatures
    rows = splits.partition_rows(sp, "final_test")
    X = scaler.transform(pd.DataFrame(ds.X[rows], columns=ds.columns)).astype(np.float32)
    p = attack_probabilities(X)
    flags = rule_flags(np.stack([p[n] for n in DETECTORS]), weights)
    attack = ds.y[rows] != ds.classes.index("BENIGN")
    sig = splits.signatures(ds.X[rows], ds.columns)
    _, cid = np.unique(sig, return_inverse=True)
    nums, dens, tags = [], [], []
    for r in VOTING_RULES:
        nums += [flags[r] & ~attack, ~flags[r] & attack]
        dens += [~attack, attack]
        tags += [(r, "false_positive_rate"), (r, "missed_attack_rate")]
    res = intervals(nums, dens, cid, rng)
    clean = {r: {} for r in VOTING_RULES}
    for (r, name), v in zip(tags, res):
        clean[r][name] = v

    out = {"detector_set": os.path.basename(MODEL_DIR), "bootstrap": {"resamples": B, "seed": SEED, "interval": "percentile 95%"},
           "probe_clusters": int(gid.max() + 1), "probe_flows": int(len(bank)),
           "row_clusters": int(cid.max() + 1), "rows": int(len(rows)),
           "detection": detection, "clean": clean}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{os.path.basename(MODEL_DIR)}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    d = detection["majority"]["64B"]
    print(f"{os.path.basename(MODEL_DIR)}: {out['probe_clusters']} probe groups, {out['row_clusters']:,} row clusters | "
          f"majority at 64 B: {d['k']:,}/{d['n']:,} [{d['lo']:.3f}, {d['hi']:.3f}]  -> {path}")


if __name__ == "__main__":
    main()
