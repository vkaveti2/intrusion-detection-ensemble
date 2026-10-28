"""
audit_splits.py — verify the split protocol (docs/07_split_plan.md); any failure blocks the freeze.

Checks, all on the group signature (splits.signatures):
  1. partitions are disjoint and cover every row; counts match splits.json; fingerprint matches
  2. guaranteed scope: no signature of a final-test (or validation) probe group occurs in the train
     partition, and the agent-train bank shares no signature with dev or final-test
  3. augmentation: padded copies (set_forward_payload at 1..1460 B) keep their source signature,
     so no augmented training row can carry a final-test signature
  4. CNN inner validation is a subset of train and shares no group with the CNN's inner-train side
  5. the scaler is fitted on train rows only (mean_ reproduces from train rows)
Informational (outside the guarantee): loose rows' signature overlap across partitions.

Output: docs/split_audit.md, docs/split_audit.json.   python audit_splits.py
"""

import json
import os
import sys

import numpy as np

import splits
from flow_features import set_forward_payload

DOCS = os.path.join(os.path.dirname(splits.CODE_DIR), "docs")


def main():
    ds = splits.load_dataset()
    sp = splits.load_splits(ds)                                  # raises on fingerprint mismatch
    part, group, bank, agent, inner = sp.part, sp.group, sp.is_bank, sp.agent_split, sp.inner_val
    sig = splits.signatures(ds.X, ds.columns)
    results, failures = [], []

    def check(name, ok, detail):
        results.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            failures.append(name)

    # 1 ── coverage and counts
    check("every row has exactly one partition", bool(np.isin(part, [0, 1, 2]).all()),
          f"{len(part):,} rows; counts " + ", ".join(f"{n}={int((part == i).sum()):,}" for i, n in enumerate(splits.PART_NAMES)))
    check("counts match splits.json",
          all(int((part == i).sum()) == sp.meta["partitions"][n]["rows"] for i, n in enumerate(splits.PART_NAMES)), "")

    # 2 ── guaranteed scope: grouped rows (probe groups and rows sharing their signature)
    grouped = group >= 0
    s_train = set(sig[(part == 0) & grouped].tolist())
    s_val   = set(sig[(part == 1) & grouped].tolist())
    s_test  = set(sig[(part == 2) & grouped].tolist())
    check("grouped rows: train ∩ validation signatures", not (s_train & s_val), f"{len(s_train & s_val)} shared")
    check("grouped rows: train ∩ final_test signatures", not (s_train & s_test), f"{len(s_train & s_test)} shared")
    check("grouped rows: validation ∩ final_test signatures", not (s_val & s_test), f"{len(s_val & s_test)} shared")

    s_agent = set(sig[bank & (agent == 0)].tolist())
    s_dev   = set(sig[bank & (agent == 1)].tolist())
    s_btest = set(sig[bank & (part == 2)].tolist())
    s_btrain = set(sig[bank & (part == 0)].tolist())
    check("agent-train bank ∩ detector-train signatures", not (s_agent & s_btrain), f"{len(s_agent & s_btrain)} shared")
    check("agent-train bank ∩ final-test bank signatures", not (s_agent & s_btest), f"{len(s_agent & s_btest)} shared")
    check("agent-train bank ∩ dev bank signatures", not (s_agent & s_dev), f"{len(s_agent & s_dev)} shared")
    check("dev bank ∩ final-test bank signatures", not (s_dev & s_btest), f"{len(s_dev & s_btest)} shared")
    check("final-test bank ∩ detector-train signatures", not (s_btest & s_btrain), f"{len(s_btest & s_btrain)} shared")

    # 3 ── augmentation inherits the source signature
    src = ds.X[bank & (part == 0)]
    rng = np.random.default_rng(0)
    worst = 0
    for lo, hi in ((1, 90), (91, 362), (363, 1460)):
        padded = set_forward_payload(src, rng.integers(lo, hi + 1, size=len(src)))
        same = splits.signatures(padded, ds.columns) == splits.signatures(src, ds.columns)
        worst = max(worst, int((~same).sum()))
    check("padded copies keep their source signature (bands A, B, C)", worst == 0, f"{worst} rows changed")
    s_pad = set(splits.signatures(set_forward_payload(src, rng.integers(1, 1461, size=len(src))), ds.columns).tolist())
    check("augmented rows ∩ final-test signatures", not (s_pad & s_btest), f"{len(s_pad & s_btest)} shared")
    check("augmented rows ∩ agent-train/dev signatures", not (s_pad & (s_agent | s_dev)), f"{len(s_pad & (s_agent | s_dev))} shared")

    # 4 ── CNN inner validation
    check("inner validation ⊂ train", bool((~inner | (part == 0)).all()), f"{int(inner.sum()):,} rows")
    g_in = set(group[inner & grouped].tolist())
    g_tr = set(group[(part == 0) & ~inner & grouped].tolist())
    check("inner-validation groups ∩ inner-train groups", not (g_in & g_tr), f"{len(g_in & g_tr)} shared")

    # 5 ── scaler fit on train only
    sc = splits.fit_scaler(ds, sp)
    ok = np.allclose(sc.mean_, ds.X[part == 0].mean(axis=0, dtype=np.float64), rtol=1e-6)
    check("scaler mean reproduces from train rows only", ok, f"{int((part == 0).sum()):,} train rows")

    # informational: rows outside the guarantee
    loose = ~grouped
    ls = {n: set(sig[loose & (part == i)].tolist()) for i, n in enumerate(splits.PART_NAMES)}
    info = {"loose_rows_signature_overlap": {
        "train ∩ final_test": len(ls["train"] & ls["final_test"]),
        "train ∩ validation": len(ls["train"] & ls["validation"]),
        "final_test_rows_sharing_train_signature": int(np.isin(sig[loose & (part == 2)], list(ls["train"])).sum()),
        "final_test_loose_rows": int((loose & (part == 2)).sum())}}

    m = sp.meta
    report = {"passed": not failures, "failures": failures, "checks": results, "informational": info,
              "partitions": m["partitions"], "bank": m["bank"], "groups_total": m["groups_total"],
              "largest_group_flows": m["largest_group_flows"], "singleton_groups": m["singleton_groups"],
              "fingerprint": m["fingerprint"], "seed": m["seed"],
              "signature_features": m["signature_columns"], "excluded_from_signature": m["excluded_from_signature"]}
    with open(os.path.join(DOCS, "split_audit.json"), "w") as f:
        json.dump(report, f, indent=2)

    L = ["# Split audit", "",
         "Separation is by invariant-feature groups: rows identical in every feature the simulator does not "
         "modify share a signature and stay in one partition. It does not establish independence by host, "
         "capture session or time; the CSVs have no source IP or timestamp.", "",
         f"**Result: {'PASSED' if not failures else 'FAILED: ' + ', '.join(failures)}**   "
         f"(seed {m['seed']}, {m['signature_columns']} signature features, fingerprint `{m['fingerprint'][:16]}…`)", "",
         "## Counts", "", "| Partition | Rows | Groups | Probe flows |", "|---|---|---|---|"]
    for n in splits.PART_NAMES:
        p = m["partitions"][n]
        L.append(f"| {n} | {p['rows']:,} | {p['groups']} | {p['probe_flows']:,} |")
    L += ["", "| Probe bank | Flows | Groups |", "|---|---|---|"]
    for n, p in m["bank"].items():
        L.append(f"| {n} | {p['flows']:,} | {p['groups']} |")
    L += ["", f"{m['groups_total']} groups in total; largest {m['largest_group_flows']} flows; "
          f"{m['singleton_groups']} singletons. CNN inner validation: "
          f"{m['partitions']['train']['inner_validation_rows']:,} train rows "
          f"({m['partitions']['train']['inner_validation_groups']} groups).", "", "## Checks", "",
          "| Check | Result | Detail |", "|---|---|---|"]
    for r in results:
        L.append(f"| {r['check']} | {'pass' if r['passed'] else '**FAIL**'} | {r['detail']} |")
    o = info["loose_rows_signature_overlap"]
    L += ["", "## Outside the guarantee (informational)", "",
          "Rows that share no signature with a probe group are split by class only. "
          f"Of {o['final_test_loose_rows']:,} such final-test rows, {o['final_test_rows_sharing_train_signature']:,} "
          f"share a signature with a train row ({o['train ∩ final_test']} distinct signatures). "
          "This is ordinary duplicate-feature overlap in the benign and attack classes and is reported, not removed.", "",
          "## Signature excludes", "", ", ".join(f"`{c}`" for c in m["excluded_from_signature"])]
    with open(os.path.join(DOCS, "split_audit.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    print("\n".join(f"{'pass' if r['passed'] else 'FAIL'}  {r['check']}  [{r['detail']}]" for r in results))
    print("informational:", json.dumps(info))
    print("AUDIT", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
