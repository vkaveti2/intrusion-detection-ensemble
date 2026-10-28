# 07 — Data Separation Protocol (v5)

Approved by Dr. Saha. Implemented in `code/splits.py`; verified by `code/audit_splits.py` (results in `split_audit.md`, `split_audit.json`). This replaces the earlier draft of this file.

## What was wrong

1. **Voting weights.** The weighted-vote weights were set from binary F1 on the same held-out split used to report every performance number.
2. **Probe bank.** Agent training and final evaluation sampled the same 17,834-flow PortScan bank, differing only by RNG seed. A different seed does not make a flow unseen.
3. **Detectors and the bank.** The bank was never disjoint from detector training in the sense that matters: 296 of its 333 invariant-feature groups also occur among the training PortScan flows (17,795 of 17,834 flows).
4. **CNN epoch selection.** The original CNN chose its best epoch and drove its learning-rate scheduler on the test split. The hardened CNNs split their validation slice off *after* adding padded copies, so a flow and its padded copy could land on opposite sides.

## The unit of separation: invariant-feature groups

The CSVs used here are the MachineLearningCVE files (77 numeric features). They have no source IP, flow ID or timestamp, so a split by scanner host, capture session or time is not possible.

Instead, each row gets a **signature**: a hash of every feature the simulator does not modify. Excluded from the signature are the destination port, the 15 payload-dependent length and byte features that `set_forward_payload` recomputes, and `PSH Flag Count` (the masked detectors cannot see it). That leaves 61 features. Rows with the same signature form a group and always sit in the same partition.

**This is separation by invariant-feature groups. It does not establish independence by host, capture session or time.**

Scope of the guarantee:

- Every single-probe PortScan flow (1 forward packet, at most 1 backward packet; the shape the simulator models) defines a group by its signature. There are 480 groups over 89,215 flows; the largest has 991 flows and 135 are singletons.
- Rows of any other class that share one of those signatures follow the group.
- All remaining rows are split stratified by class. The guarantee does not cover them (see "Outside the guarantee").

## Partitions

Group assignment is flow-weighted and seeded (seed 2026). Shares are 70 / 15 / 15.

| Partition | Rows | Groups | Probe flows | Used for |
|---|---|---|---|---|
| train | 1,764,598 | 339 | 62,505 | detector training and adversarial augmentation; scaler fit |
| validation | 377,863 | 69 | 13,141 | voting weights, thresholds, fixed-payload selection |
| final test | 378,337 | 72 | 13,569 | every reported number |

| Probe bank | Flows | Groups | Used for |
|---|---|---|---|
| detector train | 62,505 | 339 | source flows for padded training copies |
| agent train | 9,325 | 55 | DDQN training only |
| dev | 3,816 | 14 | best fixed payload; configuration checks |
| final test | 13,569 | 72 | final agent and detector evaluation |

Agent train and dev are a 70 / 30 split by group of the validation side's probe flows. Dev is small (14 groups). The fixed-payload comparison therefore selects its payload on little data, and the thesis should say so.

Rules that follow:

- **Split before augmenting.** Padded copies are made only from train-partition source flows, after the split, and keep their source flow's partition and inner-validation side.
- **Scaler.** Fitted on train rows only. `MaskedScaler` then removes `PSH Flag Count`.
- **CNN.** 5% of train, chosen by group, is the inner validation set. The best epoch and `ReduceLROnPlateau` both use it. Trees train on every train row.
- **Voting weights and thresholds** come from the validation partition. The decision threshold stays 0.5 and is tuned nowhere.
- **Final test** is unreadable until `config/protocol_v5.json` is frozen. `splits.load_bank("final_test")` and `splits.partition_rows(..., "final_test")` raise otherwise.

## Audit result

`audit_splits.py` passes. In particular, there are zero shared signatures between:

- train and validation, train and final test, validation and final test (grouped rows);
- the agent-train bank and detector train, dev and final test;
- the final-test bank and detector train;
- padded training copies (bands A, B and C) and the final-test, agent-train or dev banks.

Padded copies keep their source signature in all 62,505 rows tested, so augmentation cannot introduce a final-test signature.

## Outside the guarantee

Rows that share no signature with a probe group are split by class only. Of 364,593 final-test rows in that category, 47,339 share a signature with a train row. This is ordinary duplicate-feature overlap in the benign and other attack classes. It is reported and not removed. The claim of unseen flows applies to the PortScan probe flows that the attack experiments use.

## Consequences for earlier results

All earlier detector sets were trained on a split that is not group-disjoint from the bank, and the baseline CNN used the test split for epoch selection. Every detector set is therefore retrained under this protocol (`train_v5.py`, `models_v5/`). The earlier results are kept, labelled as the pre-correction diagnostics that motivated it.
