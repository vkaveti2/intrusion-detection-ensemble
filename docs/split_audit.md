# Split audit

Separation is by invariant-feature groups: rows identical in every feature the simulator does not modify share a signature and stay in one partition. It does not establish independence by host, capture session or time; the CSVs have no source IP or timestamp.

**Result: PASSED**   (seed 2026, 60 signature features, fingerprint `deb78d125a24c9c8…`)

## Counts

| Partition | Rows | Groups | Probe flows |
|---|---|---|---|
| train | 1,764,598 | 339 | 62,505 |
| validation | 377,863 | 69 | 13,141 |
| final_test | 378,337 | 72 | 13,569 |

| Probe bank | Flows | Groups |
|---|---|---|
| detector_train | 62,505 | 339 |
| agent_train | 9,325 | 55 |
| dev | 3,816 | 14 |
| final_test | 13,569 | 72 |

480 groups in total; largest 991 flows; 135 singletons. CNN inner validation: 88,121 train rows (11 groups).

## Checks

| Check | Result | Detail |
|---|---|---|
| every row has exactly one partition | pass | 2,520,798 rows; counts train=1,764,598, validation=377,863, final_test=378,337 |
| counts match splits.json | pass |  |
| grouped rows: train ∩ validation signatures | pass | 0 shared |
| grouped rows: train ∩ final_test signatures | pass | 0 shared |
| grouped rows: validation ∩ final_test signatures | pass | 0 shared |
| agent-train bank ∩ detector-train signatures | pass | 0 shared |
| agent-train bank ∩ final-test bank signatures | pass | 0 shared |
| agent-train bank ∩ dev bank signatures | pass | 0 shared |
| dev bank ∩ final-test bank signatures | pass | 0 shared |
| final-test bank ∩ detector-train signatures | pass | 0 shared |
| padded copies keep their source signature (bands A, B, C) | pass | 0 rows changed |
| augmented rows ∩ final-test signatures | pass | 0 shared |
| augmented rows ∩ agent-train/dev signatures | pass | 0 shared |
| inner validation ⊂ train | pass | 88,121 rows |
| inner-validation groups ∩ inner-train groups | pass | 0 shared |
| scaler mean reproduces from train rows only | pass | 1,764,598 train rows |

## Outside the guarantee (informational)

Rows that share no signature with a probe group are split by class only. Of 364,593 such final-test rows, 47,339 share a signature with a train row (25588 distinct signatures). This is ordinary duplicate-feature overlap in the benign and attack classes and is reported, not removed.

## Signature excludes

`Average Packet Size`, `Avg Fwd Segment Size`, `Destination Port`, `Flow Bytes/s`, `Fwd Packet Length Max`, `Fwd Packet Length Mean`, `Fwd Packet Length Min`, `Fwd Packet Length Std`, `Max Packet Length`, `Min Packet Length`, `PSH Flag Count`, `Packet Length Mean`, `Packet Length Std`, `Packet Length Variance`, `Subflow Fwd Bytes`, `Total Length of Fwd Packets`, `act_data_pkt_fwd`
