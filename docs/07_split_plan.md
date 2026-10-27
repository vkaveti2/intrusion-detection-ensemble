# 07 — Split Plan (draft for Dr. Saha, no reruns started)

Short version of the plan for separating validation, agent-training and final-test data. Nothing below has been run yet except the group count in section 2.

## 1. Problems being fixed

1. **Voting weights.** `evaluate_detectors.py` sets the weighted-vote weights from binary F1 on the same 20% held-out split used to report every performance number.
2. **Probe bank.** Agent training and final evaluation both sample the same 17,834-flow PortScan bank. Different RNG seeds do not make the flows unseen.
3. **Related, not in your email:** the original (cov0) CNN's best epoch was chosen on the held-out test split (`train_cnn.py`). The hardened CNNs already use a 5% validation slice of the training data.

## 2. Constraint: no source host in the data

The CSVs we use are the MachineLearningCVE files with 77 numeric features. They have no Source IP, Flow ID or Timestamp, so a split by scanner host is not possible. Instead I propose grouping flows that are identical in every feature the simulator does not modify (everything except destination port and the payload-dependent length/byte features) and keeping each group on one side of every split. On the current bank this gives **333 groups, the largest 228 flows, 66 singletons**, so the grouping is not so lumpy that a split is meaningless. This is a proxy for host separation, and the thesis will say so.

If you would prefer a true host split, I can look for the TrafficLabelling version of CIC-IDS2017 (with Source IP and Timestamp) and realign it, at extra cost.

## 3. Partitions

The 20% held-out set (504,160 rows) is split in two. BENIGN and other-attack rows are split 50/50, stratified by class. PortScan rows are split by group.

| Partition | Contents | Used for |
|---|---|---|
| Validation side | half of held-out rows | F1 and voting weights; any threshold choice |
| Agent-train bank | ~70% of bank groups on the validation side | DDQN training only |
| Dev bank | ~30% of bank groups on the validation side | selecting the best fixed payload; configuration checks |
| Final-test | the other half of the held-out rows, including its bank groups | every reported number |

Detectors are not retrained: they only ever saw the 80% training split.

## 4. Proposed rerun scope

| Option | Runs | Rough time |
|---|---|---|
| **Focused (recommended)** | cov0 and cov1 × agent seeds 0–4 × 14 experiments (140); cov1 detector seeds 1–3 × agent seed 0 (42); cov2 and cov3 at agent seed 0 only (28) | ~1 h |
| Full | all four sets × 5 seeds, plus detector-seed runs (322) | ~2 h |

Focused is justified because every cov2 and cov3 run ended at 0% evasion in the earlier replication, and their outcome is fixed by detector coverage, which I will measure directly on the final-test partition.

## 5. Questions for you

1. Is the group-disjoint split acceptable as the answer to "separate the underlying source flows", given the data limitation in section 2?
2. Focused or full rerun?
3. Should I retrain the cov0 CNN with a train-side validation slice (one CNN run), or leave it as a disclosed limitation?
4. For the Raspberry Pi slowdown figures, is standalone XGBoost an acceptable named baseline?
