# 14 — Methodology (chapter draft)

Draft prose for the methodology chapter. Every design choice here is recorded in a frozen configuration (`12_frozen_protocol.md`, `config/protocol_v5.json`) and every figure traces to a generated file. Terms follow `00_terminology.md`. Section numbers are provisional.

## 3.1 Research questions

| | Question | Evidence |
|---|---|---|
| RQ1 | How does a reinforcement-learning attacker's evasion success against a multi-model ensemble differ from its success against each standalone member? | Standalone detectors and ensemble rules scored on the same probes, with and without a trained agent |
| RQ2 | Which aggregation rule (majority, unanimous, confidence-weighted, any) resists the attacker best, and at what cost in false alarms? | The four rules compared on identical probes (paired) and against trained agents |
| RQ3 | What does multi-model inference cost on constrained hardware: latency, throughput, memory? | Timing and memory measurements of each standalone detector and the ensemble |

The central finding the methodology is designed to test fairly is stated in the results chapter: the tested ensembles did not outperform their strongest observed standalone member.

## 3.2 Scope: a constrained flow-feature simulation

No packets are sent. Each probe is a **real CIC-IDS2017 PortScan flow** (one forward packet, at most one reply) whose payload-dependent flow features are recomputed for the payload size the attacker chooses, using CICFlowMeter's own arithmetic (`flow_features.py`, validated to reproduce 116,998 real single-packet flows exactly). The detectors score the resulting 77-feature record.

The attacker's interface has 1,250 actions (50 inter-probe delays × 5 TCP-flag options × 5 payload sizes). In this simulation only the payload size changes what the detectors see: the flag columns of CIC-IDS2017 do not follow TCP semantics (99.9% of scan flows record the same PSH value), so flags keep the real flow's recorded values; and the delay separates flows rather than appearing in any per-flow feature, so it affects only the reward. The **effective action space is therefore five payload sizes** (0, 64, 128, 256, 512 bytes). Results are claims about this constrained simulation and not about live traffic; packet-level validation would be needed for broader claims.

## 3.3 Data and preprocessing

CIC-IDS2017, all eight MachineLearningCVE files: 2,830,743 rows, reduced to **2,520,798** after dropping the duplicate column `Fwd Header Length.1`, rows with infinite or missing values, and duplicate rows. 77 numeric features, 15 classes (BENIGN plus 14 attack types). The binary decision used throughout is *attack if P(BENIGN) < 0.5*.

**Masked feature.** `PSH Flag Count` is removed from every detector's input (the scaler outputs a constant for it, in training and at every inference call). This follows the diagnostic in §3.9: hardened detectors relied on it, and CIC-IDS2017 records it unreliably (99.99% of the scan flows used here have PSH = 1).

## 3.4 Data separation

The CSVs carry no source IP, flow identifier or timestamp, so separation by scanner host, capture session or time is impossible. The protocol instead separates by **invariant-feature groups**: each row gets a signature, a hash of the 60 features that the simulator neither overwrites nor masks, and rows with equal signatures always share a partition. **This does not establish independence by host, capture session or time.**

Scope of the guarantee. Each of the 89,215 single-probe PortScan flows defines a group by its signature (480 groups; largest 991 flows, 135 singletons); rows of any class sharing one of those signatures follow the group; all other rows are split stratified by class and are not covered by the guarantee (of 364,593 such final-test rows, 47,339 share a signature with a training row; this is ordinary duplicate-feature overlap and is reported, not removed).

| Partition | Rows | Groups | Probe flows | Use |
|---|---|---|---|---|
| train | 1,764,598 | 339 | 62,505 | detector training, adversarial augmentation, scaler fit |
| validation | 377,863 | 69 | 13,141 | voting weights, any threshold |
| final test | 378,337 | 72 | 13,569 | every reported number |

The validation side's probe flows are split by group into an **agent-train bank** (9,325 flows, 55 groups) and a **dev bank** (3,816 flows, 14 groups, used to choose the best fixed payload). Augmented (padded) copies are created *after* the split from train-partition flows only and stay with their source flow. An audit (`split_audit.md`) verifies zero shared signatures between train, validation and final test, between the agent-train bank and everything else, and between the augmented rows and the held-out banks. The final-test partition is unreadable by the code until the configuration is frozen.

## 3.5 Detectors and ensemble rules

Three structurally different detectors, each outputting 15-class probabilities: **XGBoost** (histogram tree method, library defaults), **Random Forest** (100 trees), and a **1D-CNN** (three Conv1d blocks of 64/128/128 channels with batch normalisation, global average pooling, fully connected 128 and 64, dropout 0.3, Adam at 1e-3, batch 512, 30 epochs, class-balanced sampling). The CNN's best epoch and its learning-rate scheduler are driven by an **inner validation slice** (5% of the train partition, chosen by group), never by held-out data; the trees train on every train row.

Seven decision rules are scored on identical detector outputs (a paired design): the three standalone detectors, and four ensemble rules — **majority** (at least 2 of 3 flag), **unanimous** (all 3), **any** (at least 1), and **weighted** (Σ wᵢ·P(attack)ᵢ ≥ 0.5, with wᵢ proportional to detector i's binary F1 on the **validation** partition; the weights are near-equal, 0.31-0.35).

**Hardening designs.** Padded copies of train-partition PortScan flows (payload uniform within a band, one copy per eligible flow, 62,505 per detector) are added to training. Bands: A 1-90 B, B 91-362 B, C 363-1460 B. Four detector sets, one per row of the design:

| Set | Hardening | Training runs |
|---|---|---|
| cov0 | none | model seed 42 |
| cov1 | one band per detector: XGBoost A, RF B, CNN C | model seeds 42, 1, 2, 3 |
| cov2 | two bands per detector: XGBoost A+B, RF B+C, CNN C+A | model seed 42 |
| cov3 | full 1-1460 B for all three | model seed 42 |

Full hardening (cov3) is the robustness reference; partial hardening (cov1, cov2) is a substantive sensitivity analysis.

## 3.6 The attacker

A **Double DQN** agent (a Stable-Baselines3 DQN subclass in which the online network selects the next action and the target network evaluates it): MLP [256, 256], learning rate 1e-4, replay buffer 100,000, learning starts at 2,000 steps, batch 128, γ = 0.99, training every 4 steps, target update every 1,000 steps, ε annealed from 1.0 to 0.05 over the first 20% of 50,000 steps (500 episodes of 100 probes). Reward is +1 − 0.1·delay/100 for a probe labelled benign and −1 for a detected one.

Two knowledge levels, named for what the agent observes: the **reward-only agent** sees nothing but the reward (its observation is always zeros); the **feature-observing agent** also sees the probe's 77 flow features, standardised with the detectors' scaler. Neither sees model parameters, gradients or detector outputs.

Agents train only on the agent-train bank. They are evaluated greedily (no exploration) for 20 episodes (2,000 probes) on the **final-test bank**. An untrained baseline (5,000 uniformly random actions, scored under all rules on the same probes) gives the evasion rate with no learning.

## 3.7 Experiments and seeds

| Part | Design | Runs |
|---|---|---|
| A: agent seeds | cov0-cov3 × agent seeds 0-4 × 14 experiments (7 rules × 2 agent types) | 280 |
| B: detector seeds | cov1 training runs with model seeds 1, 2, 3 × agent seed 0 × 14 | 42 |

Parts A and B are **separate experiments, not a fully crossed design**: Part A holds the detectors fixed and varies the agent seed; Part B varies the detector's training seed with the agent seed fixed at 0. Training is deterministic given its seed.

Two pre-registered analyses on the frozen models. **Fixed-payload comparison:** the best fixed payload size (one size, smallest delay) is chosen on the dev bank and evaluated on the final-test bank, with all five sizes reported as a reference, and compared with the trained agents. **CNN band rotation:** the CNN is hardened on band A or B instead of C (model seeds 1 and 2), with XGBoost and RF held fixed, to test whether the CNN's blind sizes follow its assigned band (design written before any result: `10_band_rotation.md`).

## 3.8 Measures

- **Detection rate** (no agent): the share of padded probes a detector or rule flags, reported as flagged / n.
- **Evasion rate** (with agent): the share of an agent's probes labelled benign, reported as evaded / 2,000. Detection and evasion rates are different quantities and are never mixed.
- **Clean-traffic cost:** false positives and missed attacks on the final-test partition, with rates.
- **Intervals.** The final-test probe bank has 13,569 flows but 72 groups, and flows in a group are near-identical, so binomial intervals would be far too narrow. Detection and clean-traffic rates carry 95% intervals from resampling whole groups (probe tables) or whole signatures (clean-traffic tables), 2,000 resamples (`intervals_v5.py`). These describe variation across groups for a fixed detector, not variation from retraining; Part B and the agent-seed ranges report that separately. Agent results are reported as the five per-seed counts and their range, not as a mean with an interval.

## 3.9 The PSH diagnostic and the correction it motivated

Before the corrected protocol, hardened detectors appeared robust at every payload size. A pre-registered inference-time test (`09_psh_diagnostic_design.md`) set `PSH Flag Count` to 0 on padded probes and to 1 on benign rows. Under the threshold fixed in advance (a detection drop above 5 points, or a benign false-positive rise above 0.5 points), every hardened set failed; `Fwd PSH Flags` had no effect. Retraining without the column preserved full hardening but removed the apparent transfer of partial hardening for RF and the CNN, and with it the robustness of majority voting under partial hardening. The original (PSH-visible) results are reported only as the diagnostic comparison. Whether adversarial training itself introduced the dependence is suggestive, not shown: the unhardened trees catch almost no padded probes, so the attack-side test says nothing about them.

## 3.10 Inference-cost measurement

Method in `13_edge_benchmark.md`. Each of the three detectors and four rules is timed on 2,000 dev-bank flows per repeat, five repeats, in a fresh process each and a shuffled order, single-threaded, batch size 1 (batched throughput is reported but is platform-specific for the CNN). Reported: latency percentiles (p50-max), throughput, peak resident memory, and slowdown relative to **standalone XGBoost**, the named baseline. **Only inference is timed**: packet capture and flow-feature extraction are not measured and not included.

## 3.11 Threats to validity

- **Dataset and simulation.** One dataset (CIC-IDS2017) and a constrained flow-feature simulation. Flag and delay effects, and any behaviour outside the five payload sizes, are not tested.
- **Grouping is a proxy.** Separation by invariant-feature groups is not separation by host, capture session or time.
- **Few groups.** 72 final-test and 14 dev groups; the dev bank is small for choosing the fixed payload (the choice matched the final-test best in all 28 set-and-rule pairs, but that is weak evidence).
- **One training seed** for cov0, cov2 and cov3; four for cov1. Agent results vary by seed (see the ranges), and some agents settle on payloads below the best available.
- **Unexplained mechanism.** Why tree models transfer hardening across payload sizes and the CNN does not is not investigated.
- **Reward estimate.** The DDQN's reward in the fixed-payload comparison is estimated from its evasion rate and median delay, because the evaluation stored the median.
- **Hardware.** Cost measurements come from specific machines; absolute latencies do not transfer, and slowdown ratios transfer better but are still measurements on that hardware.
