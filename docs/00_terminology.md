# 00 — Terminology

Fixed terms for the thesis. `code/terminology.py` holds the same labels for scripts, tables and figures. `code/check_terminology.py` fails if a retired term appears in a thesis-facing file.

## Agent knowledge levels

| Use | Not | Meaning |
|---|---|---|
| **reward-only agent** | black-box | Sees only the reward (detected or not). Its observation is always zeros. |
| **feature-observing agent** | white-box | Also observes the probe's 77 flow features, standardised with the detectors' scaler. It sees flow features, **not** model parameters, gradients or outputs, so "white-box" would overstate what it knows. |

The internal keys `black` and `white` remain in file names (`convergence_ddqn_<rule>_white.json`, `ddqn_<rule>_white.zip`), result dictionaries and command-line choices. Renaming them would orphan trained agents and archived results. They never appear in prose. A third level, `gray` (a 4-feature subset), exists in `ids_env.py` and is used in no reported experiment.

## Experiments

| Use | Meaning |
|---|---|
| **constrained flow-feature simulation** | Each probe is a real CIC-IDS2017 PortScan flow whose payload-dependent features are recomputed. No packets are sent. Claims about packet-level behaviour are not made unless packet-level validation supports them. |
| **effective action space: 5 payload sizes** | The interface has 1,250 actions (50 delays × 5 flag options × 5 payload sizes). Only payload size changes the simulated detector input. Flags are not simulated and delay only affects reward. |
| **agent-seed experiment** (Part A) | Five agent seeds on each of four detector sets. Detector training is fixed. |
| **detector-seed experiment** (Part B) | Four training runs of one detector set, each attacked with agent seed 0 only. |

Part A and Part B are separate experiments. They are **not** a fully crossed design: no combination of several agent seeds with several detector seeds was run.

## Data separation

| Use | Meaning |
|---|---|
| **invariant-feature group** | Rows identical in every feature the simulator does not modify. All rows of a group stay in one partition. |
| **separation by invariant-feature groups** | The only independence claim made. It does **not** establish independence by host, capture session or time (the CSVs have no source IP or timestamp). |
| **train / validation / final test** | The three partitions. Voting weights and thresholds come from validation. Final test is read only after the protocol is frozen. |
| **agent-train bank / dev bank / final-test bank** | Probe-flow partitions. Agents train on the first, fixed-payload choices use the second, results use the third. |

## Measurements

| Use | Meaning |
|---|---|
| **detection rate** | Share of padded probes a detector or rule flags, with no agent involved. Reported as k/n. |
| **evasion rate** | Share of probes labelled benign during a trained agent's greedy evaluation. |

Never use the two interchangeably. Detection rates describe detectors; evasion rates describe detector–agent pairs.

## Claims about detectors

| Use | Not |
|---|---|
| **strongest observed standalone comparator** (under the tested conditions) | optimal, best possible |
| exact counts, e.g. 17,826 of 17,834 | "99.9–100%" as a robustness guarantee |

Phrase the central result as: *the tested ensembles did not outperform their strongest observed standalone member.*

## Retired terms

`white-box`, `black-box`, `optimal` (as a description of a detector or a policy). The dated project log (`PROJECT_LOG.txt`), the first build notes and files under `docs/logs/` are historical records and keep their original wording.
