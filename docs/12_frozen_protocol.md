# 12 — Frozen Protocol (v5)

Frozen 2026-10-27 14:23 (-04:00). The machine-readable record is `config/protocol_v5.json`; it holds the SHA-256 of every detector file and of the twelve code files that define the protocol. RF models are untracked (150 MB each) and verified against `docs/model_manifest.csv`.

**What frozen means.** The split, the seven detector sets, the voting weights and the decision threshold are fixed. Final-test data may be read only by these frozen models and by policies trained on the agent-train bank. A change to any of them requires setting `"frozen": false`, redoing the affected work, and tagging again. Results produced before the freeze are labelled preliminary.

## Separation

Invariant-feature groups: rows identical in every feature the simulator does not modify share a group and stay in one partition. **This does not establish independence by host, capture session or time.** Details and counts: `07_split_plan.md`, `split_audit.md` (audit passed, 16 checks, fingerprint `deb78d125a24c9c86…`).

| Partition | Rows | Groups | Probe flows |
|---|---|---|---|
| train | 1,764,598 | 339 | 62,505 |
| validation | 377,863 | 69 | 13,141 |
| final test | 378,337 | 72 | 13,569 |

Agent-train bank 9,325 flows (55 groups); dev bank 3,816 flows (14 groups).

## Detectors

Seven sets, all trained on the train partition only with `PSH Flag Count` masked, and with the scaler fitted on train rows only.

| Set | Hardening | Training runs |
|---|---|---|
| cov0 | none | model seed 42 |
| cov1 | one band per detector (XGBoost A, RF B, CNN C) | model seeds 42, 1, 2, 3 |
| cov2 | two bands per detector | model seed 42 |
| cov3 | full 1-1460 B | model seed 42 |

Padded copies come from train-partition probe flows, are created after the split, and stay with their source flow. The CNN picks its epoch and drives its scheduler on a 5% inner-validation slice of train, chosen by group. Trees use every train row.

## Weights and thresholds

Weighted-vote weights are proportional to each detector's binary F1 on the **validation partition**. They are stored per set in `ensemble_weights.json`. The decision threshold is 0.5 and was tuned nowhere. Weights across the seven sets fall in 0.342-0.347 (XGBoost), 0.342-0.346 (RF) and 0.307-0.316 (CNN).

## Experiments

| Part | Design | Runs |
|---|---|---|
| A: agent seeds | cov0-cov3 × agent seeds 0-4 × 14 experiments | 280 |
| B: detector seeds | cov1 training runs with model seeds 1, 2, 3 × agent seed 0 × 14 (model seed 42 is shared with A) | 42 |

Parts A and B are separate experiments, not a crossed design. Each batch is 7 defences × 2 knowledge levels. Agents train on the agent-train bank and are evaluated greedily on the final-test bank. Analyses that follow: fixed-payload comparison (best payload chosen on validation-side probe flows), hardening comparison (cov3 as the robustness reference, cov1 and cov2 as sensitivity analyses), a bounded CNN band rotation, and a local inference benchmark.

## Known limits, stated now

- The dev bank has 14 groups. The final-test bank has 72, so rates computed on its 13,569 flows carry less information than the count suggests; report group-clustered intervals.
- 47,339 of 364,593 final-test rows outside the probe groups share a signature with a train row. The unseen-flow claim applies to PortScan probe flows only.
- The baseline CNN selected epochs on the test split in the earlier pipeline. That pipeline is not used for any number reported from here on.
- Packet capture and feature extraction are not simulated. The experiments are constrained flow-feature simulations.

## Amendment (after the freeze, before any agent run)

Two code files were edited after the freeze: `train_ensemble.py` (agents train on the agent-train bank only; new `--fresh-incomplete` flag skips finished agents and restarts unfinished ones from scratch) and `evaluate_agents.py` (reads the final-test bank; saves exact counts next to each rate). Their hashes in `config/protocol_v5.json` were updated, and the hashes at the freeze are kept under `post_freeze_amendments`. The split, partitions, detector files, voting weights, thresholds and `ids_env.py` are identical to the freeze. No agent had been trained or evaluated when the edits were made.
