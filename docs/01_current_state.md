# 01 — Current State of the Project (as of 2026-10-14)

Working title: *Intrusion Detection: Evaluating Adversarial Resilience and
Convergence Dynamics of Multi-Model Ensemble Systems on Edge Hardware*

This document records what exists before the detector-diversity experiment
(documented in `03_diversity_experiment_design.md`). The dated, step-by-step
record of every run is in `PROJECT_LOG.txt`; this file is the consolidated
summary to write the methodology and results chapters from.

---

## 1. Research questions

| RQ | Question | Evidence that answers it |
|---|---|---|
| RQ1 | How does the DDQN agent's evasion success differ between an ensemble and its standalone member models? | Standalone defences (XGBoost, RF, CNN) vs ensemble rules, same agent and probes |
| RQ2 | Which aggregation rule (majority, unanimous, confidence-weighted) resists the agent best? | Voting rules compared on identical probes (paired) and with trained agents |
| RQ3 | What does multi-model inference cost on edge hardware (latency, memory, throughput)? | `benchmark_edge.py` (Mac dry run done; Raspberry Pi run deferred) |

---

## 2. Pipeline overview

```
CIC-IDS2017 CSVs ──► train_models.py ──► RF, XGBoost, scaler, label encoder   (code/models/)
                └──► train_cnn.py    ──► 1D-CNN                                (code/models/)
                └──► build_probe_bank.py ──► 17,834 held-out PortScan flows    (code/models/probe_bank.npz)
                └──► train_adversarial.py ──► retrained detectors              (code/models_adv/)

evaluate_detectors.py ── held-out accuracy / recall / false-positive rate, robustness to padded probes
ensemble.py           ── 3 detectors + 7 decision rules
ids_env.py            ── Gymnasium environment: each step = one probe built from a real held-out flow
train_ensemble.py     ── Double DQN agent, one process per (defence × knowledge level)
evaluate_agents.py    ── untrained baseline (paired across rules) + greedy evaluation of trained agents
plot_results.py, plot_comparison.py ── thesis figures
benchmark_edge.py     ── RQ3 latency / throughput / memory
project_log.py        ── appends [auto] entries to docs/PROJECT_LOG.txt
```

The detector set used by every script is selected with the environment
variable `IDS_MODEL_DIR` (default `code/models`). The probe bank always comes
from `code/models/probe_bank.npz`, so every detector set is attacked with the
same held-out flows.

---

## 3. Dataset and preprocessing

- **CIC-IDS2017**, all 8 MachineLearningCVE CSV files: 2,830,743 raw rows.
- Cleaning (identical in every script, `build_probe_bank.load_and_clean_data`):
  strip column names, drop the duplicate column `Fwd Header Length.1`,
  remove ±inf/NaN rows, drop duplicate rows → **2,520,798 rows**.
- **77 numeric features** (78 is often quoted; one is the duplicate column).
- **15 classes** (BENIGN + 14 attack types). `LabelEncoder` sorts
  alphabetically, so index 0 = BENIGN.
- Stratified 80/20 split, `random_state=42`: **2,016,638 train /
  504,160 held-out test rows**. `StandardScaler` fit on the training split.
- The split was reconstructed and checked against the saved scaler (means
  and variances match exactly), so every later script uses the same test set.

---

## 4. Detectors

| Detector | Configuration | Source |
|---|---|---|
| XGBoost | `tree_method="hist"`, library defaults otherwise, `random_state=42` | `train_models.py` |
| Random Forest | 100 trees, `random_state=42` | `train_models.py` |
| 1D-CNN | 3 Conv1d blocks (64/128/128 channels, kernel 3, BatchNorm, ReLU) → global average pool → FC 128 → 64 → 15; dropout 0.3; Adam lr 1e-3; batch 512; 30 epochs; class-balanced sampling | `cnn_arch.py`, `train_cnn.py` |

All three output 15-class probabilities. The ensemble uses them as a binary
decision: **flag as attack if P(BENIGN) < 0.5**.

### Held-out performance (504,160 test rows)

| Detector | Accuracy (15-class) | Attack recall | False-positive rate | PortScan caught |
|---|---|---|---|---|
| XGBoost | 99.89% | 99.87% | 0.07% | 99.99% |
| Random Forest | 99.82% | 99.51% | 0.07% | 98.85% |
| 1D-CNN | 96.61% | 94.46% | 2.85% | 95.13% |

Source: `code/models/detector_eval.json`.

---

## 5. Defences compared (7)

Defined in `code/ensemble.py` (`EnsembleIDS.decide`).

| Defence | Flags a probe as an attack when… | Role |
|---|---|---|
| `xgboost`, `rf`, `cnn` | that single detector flags it | RQ1 standalone baselines |
| `unanimous` | all 3 detectors flag it | most permissive rule |
| `majority` | at least 2 of 3 flag it | RQ2 |
| `weighted` | Σ wᵢ · P(attack)ᵢ ≥ 0.5 | RQ2 (confidence-weighted) |
| `any` | at least 1 of 3 flags it | strictest rule |

**Weighted rule.** Each detector contributes its attack probability
P(attack) = 1 − P(BENIGN), multiplied by a weight proportional to its
held-out binary F1 score (`evaluate_detectors.py`, saved as
`ensemble_weights.json`). Original detectors: XGBoost 0.344, RF 0.343,
CNN 0.313. Because the weights are nearly equal, the rule behaves like a
probability-averaged majority vote; one confident detector alone (≈0.34)
cannot reach the 0.5 threshold.

`decide()` is separate from `classify()` so one set of detector outputs can
be scored under every rule. This makes rule comparisons **paired**: every
rule sees exactly the same probes.

---

## 6. Simulation environment (`code/ids_env.py`)

- Each environment step is one scan probe against a new destination port;
  an episode is 100 probes.
- **Real-flow simulator (from v2 on).** Every probe starts from a real
  PortScan flow drawn from the probe bank: 17,834 single-probe flows (one
  forward packet, at most one reply) taken from the *held-out test split*,
  so no detector has seen them. Only the features the agent's choice
  actually changes are recomputed, using CICFlowMeter's own formulas
  (`code/flow_features.py`), validated to reproduce 116,998 real
  single-packet flows exactly.
- **Agent choices.** 1,250 discrete actions = 50 inter-probe delays × 5 TCP
  flag options × 5 payload sizes (0, 64, 128, 256, 512 bytes). In the
  simulator only the payload size changes the flow record. The delay is the
  gap *between* separate flows, so it never enters a per-flow feature; it
  only costs reward. The flag columns in CIC-IDS2017 don't follow TCP
  semantics (99.9% of scan flows record PSH=1 only), so flags keep the real
  flow's recorded values.
- **Reward.** +1 − 0.1 · delay/100 if the defence labels the probe benign;
  −1 if it flags it.
- **Knowledge levels.**
  - *Reward-only*: the observation is always zeros. The agent learns only
    from the reward (detected or not).
  - *Feature-observing*: the observation is the probe's 77 flow features,
    standardised with the detectors' scaler and clipped to ±10σ. The agent
    sees flow features, not model parameters or gradients, which is why this
    is not called "white-box" (see `00_terminology.md`).

---

## 7. Agent (`code/train_ensemble.py`)

- **Double DQN** (van Hasselt et al., 2016): a `DoubleDQN` subclass of
  Stable-Baselines3 `DQN` (v2.7.1). The online network selects the next
  action and the target network evaluates it. SB3's own DQN uses the vanilla
  max over the target network.
- Hyperparameters: MLP [256, 256], lr 1e-4, replay buffer 100,000,
  learning starts 2,000, batch 128, γ = 0.99, train every 4 steps,
  target update every 1,000 steps, ε annealed 1.0 → 0.05 over the first 20%
  of training, **50,000 steps = 500 episodes**, seed 0.
- One process per experiment. 7 defences × 2 knowledge levels = 14
  experiments per detector set.

---

## 8. Evaluation protocol (`code/evaluate_agents.py`)

1. **Untrained baseline.** 5,000 uniformly random actions, each scored
   under all 7 rules on the same probe (paired), seed 12345. This is the
   evasion rate with no learning.
2. **Trained agents.** Each final agent runs greedily (no exploration) for
   20 episodes (2,000 probes) on seeds that weren't used for training.
3. **Training curves.** Per-episode evasion rate during training
   (`convergence_ddqn_{rule}_{knowledge}.json`).

Detection of padded probes is also measured with no agent at all: the full
probe bank padded to each payload size, scored by each detector
(`evaluate_detectors.py`).

---

## 9. Run history

### Pre-v1 runs — sparse feature vectors (invalid, archived)
`results_flat_baseline_archive/`, `checkpoints_flat_baseline_archive/`,
`convergence_ensemble_*.json`, `dqn_ensemble_*/` (Sept 24 – Oct 13).
The early simulator filled only 9 of 77 features and left the rest at
zero. The detectors read that as idle benign traffic, so evasion was 100%
from the first episode and nothing was learned. The fix was to ground
probes in real flows.

### v1 — heuristic "blend" simulator (invalid, archived)
`results_ddqn_v1_blend_sim_archive/`. Probes were interpolations between
the median PortScan and median BENIGN feature vectors, with hand-set weights.
**Invalid**: an untrained random policy already reached 93.9–99.0%
evasion, the decision boundary flipped after a 3–10% shift, delay was
wrongly injected into per-flow features, the feature-observing agent's observations were
unscaled, and the threat model treated unanimous voting as strict.

### v2 — real-flow simulator, original detectors (seed 0)
Results: repository root (`convergence_ddqn_*.json`, `eval_results.json`,
agents in `models/`). Detectors: `code/models/`.

Detection of padded held-out probes (17,834 per cell; each flow keeps its
recorded destination port):

| Detector | 0 B | 64 B | 128 B | 256 B | 512 B | random 1–1460 B |
|---|---|---|---|---|---|---|
| XGBoost | 87.4% | 0.6% | 0.6% | 3.1% | 3.1% | 2.8% |
| RF | 100.0% | 0.0% | 0.0% | 0.0% | 5.7% | 1.2% |
| CNN | 99.9% | 21.4% | 33.4% | 14.0% | 0.7% | 5.7% |

At the ports the simulator actually uses (1001–1100), detection is lower
still: CNN 3.1% / 41.2% / 0.2% / 0.0% at 64 / 128 / 256 / 512 B, and
XGBoost and RF 0.0% at every non-zero size (`02_voting_rule_metrics.md`).

Evasion rate (share of probes labelled benign):

| Defence | Untrained (random) | Trained reward-only | Trained feature-observing |
|---|---|---|---|
| XGBoost | 83.4% | 100.0% | 100.0% |
| RF | 80.5% | 100.0% | 100.0% |
| CNN | 72.1% | 100.0% | 58.5%* |
| Unanimous | 83.4% | 100.0% | 100.0% |
| Majority | 80.5% | 100.0% | 100.0% |
| Weighted | 80.5% | 100.0% | 100.0% |
| Any | 72.1% | 100.0% | 58.5%* |

\* The feature-observing agent settled on an action below the best available one (a local optimum).
The reward-only agent shows the defence itself is fully evadable, so this is
an agent-side optimisation failure, not resilience.

### v3 — adversarially trained detectors (seed 0)
Results: `results_v3_adversarial/`. Detectors: `code/models_adv/`
(`train_adversarial.py`). The original training split plus 71,381
payload-padded copies of the training-split single-probe PortScan flows
(payload drawn uniformly from 1–1460 B, seed 2026). Split, scaler,
label encoder and hyperparameters are unchanged; the test split and probe
bank stay untouched.

| Detector | Accuracy | Attack recall | False-positive rate | Padded probes caught (all sizes) |
|---|---|---|---|---|
| XGBoost | 99.89% | 99.86% | 0.07% | 99.99–100% |
| RF | 99.82% | 99.49% | 0.07% | 99.99–100% |
| CNN | 96.20% | 95.45% | 3.57% | 99.97–100% |

Evasion: **0.0% for all 7 defences**, both knowledge levels (untrained:
1 of 5,000 random probes, 0.02%). A follow-up check showed the retrained
detectors do not depend on the scanner's TCP window value to catch padded
probes.

---

### v4 — controlled detector diversity
Design: `03_diversity_experiment_design.md`. Results:
`04_diversity_results.md`. Detectors: `code/models_cov1/`,
`code/models_cov2/`. Results: `results_v4_cov1/`, `results_v4_cov2/`.

---

## 10. Findings so far (before v4)

1. **Shared blind spot.** Every CIC-IDS2017 PortScan flow carries 0–2
   payload bytes, so all three detectors learned "non-zero payload ⇒
   benign". The weakness comes from the training data, so every model
   trained on it inherits it.
2. **Ensemble collapse (v2).** On the probes the simulator sends, the
   detectors' detections are nested: every padded probe XGBoost catches,
   RF also catches, and every one RF catches, the CNN also catches (at most
   0.01% exceptions). Pairs of detectors are wrong together on 91–100% of
   padded probes. As a result each voting rule reproduces exactly one
   member, with identical training curves (500/500 episodes):
   unanimous ≡ XGBoost, majority ≡ weighted ≡ RF, any ≡ CNN. The one
   partial exception is unanimous vs XGBoost feature-observing (370/500 identical).
   Voting adds no resilience beyond its members.
3. **Adversarial training closes the gap (v3).** All 7 defences reach 0%,
   but again the voting rule makes no difference, because each member is
   already robust on its own.
4. **Implication for RQ2.** The rules can only behave differently when the
   members make *different* mistakes. Neither v2 (same mistake everywhere)
   nor v3 (no mistakes) has that property. This motivates the diversity
   experiment (`03_diversity_experiment_design.md`).
5. **Cost of the CNN.** The CNN has a 40× higher false-positive rate than
   the trees (2.85% vs 0.07%). Rules that let the CNN decide alone (`any`)
   inherit that cost (quantified in `02_voting_rule_metrics.md`).

### RQ3 (Mac dry run, Apple M-series, 2,000 flows)
XGBoost p50 0.40 ms, CNN 0.41 ms, RF 2.06 ms, full 3-model ensemble
2.87 ms (≈3× a single fast model). RF dominates latency and memory (662 MB
with all three loaded). The Raspberry Pi run is deferred.

---

## 11. Limitations to state in the thesis

- **One training seed** per experiment so far (no variance estimates).
- **Small effective action space.** Only the 5 payload sizes change the
  flow record, so the agent's problem is close to choosing the best of
  5 options. Delay only affects reward; flags aren't simulated.
- **Fixed scanner constants.** Header lengths, reply window and flow
  duration are inherited from the real scan flows and never varied. Of
  these, only the TCP window was checked (v3).
- **Feature-observing local optimum** in v2 (CNN / any).
- **Dataset-specific shortcut.** Conclusions are about detectors trained on
  CIC-IDS2017, and may not transfer to other datasets.

---

## 12. Where things are

| Path | Contents |
|---|---|
| `code/` | All pipeline scripts |
| `code/models/` | Original detectors, scaler, label encoder, probe bank, F1 weights, `detector_eval.json` |
| `code/models_adv/` | Adversarially trained detectors (v3), `augmentation.json` |
| `code/models_cov1/`, `code/models_cov2/` | v4 partially hardened detector sets (see `03_diversity_experiment_design.md`) |
| `results_v4_cov1/`, `results_v4_cov2/` | v4 results, agents, figures, training logs |
| `summarize_results.py` (root) | Cross-set tables and figures → `docs/results_summary.*`, `docs/figures/` |
| repo root `convergence_ddqn_*.json`, `eval_results.json`, `models/` | v2 results and trained agents |
| `results_v3_adversarial/` | v3 results, agents, figures, training log |
| `thesis_*.png` (root) | v2 figures and the v2-vs-v3 comparison figure |
| `results_ddqn_v1_blend_sim_archive/` | v1 (invalid) |
| `results_flat_baseline_archive/`, `checkpoints_flat_baseline_archive/` | pre-v1 sparse-feature runs (invalid) |
| `docs/` | This documentation and `PROJECT_LOG.txt` |
