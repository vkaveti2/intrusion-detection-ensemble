# 05 — Multi-Seed Replication (Step 3): Rationale and Design

Written 2026-10-15 while the runs were in progress, **before any result was
inspected**. The predictions in section 7 are therefore genuine predictions.
Driver script: `code/run_step3.sh`.

---

## 1. Why replicate

Every RL number in v2–v4 comes from **one training run per configuration**
(seed 0). Deep RL results can change substantially with the random seed,
and a single run is not considered evidence on its own (Henderson et al.,
2018; Agarwal et al., 2021). Step 3 does not test a new idea. It checks
that the existing findings are not luck, and it puts error bars on the
numbers that are genuinely random.

The existing claims fall into three groups:

| Claim | Depends on the agent's seed? | Depends on the detectors' training seed? |
|---|---|---|
| Final evasion rate of a defence (e.g. cov1 CNN 94.5%, everything hardened 0%) | Only if the agent fails to find the best payload option. The ceiling is set by the detectors | Yes. It is a property of the trained models |
| Learning speed: episodes to 90% evasion, probes detected during training (H4) | **Yes, directly** | Indirectly |
| Feature-observing vs reward-only differences (e.g. the v2 feature-observing "local optimum") | **Yes** | Indirectly |
| Ensemble collapse (rule ≡ one member) | No. It follows from the detectors' decisions | Yes |
| The single evadable v4 result (cov1 CNN's 64 B gap) | No | **Yes. It rests on one CNN training run** |

Step 3 therefore has two parts: **agent seeds** (Part A) and **detector
seeds** for cov1 (Part B).

## 2. What a seed controls

| Seed | Set by | Controls | Does not change |
|---|---|---|---|
| Agent seed | `train_ensemble.py --seed` | Q-network initial weights; ε-greedy exploration draws; replay-buffer sampling; the environment's choice of which real probe-bank flow each probe uses | The detectors; the evaluation probes (`evaluate_agents.py` uses fixed seed 12345 for every agent, so evaluation is paired across seeds) |
| Model seed | `train_adversarial.py --model-seed` | RF bootstrap and feature sampling; CNN weight initialisation and the class-balanced sampler's draws; XGBoost `random_state` | Training data, padded rows (padding seed 2026), train/test split, CNN validation split. XGBoost uses no row/column subsampling by default, so its seed should have no effect (checked in Part B) |

## 3. Determinism check (done before launching)

Seed 0 of the existing runs was re-run with the modified script (XGBoost,
reward-only, original detectors). The result was **identical to the original
in all 500 episodes**. Runs are deterministic given the seed, so the
existing seed-0 results are reused and only seeds 1–4 are new.

## 4. Design

**Part A — agent seeds.** 4 detector sets × 5 agent seeds (0–4) × 14
experiments (7 defences × reward-only / feature-observing) = 280 runs, 224 of them new.
Everything else is unchanged: 50,000 steps, same hyperparameters, same
evaluation.

**Part B — detector seeds.** cov1 retrained with model seeds 1, 2, 3 (the
original used 42). Same data, bands and padded rows. For each: full detector
and rule evaluation (measured coverage), then the 14 experiments with agent
seed 0.

| Data | Location |
|---|---|
| Seed 0, cov0 / cov1 / cov2 / cov3 | repository root / `results_v4_cov1/` / `results_v4_cov2/` / `results_v3_adversarial/` |
| Seeds 1–4 | `results_multiseed/<set>/seed_<n>/` |
| cov1 model seeds 1–3: detectors | `code/models_cov1_ms<s>/` |
| cov1 model seeds 1–3: results | `results_multiseed/cov1_ms<s>/` |
| `[auto]` log entries for step 3 | `docs/logs/PROJECT_LOG_step3_auto.txt` (kept out of the main log, about 500 entries) |

## 5. Metrics (per detector set × defence × knowledge level, over 5 seeds)

- **Final greedy evasion**: mean and min–max.
- **Found the best option**: the agent's final evasion is within 5
  percentage points of the best payload option available against that
  defence. That option comes from the no-agent padded-probe table at
  simulator ports, e.g. 94.5% for cov1 CNN at 64 B. Not applicable where no
  option evades.
- **Episodes to 50% / 90% evasion** (15-episode rolling mean of training
  evasion): mean ± SD, min–max, and how many seeds reached it.
- **Probes detected during training** (of 50,000): mean ± SD, min–max.
- **Collapse check**: in each seed, are the training curves of
  unanimous/XGBoost, majority/RF, weighted/RF and any/CNN identical?

## 6. Pre-specified comparisons

| # | Question | Measure | Groups (reward-only unless stated) |
|---|---|---|---|
| C1 | Does the agent learn more slowly when fewer payload options evade? (H4) | Episodes to 90% evasion | cov1 CNN (1 option) vs cov0 CNN (3) and vs cov0 RF (4) |
| C2 | Is it detected more often on the way? (H4) | Probes detected during training | same groups as C1 |
| C3 | Does feature observation help? | Probes detected during training, reward-only vs feature-observing | every configuration where evasion is possible |
| C4 | How often does the feature-observing agent settle below the best option? | Count of seeds with feature-observing final evasion > 5 points below the best option | every configuration where evasion is possible |

**Decision rule.** Report mean ± SD and min–max. Call a difference
*consistent* only if the two groups' min–max ranges do not overlap. Also
report the exact two-sided Mann–Whitney U p-value. With 5 vs 5 runs the
smallest possible p is 0.008, reached only when the groups are completely
separated. The comparison list is short and fixed in advance; no other
corrections are applied.

## 7. Predictions

1. **Reward-only final evasion** is identical across seeds (± 2 points):
   cov0 all 100%; cov1 CNN and unanimous ≈ 94.5%, all others 0%;
   cov2 and cov3 all 0%.
2. **Feature-observing** final evasion sometimes falls short of the best option, as
   in v2 seed 0 (CNN/any 58.5%). How often is unknown.
3. **Collapse** is exact in every seed for reward-only agents. Each pair sees
   identical decisions, so its training is identical.
4. **C1, C2:** the cov1 CNN takes longer and is detected more often than
   cov0 CNN and RF in every seed (non-overlapping ranges).
5. **C3:** the feature-observing agent is detected more often than the reward-only
   agent. This held in every evadable configuration in seed 0, but the
   effect may be small relative to seed noise.
6. **Part B:** no confident prediction. Either outcome is informative:
   - If the 64 B gap appears for every model seed, it is a stable property
     of a CNN hardened only on large payloads.
   - If it disappears for some seeds, the v4 result is a training-run
     effect, and the claim becomes "partial hardening *can* leave a gap".
     The "unanimous is worst" conclusion would then rest on a fragile
     example.

## 8. What would change the conclusions

- Large variation in reward-only final evasion across agent seeds would mean
  the RQ answers must be stated as distributions, not single numbers.
- If Part B shows the 64 B gap is not reproducible, RQ2's "unanimous is
  worst" stays true by construction (a 3-of-3 rule can never be more robust
  than its weakest member). But its empirical demonstration in this thesis
  would depend on a single CNN.
- If C1/C2 ranges overlap, H4 is reported as not supported.

## 9. References

- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D.
  (2018). Deep reinforcement learning that matters. *AAAI 2018*.
- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., & Bellemare,
  M. G. (2021). Deep reinforcement learning at the edge of the statistical
  precipice. *NeurIPS 2021*.
