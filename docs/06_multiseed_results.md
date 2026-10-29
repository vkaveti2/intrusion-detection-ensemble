# 06 — Multi-Seed Replication (Step 3): Results

> **Superseded by the corrected protocol (v5).** This document records the first pipeline: detectors trained with `PSH Flag Count` visible, a train/test split that is not group-disjoint from the probe bank, and (for the original CNN) epoch selection on the test split. In particular its statements that majority, weighted and any were never evaded under partial hardening, and that tree robustness spreads to every payload size, held only with `PSH Flag Count` visible. The numbers below are kept as the diagnostic comparison that motivated the correction (`09_psh_diagnostic_design.md`, `12_frozen_protocol.md`). **Reported results are in `v5_results.md`.**

Run 2026-10-15, 00:43–02:20, with `code/run_step3.sh`. Design and
predictions: `05_multiseed_design.md`, written before any result was
inspected. Full generated tables: `docs/multiseed_summary.md`. Raw
per-run data: `docs/multiseed_runs.csv`, `docs/multiseed_model_seeds.csv`.
Figures in `docs/figures/` (listed in section 7).

**Short version.** The findings replicate:
- **Every hardened defence held in every run.** All 190 agents trained
  against a hardened defence ended at exactly 0.0% evasion.
- **Every defence that was evadable at seed 0 stayed evadable**, and the
  reward-only agent found its best option in all 45 such runs.
- **The learning-speed differences are consistent** across seeds (ranges
  don't overlap, p = 0.008).
- **Feature observation did not help the agent.** It was detected more
  often in every configuration and fell short of the best option in 6 of
  45 runs (3 distinct cases; collapsed pairs train identically, so each
  case appears twice).

Retraining cov1 with new model seeds showed that the CNN's leftover gap
appears in every training run, but its position changes from run to run.
Unanimous voting inherited it every time; majority, weighted and any never
did.

---

## 1. What ran

The two parts below are **separate experiments, not a fully crossed design**.
Part A varies the agent seed with the detectors fixed. Part B varies the
detector training seed with the agent seed fixed at 0. No combination of
several agent seeds with several detector seeds was run.

| Part | Runs | Status |
|---|---|---|
| A: agent seeds 1–4 × 4 detector sets × 14 experiments | 224 new runs (plus 56 existing seed-0 runs = 280) | all complete; 42 real-model loads per batch, no mock or error lines |
| B: cov1 retrained with model seeds 1, 2, 3, then 14 experiments each (agent seed 0) | 3 detector sets, 42 runs | all complete; every CNN ran all 30 epochs |

Checks behind the design:
- **Determinism.** Re-running seed 0 reproduced the original result in all
  500 episodes, so the existing results serve as seed 0.
- **XGBoost and model seeds.** XGBoost's 1,500 trees are identical across
  all four model seeds (no row/column subsampling), so in Part B only RF
  and the CNN vary.

## 2. Part A — agent seeds

### 2.1 Defence outcomes are reproducible

Final greedy evasion, mean [min–max] over 5 agent seeds:

| Defence | cov0 black | cov0 white | cov1 black | cov1 white |
|---|---|---|---|---|
| XGBoost | 100.0% [100–100] | 99.8% [99.1–100] | 0.0% | 0.0% |
| RF | 100.0% [100–100] | 100.0% [100–100] | 0.0% | 0.0% |
| CNN | 99.4% [97.5–100] | 82.7% [58.5–99.8] | 94.5% [94.5–94.5] | 75.1% [0.9–94.5] |
| Unanimous | 100.0% [100–100] | 99.8% [99.1–100] | 94.5% [94.5–94.5] | 75.1% [0.9–94.5] |
| Majority | 100.0% [100–100] | 100.0% [100–100] | 0.0% | 0.0% |
| Weighted | 100.0% [100–100] | 100.0% [100–100] | 0.0% | 0.0% |
| Any | 99.4% [97.5–100] | 82.3% [58.5–99.8] | 0.0% | 0.0% |

cov2 and cov3: **0.0% for every defence, agent type and seed**.

- **Every hardened defence held in all 190 runs**: cov1 XGBoost, RF,
  majority, weighted and any, plus all of cov2 and cov3. Its robustness
  does not depend on the agent's luck.
- **Every evadable defence stayed evadable.** The reward-only agent reached
  the best available outcome in **all 45** such runs (A2 in the summary).
  Its final evasion varied by at most 2.5 points between seeds.

### 2.2 Other replicated measurements

The pre-specified comparisons C1–C3 were consistent across all seeds:
the groups' ranges did not overlap, and p = 0.008 for each, the smallest
value possible with 5 vs 5 runs. C4 is a count: 6 of the 45 feature-observing runs
against evadable defences ended below the best available outcome (3
distinct cases, each duplicated by a collapsed pair), and no reward-only run
did. The numbers are in `multiseed_summary.md`, sections A2–A4,
and `figures/multiseed_learning_speed.png`.

### 2.3 Ensemble collapse across seeds (cov0)

- **Majority and weighted** behaved exactly like RF in every seed and for
  both agent types (identical training in all 500 episodes).
- **Any ≡ CNN and unanimous ≡ XGBoost** were identical in most runs. In
  seed 3 (and one feature-observing run of seed 0), a single probe was decided
  differently by the pair. That is a rare exception to the nested
  detections (nesting statistics in `02_voting_rule_metrics.md`), and
  training diverged from that point (episode 66 and 368). Final results
  still matched in every run but one (any vs CNN, feature-observing, seed 3:
  97.1% vs 98.9%).

The collapse finding stands. It is near-exact rather than exact, which is
how it should be described in the thesis.

## 3. Part B — detector seeds (cov1 retrained 3 more times)

Same training data, bands and padded rows; only training randomness
differs (`docs/figures/detector_seed_coverage.png`).

| Detector | Detection of padded probes across the 4 training runs (model seeds 42, 1, 2, 3) |
|---|---|
| XGBoost (band A) | 100% at every payload size, every run (identical model) |
| RF (band B) | 100% at every agent payload size, ≥ 99.7% at random sizes, every run |
| CNN (band C) | 100% inside its own band in every run. **Every run left a gap at smaller payloads**: lowest detection 5.5%, 16.9%, 0.5% and 21.2%. How wide the gap is, and where it lies, differs between runs |

| Defence | Trained-agent evasion, all 4 training runs (agent seed 0) | False-positive rate (range) |
|---|---|---|
| Majority | **0.0% in every run** (both agent types) | 0.070–0.081% |
| Weighted | **0.0% in every run** | 0.084–0.097% |
| Any | **0.0% in every run** | 3.56–4.71% |
| XGBoost, RF | 0.0% in every run | 0.068% / 0.070–0.073% |
| CNN | evaded in every run (reward-only 79.3–98.7%) | 3.55–4.71% |
| Unanimous | evaded in every run (reward-only 79.3–98.7%) | 0.050–0.058% |

Conclusions from Part B:
1. **Trees.** Their robustness transferring to every payload size is
   reproducible (RF in 4 of 4 runs; XGBoost is deterministic).
2. **CNN.** A CNN hardened only on large payloads **always** left a gap at
   smaller ones, but **where** the gap falls depends on the training run.
   Its false-positive rate also varies by run (3.5–4.7%).
3. **Rules.** Unanimous inherited the CNN's gap in every run; majority,
   weighted and any covered it in every run. The v4 result was not an
   artefact of one CNN.

## 4. Predictions from `05_multiseed_design.md`: verdicts

| # | Prediction | Verdict |
|---|---|---|
| 1 | Reward-only final evasion stable across seeds (± 2 points) | **Held.** Largest spread 2.5 points (cov0 CNN/any, one seed at 97.5%) |
| 2 | Feature-observing sometimes falls short of the best option | **Held.** 6 of 45 runs (3 distinct cases) |
| 3 | Collapse exact in every seed | **Mostly.** Exact for majority/weighted; any/unanimous diverged after a rare differing probe in 1 of 5 seeds, with final results matching (section 2.3) |
| 4 | C1, C2 consistent | **Held** (p = 0.008) |
| 5 | C3 consistent | **Held** in all 9 configurations (p = 0.008) |
| 6 | Part B: does the cov1 CNN gap reproduce? | **Yes, but it changes.** Present in 4 of 4 runs; its width and position vary |

## 5. Conclusions as first drawn, and what the corrected protocol changed

*First drawn (PSH visible):*
- **RQ1.** No ensemble rule was ever more robust than its most robust member; majority, weighted and any were never evaded when at least two members were robust; unanimous was evaded whenever its weakest member had a gap.
- **RQ2.** Majority (or near-equal confidence weighting) was the best robustness / false-alarm trade-off; *any* matched it in robustness at 50-60 times the false alarms; unanimous was the least robust.
- **Partial hardening.** Tree robustness spread to every payload size in every training run; the CNN's did not.

*What changed under the corrected protocol (`v5_results.md`, `10_band_rotation.md`):*
- **RQ1 stands, and is stated more carefully.** The tested ensembles did not outperform their strongest observed standalone member. XGBoost alone is never evaded in any hardened set and flags 13,569 of 13,569 padded probes at every agent size in every set; no rule beats it.
- **The partial-hardening claim is overturned.** With `PSH Flag Count` removed, only XGBoost still covers every payload size. RF and the CNN keep gaps that depend on the training run, and *majority is not robust under partial hardening*: it is evaded 922 of 2,000 times in every cov1 run of the main sweep, because RF shares the gap at 64 B. Weighted voting is evaded less (729 of 2,000 for the reward-only agent), `any` never, unanimous inherits the CNN's gap.
- **RQ2 is restated.** Under full hardening every rule resists. Under partial hardening the ordering is any (never evaded, at about 2.7% false alarms against 0.07-0.08%) above weighted above majority, with unanimous the least robust. The earlier claim that majority insures against an unpredictable member holds only when at least two members cover the weak size; the band-rotation diagnostic shows which sizes those are.
- **Feature observation did not help the agent**, as before, and the feature-observing agent was less consistent than the reward-only agent.

## 6. Limitations that remain

- One band assignment, and the padding seed (2026) was never varied.
- The reliance on *PSH Flag Count* was checked and found substantial (`09_psh_diagnostic_design.md`); the corrected protocol removes it.
- Why hardening transfers is still unverified.
- Small effective action space; a single dataset (CIC-IDS2017).
- The inference-cost results are in `13_benchmark_results.md`.

## 7. Figures and tables

| File | Shows |
|---|---|
| `figures/multiseed_convergence_cov0.png`, `..._cov1.png` | Training curves, mean of 5 seeds with min–max band |
| `figures/multiseed_learning_speed.png` | Per-seed values behind C1–C3 |
| `figures/detector_seed_coverage.png` | Part B coverage for the 4 cov1 training runs |
| `multiseed_summary.md` | Every table (A1–A5, B1–B2) |
| `multiseed_runs.csv`, `multiseed_model_seeds.csv` | Raw per-run numbers |