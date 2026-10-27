# 09 — PSH Flag Count Diagnostic: Design

Written before any result. Addresses the validity flag in `04_diversity_results.md` section 2 and `06_multiseed_results.md` section 6.

## Why

99.994% of the 17,834 bank flows have `PSH Flag Count` = 1 (column 46), the known CIC-IDS2017 flag artefact. The simulator keeps each flow's recorded flags, and XGBoost ranks this feature second in the cov1 and cov2 sets. If hardened detectors catch padded probes mainly because PSH = 1, the central robustness result could be an artefact of the dataset.

## Interventions (inference only, no retraining)

| Test | Data | Change | Measures |
|---|---|---|---|
| Attack detection | padded probes at 64, 128, 256, 512 B and random sizes, simulator ports | PSH 1 → 0 | change in detection rate |
| Benign false alarms | held-out BENIGN rows | PSH forced to 1 | change in false-positive rate |
| Reference | both | unchanged | baseline; must reproduce `rule_eval.json` |

The same tests run on `Fwd PSH Flags` (column 30). Applied to every detector and every rule in cov0, cov1 (four training runs), cov2 and cov3.

## Decision rule (fixed in advance)

Dependence is **substantial** if either holds for a hardened detector:
- detection of padded probes falls by more than **5 percentage points** when PSH is set to 0, or
- the benign false-positive rate rises by more than **0.5 points** when PSH is set to 1.

## Consequences

- **Substantial:** retrain the affected sets with the column masked (zeroed in all training and padded rows), then repeat coverage and the agent runs and restate the robustness result.
- **Not substantial:** report the table and close the item.

Either way the table is reported for both detection and false alarms, as requested.

---

# Results (2026-10-26)

Detection-level only: no agent has been run against the masked sets yet. Inference sensitivity: `code/psh_sensitivity.py`. Retraining: `code/run_psh_retrain.sh`, with the column removed through `masked_scaler.MaskedScaler` (the scaler outputs 0 for it in training and at every inference call). Seed 42, same data and hyperparameters as the unmasked sets. Logs in `results_psh_masked/`.

## Inference sensitivity (PSH Flag Count 1 → 0 on padded probes; 1 on benign)

Substantial dependence by the pre-set rule in every hardened set tested (cov1, cov2, v3/cov3). The original (cov0) trees showed no change either way (0.1 pt drop, benign false positives 0.07% to 0.07%), but the attack-side test is uninformative for them: they catch almost no padded probes with PSH intact, so there is nothing to lose. Only their benign result says they do not lean on PSH; whether adversarial training introduced the dependence is therefore suggestive, not shown. `Fwd PSH Flags` had no effect anywhere. The inference test was not run on cov1 model seeds 1–3.

## Retrained without PSH: padded probes caught, simulator ports

| Set | Detector | 64 B | 128 B | 256 B | 512 B | random 1–1460 B |
|---|---|---|---|---|---|---|
| v3 (full hardening) | XGBoost / RF | 100% | 100% | 100% | 100% | 100% |
| v3 (full hardening) | CNN | 99.9% | 100% | 100% | 100% | 100% |
| cov1, with PSH (earlier) | XGBoost / RF / CNN | 100 / 100 / 5.5% | 100 / 100 / 99.0% | 100 / 100 / 99.9% | 100% | 100 / 99.8 / 95.4% |
| cov1, no PSH | XGBoost | 100% | 100% | 100% | 100% | 100% |
| cov1, no PSH | RF | **62.2%** | 100% | 100% | 100% | **92.7%** |
| cov1, no PSH | CNN | **0.5%** | **50.1%** | **54.2%** | 100% | **86.2%** |

Rules, cov1 no PSH (64 B): any 100%, weighted 93.9%, majority 62.2%, unanimous 0.5%. At the probes' recorded ports the v3 CNN catches 93.9% at 64 B.

Clean-traffic cost: v3 and cov1 trees unchanged (0.07% false positives). CNN 3.23% (v3) and 3.54% (cov1).

## Reading

1. **Full hardening holds without PSH.** The central robustness result survives for the set where every detector sees padded scans of every size.
2. **Partial-hardening transfer does not.** Without PSH, RF (trained on 91–362 B) no longer covers 64 B, and the CNN's gap widens to 128 B and 256 B. Only XGBoost still transfers to every size. The earlier claim that tree hardening spreads to every payload size was partly a PSH effect.
3. **Voting rules now differ.** At 64 B only one member (XGBoost) is robust, so majority (62.2%) falls short of it, as the t-of-n prediction says. Weighted catches more (93.9%) because it averages probabilities.
4. **No rule beats the best standalone member.** XGBoost alone is 100% at every size; any matches it at 3.55% false positives against 0.07%.

## Open

- Detection rates, not evasion: agent runs on the masked sets are pending.
- One training seed for the full-hardening masked set (replicates for cov1 and cov2 are in the next section).
- Voting weights in the masked directories come from F1 on the reporting split (the data-separation issue, `07_split_plan.md`).

---

# Replication without PSH: cov2 and cov1 model seeds 1–3 (2026-10-26)

`code/run_psh_retrain_rest.sh`; same masked-scaler method, detection rates only, simulator ports, 17,834 probes per cell. Logs in `results_psh_masked/`. The inference test on these sets reads 0 points everywhere, as it must: the scaler outputs a constant for that column, so it is a check that the mask works, not a finding.

## cov1 without PSH, four training runs (model seeds 42, 1, 2, 3)

| Detector / rule | 64 B | 128 B | 256 B | 512 B |
|---|---|---|---|---|
| XGBoost | 100 / 100 / 100 / 100% | 100% in all | 100% in all | 100% in all |
| RF | 62.2 / 100 / 60.8 / 83.8% | 100% in all | 100% in all | 100% in all |
| CNN | 0.5 / 0.8 / 1.5 / 0.0% | 50.1 / 50.0 / 8.0 / 0.7% | 54.2 / 68.8 / 99.7 / 46.5% | 100% in all |
| Majority | 62.2 / 100 / 61.2 / 83.8% | 100% in all | 100% in all | 100% in all |
| Weighted | 93.9 / 100 / 99.3 / 98.9% | 100% in all | 100% in all | 100% in all |
| Any | 100% in all | 100% in all | 100% in all | 100% in all |
| Unanimous | 0.5 / 0.8 / 1.1 / 0.0% | 50.1 / 50.0 / 8.0 / 0.7% | 54.2 / 68.8 / 99.7 / 46.5% | 100% in all |

(Order: seed 42 / 1 / 2 / 3.)

## cov2 without PSH (seed 42)

| Detector / rule | 64 B | 128 B | 256 B | 512 B | random |
|---|---|---|---|---|---|
| XGBoost | 100% | 100% | 100% | 100% | 100% |
| RF | 92.9% | 100% | 100% | 100% | 98.4% |
| CNN | 100% | 99.8% | 58.3% | 100% | 96.6% |
| Majority / weighted / any | 100% | 100% | 100% | 100% | 100% |
| Unanimous | 92.9% | 99.8% | 58.3% | 100% | 95.0% |

## Reading

1. **XGBoost is the only detector that is fully robust in every masked run** (5 of 5 sets, all payload sizes). Its transfer from one or two bands to every size does not depend on PSH.
2. **RF and CNN gaps are real and seed-dependent.** In cov1 the RF gap at 64 B ranges 0% to 38 points, and the CNN gap is wide at 64 B in every run, while its 128 B and 256 B behaviour differs by seed (8% to 100% at 256 B).
3. **Majority is no longer reliably robust under partial hardening.** At 64 B it catches 61–84% in three of four cov1 runs. Weighted voting catches 93.9–100%, because it averages probabilities and one very confident tree can lift borderline cases. *Any* is 100% everywhere. Unanimous inherits the CNN gap in every run.
4. **With two bands per detector (cov2), majority, weighted and any are fully robust**, because at every size at least two detectors cover it. Individual RF and CNN gaps remain.
5. **No rule beats the best standalone member.** XGBoost alone matches every robust rule at 0.07% false positives. *Any* matches it at 3.0–4.6% (CNN-driven).
6. **This changes the earlier statement** (docs 04 and 06) that majority, weighted and any were never evaded in cov1. That held only with PSH present. The ordering any ≥ weighted ≥ majority ≥ unanimous, and the cost ordering, are the same.

## Still open

- Agent evasion on the masked sets is not yet measured. The detection table implies a gap at 64 B for majority in cov1, so the agent should find it; this is a prediction, not a result.
- Masked weights use F1 on the reporting split (to be redone with the split fix).
