# 03 — Diversity Experiment (v4): Reasoning and Design

> **Superseded by the corrected protocol (v5).** This document records the first pipeline: detectors trained with `PSH Flag Count` visible, a train/test split that is not group-disjoint from the probe bank, and (for the original CNN) epoch selection on the test split. The hypotheses and design (written before results) are still the design of the experiment; its results were produced by the first pipeline. The numbers below are kept as the diagnostic comparison that motivated the correction (`09_psh_diagnostic_design.md`, `12_frozen_protocol.md`). **Reported results are in `v5_results.md`.**

Written 2026-10-14, **before** any v4 detector was trained or any result
seen, so the predictions below are genuine predictions.

---

## 1. The problem this experiment solves

RQ2 asks which aggregation rule (majority, unanimous, confidence-weighted)
resists the DDQN agent best. So far the experiments cannot answer it:

| Run | Detector state | What the rules did | Why |
|---|---|---|---|
| v2 | All three share the same blind spot (padded scans look benign) | Each rule copied one member exactly: unanimous ≡ XGBoost, majority ≡ weighted ≡ RF, any ≡ CNN (identical 500-episode training curves) | The members fail on the *same* probes |
| v3 | All three hardened over the full payload range | All 7 defences at 0% evasion | No member fails anywhere, so there is nothing for a rule to combine |

Step 1 (`02_voting_rule_metrics.md`) measured this directly. On padded
held-out probes, every pair of original detectors is wrong *together* on
83–97% of probes (double-fault rate). Under attack, the members are
almost perfectly redundant.

## 2. Why diversity is the missing ingredient

Ensemble theory says a combination of classifiers can only beat its
members when the members make **different errors**:

- Hansen & Salamon (1990) showed that majority voting reduces error only
  when member errors are not perfectly correlated.
- Krogh & Vedelsby (1995) decomposed ensemble error into the average member
  error minus the members' disagreement ("ambiguity"): with no
  disagreement, the ensemble is exactly as good as its members.
- Kuncheva & Whitaker (2003) formalised this with diversity measures such
  as disagreement and double-fault, which step 1 now reports.
- In adversarial settings, Tramèr et al. (2018) and Pang et al. (2019)
  argue the same for robustness: ensemble members that share a
  vulnerability are attacked as one model.

v2 and v3 sit at the two extremes where no rule can matter: zero diversity
under attack (v2), and nothing left to fix (v3). To compare the rules we
need **members whose robustness differs across the attack space**, and we
need that difference to be controlled and measurable.

## 3. Method: partitioned adversarial training

v3 hardened each detector by adding payload-padded copies of the
training-split PortScan flows, with payload drawn uniformly from 1–1460 B.
v4 keeps everything identical except **which payload sizes each detector
is hardened on**.

### 3.1 Payload bands

The padding range 1–1460 B is split into three contiguous bands:

| Band | Payload range | Simulator payload options inside it |
|---|---|---|
| A | 1–90 B | 64 |
| B | 91–362 B | 128, 256 |
| C | 363–1460 B | 512 |

Band edges sit at the geometric midpoints between consecutive payload
options of the simulator (√(64·128) ≈ 90, √(256·512) ≈ 362). This
guarantees two things:
1. Each band contains at least one payload the agent can actually send.
   Otherwise a detector's hardening would never be tested by the agent.
2. No payload option lies on a band edge, so every option clearly belongs
   to one band.

As in v3, training payloads are drawn **uniformly within the band**, never
at the simulator's exact values, so no detector is trained on the test
probes. Robustness at random sizes (1–1460 B) is evaluated as well.

### 3.2 Two new detector sets

Let *k* be the number of detectors hardened in a band. The existing runs
are k = 0 (v2) and k = 3 (v3). v4 adds the two missing levels:

| Set | Directory | XGBoost hardened on | RF hardened on | CNN hardened on | Detectors per band |
|---|---|---|---|---|---|
| cov0 (= v2) | `code/models/` | — | — | — | 0 |
| **cov1: disjoint** | `code/models_cov1/` | A | B | C | 1 |
| **cov2: overlapping** | `code/models_cov2/` | A + B | B + C | C + A | 2 |
| cov3 (= v3) | `code/models_adv/` | A + B + C | A + B + C | A + B + C | 3 |

The assignment was fixed in advance by ordering: detectors in pipeline
order (XGBoost, RF, CNN) and bands in increasing size, assigned cyclically.
It was not tuned.

Together the four sets form a **coverage ladder** (k = 0, 1, 2, 3). Every
voting rule can be tested at every coverage level.

### 3.3 What is held constant

| Element | Value (same in every set) |
|---|---|
| Training split, scaler, label encoder | original (random_state=42), copied |
| Model hyperparameters | `train_models.py`, `train_cnn.py` |
| Padded rows added per detector | 71,381 (one per eligible training-split PortScan flow, as in v3) |
| Padded-row arithmetic | `flow_features.set_forward_payload` (validated CICFlowMeter formulas) |
| CNN epoch selection | 5% validation slice of the training data (as in v3) |
| Test split and probe bank | untouched, never used for training |
| Agent, environment, evaluation | identical to v2/v3 (`train_ensemble.py`, `evaluate_agents.py`), seed 0 |
| Weighted-rule weights | recomputed from each set's own held-out F1, same procedure as before |

For a detector hardened on two bands (cov2), the 71,381 rows are split
equally between its bands, then drawn uniformly within each band. This
keeps the narrow band A (90 sizes) from being swamped by band C
(1,098 sizes).

## 4. Predictions

### 4.1 The k-out-of-n argument

Each voting rule is a *t-out-of-3* decision: it flags a probe when at
least *t* members flag it. This is the k-out-of-n structure from
reliability engineering (Barlow & Proschan, 1975).

| Rule | Votes needed (t) |
|---|---|
| any | 1 |
| majority | 2 |
| unanimous | 3 |
| weighted | ≈ 2. With near-equal weights (≈ 0.34 each), one confident detector gives ≈ 0.34 < 0.5 and two give ≈ 0.66 ≥ 0.5 |

If hardening stays inside its band, a rule resists padded probes in a band
**if and only if k ≥ t** for that band.

| Coverage | any (t=1) | majority (t=2) | weighted (≈2) | unanimous (t=3) | each standalone |
|---|---|---|---|---|---|
| k = 0 (v2, observed) | fails | fails | fails | fails | fails |
| **k = 1 (cov1)** | **resists** | fails | fails | fails | resists only in its own band |
| **k = 2 (cov2)** | **resists** | **resists** | **resists** | fails | resists in 2 of 3 bands |
| k = 3 (v3, observed) | resists | resists | resists | resists | resists |

### 4.2 Hypotheses

- **H1 (coverage threshold).** A rule needing *t* votes resists padded
  probes in every band covered by at least *t* hardened members, and fails
  in bands covered by fewer.
- **H2 (RQ1).** In cov1 and cov2, every standalone detector stays evadable
  (outside its band). Some ensemble rules do not (any in cov1; any,
  majority and weighted in cov2). Only in these sets can an ensemble be
  more robust than every one of its members.
- **H3 (cost of robustness).** Lower vote thresholds inherit more false
  alarms. *any* has the union of the members' false positives (step 1:
  2.87–3.58%, driven by the CNN); majority stays near the trees (≈ 0.07%).
  In cov2, majority should therefore match *any*'s robustness at about
  1/40 of its false-positive rate.
- **H4 (exploratory, convergence).** When a defence is evadable only at a
  few payload options, the agent should need more episodes to find them
  than when most options evade.

### 4.3 What would contradict the predictions, and what that would mean

- **Hardening transfers outside a band.** For example, a CNN hardened on
  band C might also catch band B probes by interpolating between the
  unpadded (0–2 B) and padded (≥ 363 B) scans it was trained on. Then the
  *measured* coverage exceeds the designed coverage, and rules do better
  than predicted. This is a finding in itself: it shows which model
  families generalise robustness beyond the data they were hardened on.
  Trees are expected to stay local because their splits are thresholds
  learned from the data they saw.
- **Partial detection by unhardened members.** The original CNN already
  catches some padded probes. This can push a rule above its predicted
  robustness.
- **Weighted ≠ majority.** If unhardened members return moderate attack
  probabilities, the weighted score can reach 0.5 with only one confident
  member.

To separate these effects, the per-detector detection rate is measured at
every payload size (`evaluate_detectors.py`), giving the *measured*
coverage matrix next to the designed one.

## 5. Procedure

1. `train_adversarial.py --design cov1|cov2 --stage trees|cnn`: build the
   per-detector augmented training sets and train (trees and CNN run in
   separate processes because of the libomp clash).
2. `IDS_MODEL_DIR=code/models_covK evaluate_detectors.py`: held-out
   metrics, measured coverage, per-rule metrics, diversity, F1 weights.
3. From `results_v4_covK/`:
   `IDS_MODEL_DIR=… train_ensemble.py` (14 experiments: 7 defences ×
   reward-only / feature-observing, 50,000 steps, seed 0), then `evaluate_agents.py`.
4. `summarize_results.py`: one table and figure across the coverage ladder
   (cov0–cov3) for every defence.

## 6. Metrics reported for every defence × coverage level

| Metric | Source | Answers |
|---|---|---|
| Attack recall, false-positive rate, F1 on held-out test split | `rule_eval.json` | cost of each rule (H3) |
| Padded-probe detection per payload size (no agent) | `rule_eval.json` | predicted robustness (H1) |
| Untrained evasion (random probes, paired) | `eval_results.json` | baseline |
| Trained reward-only / feature-observing evasion (greedy) | `eval_results.json` | RQ1, RQ2 (H1, H2) |
| Episodes until training evasion first exceeds 50% / 90% (15-episode rolling mean) | convergence JSON | convergence dynamics (H4) |
| Probes detected during training | convergence JSON | how much the attacker paid before succeeding |

## 7. Threats to validity

- **Assignment choice.** A different band-to-detector assignment could
  change results where model families generalise differently. The measured
  coverage matrix makes this visible. Rotating the assignment is a possible
  follow-up.
- **Band edges** are placed relative to the simulator's payload options.
  This is needed for the agent to test every band, but robustness is also
  reported at random sizes, so the conclusions don't rest only on the five
  options.
- **One seed** (multi-seed replication is the next step).
- **Effective action space.** Only payload size changes the flow record,
  so the agent's task is close to choosing among five payload options.
  Convergence differences (H4) will be modest.
- **Single attack dimension.** Diversity here exists along payload only.
  Real deployments differ along many dimensions at once.

## 8. References

- Barlow, R. E., & Proschan, F. (1975). *Statistical Theory of Reliability
  and Life Testing: Probability Models*. Holt, Rinehart and Winston.
- Dietterich, T. G. (2000). Ensemble methods in machine learning.
  *Multiple Classifier Systems (MCS 2000)*, LNCS 1857, 1–15.
- Hansen, L. K., & Salamon, P. (1990). Neural network ensembles.
  *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(10),
  993–1001.
- Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross
  validation, and active learning. *Advances in Neural Information
  Processing Systems 7*, 231–238.
- Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in
  classifier ensembles and their relationship with the ensemble accuracy.
  *Machine Learning*, 51(2), 181–207.
- Lam, L., & Suen, C. Y. (1997). Application of majority voting to pattern
  recognition: an analysis of its behavior and performance. *IEEE
  Transactions on Systems, Man, and Cybernetics — Part A*, 27(5), 553–568.
- Pang, T., Xu, K., Du, C., Chen, N., & Zhu, J. (2019). Improving
  adversarial robustness via promoting ensemble diversity. *ICML 2019*.
- Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D., &
  McDaniel, P. (2018). Ensemble adversarial training: attacks and
  defenses. *ICLR 2018*.

Related, for the discussion of the shared blind spot: Geirhos et al.
(2020), Shortcut learning in deep neural networks, *Nature Machine
Intelligence* 2, 665–673. Arp et al. (2022), Dos and Don'ts of Machine
Learning in Computer Security, *USENIX Security 2022*. Engelen, Rimmer &
Joosen (2021), Troubleshooting an intrusion detection dataset: the
CICIDS2017 case study, *IEEE Security and Privacy Workshops*.
Check page numbers against your reference manager before citing.
