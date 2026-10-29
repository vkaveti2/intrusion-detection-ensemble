# 02 — Decision-Rule Metrics Without an Agent (Step 1)

> **Superseded by the corrected protocol (v5).** This document records the first pipeline: detectors trained with `PSH Flag Count` visible, a train/test split that is not group-disjoint from the probe bank, and (for the original CNN) epoch selection on the test split. Its detection rates and rule metrics came from detectors that relied on `PSH Flag Count`. The numbers below are kept as the diagnostic comparison that motivated the correction (`09_psh_diagnostic_design.md`, `12_frozen_protocol.md`). **Reported results are in `v5_results.md`.**

Run 2026-10-14 with `evaluate_detectors.py` on the original detectors
(`code/models/`, used in v2) and the adversarially trained detectors
(`code/models_adv/`, used in v3). Raw numbers: `rule_eval.json` and
`detector_eval.json` in each directory.

## What was measured

Every rule is scored on **the same detector outputs** (a paired comparison),
using exactly the decision logic of `ensemble.py`. The vectorised rules were
checked against `EnsembleIDS.decide` on 20,000 random inputs: 0 mismatches
for all 7 rules.

1. **Held-out test split** (504,160 rows: 419,012 benign, 85,148 attacks):
   attack recall, false-positive rate, precision, F1, and per-class
   detection rate.
2. **Padded held-out scan probes** (17,834 probe-bank flows per payload
   size). Two port settings:
   - *simulator ports*: destination port 1001–1100, exactly what the DDQN
     agent sends (the environment scans ports 1000 + step). Use these to
     interpret the RL results.
   - *recorded ports*: each flow keeps its original destination port. This
     is a broader robustness check.

   Destination port is one of the 77 features, and the detectors use it,
   so the two settings give different numbers. For the CNN at 64 B: 21.4%
   caught at recorded ports, 3.1% at simulator ports.
3. **Pairwise diversity** between detectors (Kuncheva & Whitaker, 2003):
   - *disagreement*: share of rows where exactly one of the pair is right
   - *double fault*: share of rows where both are wrong

### Two corrections to earlier numbers
- The old per-detector evaluation flagged an attack when P(BENIGN) < 0.5.
  `ensemble.py` flags when P(attack) ≥ 0.5, i.e. P(BENIGN) ≤ 0.5. Random
  Forest probabilities are vote fractions of 100 trees, so exact 50/50
  ties occur. The evaluation now uses the ensemble's rule.
- Multi-threaded RF prediction sums tree votes in a varying order, so tied
  rows fell either side of 0.5 from run to run. RF is now evaluated
  single-threaded, which is deterministic and matches the ensemble's
  inference.

Effect: RF attack recall 0.99510 → 0.99515, false-positive rate 0.0718% →
0.0733%; padded-probe numbers move by ≤ 0.1 percentage point. XGBoost and
CNN are unchanged. No v2/v3 RL result is affected, since the ensemble
always used the correct rule.

---

## 1. Clean held-out traffic: the cost of each rule

### Original detectors (v2)

| Rule | Attack recall | False-positive rate | Precision | F1 | False alarms | Missed attacks |
|---|---|---|---|---|---|---|
| any (1 of 3) | **99.99%** | 2.873% | 87.61% | 0.9339 | 12,038 | **6** |
| weighted | 99.83% | 0.085% | 99.58% | 0.9971 | 356 | 146 |
| majority (2 of 3) | 99.77% | 0.072% | 99.65% | 0.9971 | 301 | 192 |
| unanimous (3 of 3) | 94.08% | **0.049%** | 99.74% | 0.9683 | **205** | 5,043 |
| XGBoost | 99.87% | 0.069% | 99.66% | **0.9977** | 289 | 110 |
| RF | 99.51% | 0.073% | 99.64% | 0.9958 | 307 | 413 |
| CNN | 94.46% | 2.851% | 87.07% | 0.9061 | 11,948 | 4,718 |

### Adversarially trained detectors (v3)

| Rule | Attack recall | False-positive rate | Precision | F1 | False alarms | Missed attacks |
|---|---|---|---|---|---|---|
| any | 99.99% | 3.584% | 85.01% | 0.9189 | 15,017 | 8 |
| weighted | 99.83% | 0.085% | 99.58% | 0.9970 | 356 | 148 |
| majority | 99.77% | 0.070% | 99.66% | 0.9971 | 293 | 194 |
| unanimous | 95.05% | 0.051% | 99.74% | 0.9734 | 214 | 4,216 |
| XGBoost | 99.86% | 0.067% | 99.67% | 0.9977 | 281 | 117 |
| RF | 99.50% | 0.071% | 99.65% | 0.9957 | 299 | 428 |
| CNN | 95.45% | 3.566% | 84.47% | 0.8962 | 14,944 | 3,873 |

### Observations
1. **No rule beats the best single model on F1.** XGBoost has the highest
   F1 in both sets (0.9977). Majority and weighted come next (0.9971). On
   clean traffic, XGBoost and RF agree on all but 0.09% of rows, so there
   is little left for voting to correct.
2. **The rules trade recall for false alarms, as expected.** *any* catches
   all but 6 of 85,148 attacks but raises 12,038 false alarms (40× majority).
   *unanimous* has the fewest false alarms but misses 5,043 attacks.
3. **The weakest member sets the cost of the extreme rules.** 99% of
   *any*'s false alarms are the CNN's (11,948 of 12,038), and 4,718 of
   *unanimous*'s 5,043 misses are attacks the CNN missed. Majority and
   weighted let the two trees outvote the CNN, so they stay near the trees'
   numbers.
4. **Adversarial training raised the CNN's false alarms** (11,948 →
   14,944, +25%), and *any* inherits all of that (2.87% → 3.58%). The
   trees were unaffected.
5. **Weighted ≠ majority on some classes.** On *Bot*, weighted catches
   91.0% vs majority's 81.3%: the CNN's confident vote (93.3% alone) plus
   the trees' partial probabilities reaches the 0.5 threshold even when the
   trees vote "benign". This is the largest per-class difference between
   weighted and majority in the original set.

Selected per-class detection rates, original detectors (full table in
`rule_eval.json`; Heartbleed has 2 test rows and SQL injection 4, so
treat those as anecdotal):

| Class | majority | unanimous | weighted | any | XGBoost | RF | CNN |
|---|---|---|---|---|---|---|---|
| Bot | 81.3% | 65.1% | 91.0% | 99.7% | 75.6% | 77.2% | 93.3% |
| DoS Hulk | 99.8% | 90.4% | 99.7% | 100.0% | 100.0% | 99.8% | 90.4% |
| PortScan | 99.9% | 94.1% | 100.0% | 100.0% | 100.0% | 98.9% | 95.1% |
| Web Attack – XSS | 96.9% | 95.4% | 96.9% | 100.0% | 100.0% | 95.4% | 96.9% |

---

## 2. Padded scan probes: robustness of each rule (no agent)

Share of padded held-out probes flagged, simulator ports (what the agent
faces). Original detectors:

| Rule | 0 B | 64 B | 128 B | 256 B | 512 B | random 1–1460 B |
|---|---|---|---|---|---|---|
| any | 100.0% | 3.1% | 41.2% | 0.2% | 0.0% | 2.1% |
| weighted | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% |
| majority | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% |
| unanimous | 84.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% |
| XGBoost | 84.9% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% |
| RF | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.2% |
| CNN | 100.0% | 3.1% | 41.2% | 0.2% | 0.0% | 2.1% |

At recorded ports the pattern is the same, with slightly more detection
(CNN 21.4% / 33.4% / 14.0% / 0.7% at 64–512 B; *any* up to 33.9%).
Adversarially trained detectors: **100.0% for every rule at every size**,
under both port settings.

These tables are the agent-independent version of the v2/v3 RL results.
The agent's best achievable evasion against a defence is set by its best
payload option. For every defence in the original set, at least one option
evades almost all probes: 512 B evades all seven defences.

---

## 3. Why the rules collapse: detector diversity

| Pair | Clean test split: disagreement / double fault | Padded probes, simulator ports: disagreement / double fault |
|---|---|---|
| XGBoost–RF | 0.09% / 0.06% | 0.01% / **99.96%** |
| XGBoost–CNN | 3.29% / 0.05% | 9.29% / **90.67%** |
| RF–CNN | 3.31% / 0.07% | 9.30% / **90.67%** |

On padded probes (original detectors), every disagreement goes the same
way. The share "only the tree is right" is 0.00% for both tree–CNN pairs,
and 0.01% for XGBoost over RF. The detectors' correct answers are
**nested**:

> XGBoost's detections ⊆ RF's detections ⊆ CNN's detections.

With nested members, a *t*-of-3 rule always gives the same answer as one
specific member: any = CNN, majority = RF, unanimous = XGBoost. Weighted
follows majority because no single weight reaches 0.5. This is exactly the
collapse seen in the v2 training curves (500/500 identical episodes for
each pair).

In the adversarially trained set, the double-fault rate on padded probes
is 0.00%. No detector fails, so there is nothing to vote on.

## 4. What this means for RQ2

- On clean traffic, the voting rule mainly decides **whose errors you
  inherit**: *any* takes on the CNN's false alarms, *unanimous* its misses,
  and majority/weighted stay close to the trees.
- Under attack, the original members fail together (double fault 91–100%),
  so **no rule adds robustness**. After full hardening, none is needed.
- A rule can only add robustness when members fail on *different* probes.
  The v4 diversity experiment creates exactly that condition
  (`03_diversity_experiment_design.md`). These cost numbers (false alarms
  per rule) are the other half of the trade-off it will measure.
