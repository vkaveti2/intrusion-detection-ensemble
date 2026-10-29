# 15 — Results (chapter draft)

Draft prose for the results chapter, on the frozen protocol (`12_frozen_protocol.md`). Full tables with exact counts and intervals are in `v5_results.md`; every figure quoted here is taken from that file, `fixed_payload_summary.md`, `band_rotation_summary.md` or `13_benchmark_results.md`. Terms follow `00_terminology.md`. Detection rates (no agent) and evasion rates (trained agent) are reported separately throughout.

## 4.1 Central finding

**The tested ensembles did not outperform their strongest observed standalone member.** In every hardened detector set, XGBoost alone is never evaded by a trained agent and flags every padded probe at every agent payload size; no ensemble rule did better, and the rules that matched it either cost far more false alarms (`any`) or were evaded under partial hardening (majority, weighted, unanimous). Among the ensemble rules, majority and weighted voting performed best on clean traffic and under full hardening. XGBoost matched their robustness with the same false-alarm rate and a smaller inference cost. The result differs from the hypothesis that motivated the work, and it is stated here as the finding, with the conditions under which it was tested: one dataset, a constrained flow-feature simulation, and hardening against a single attack family (padded scan probes).

This does not mean ensembles are never useful. The band-rotation diagnostic (§4.8) shows an ensemble covering a weak member's blind sizes when two other members are strong there. It shows that the benefit depends on which members are strong where, and that the strongest member alone was sufficient in every set tested.

## 4.2 Cost on clean traffic

Final-test partition, 378,337 rows (full hardening, cov3; 95% intervals resample whole signatures):

| Rule | False positives (rate, interval) | Missed attacks (rate, interval) |
|---|---|---|
| XGBoost | 241 (0.077%, [0.041%, 0.127%]) | 116 (0.181%, [0.146%, 0.218%]) |
| RF | 254 (0.081%, [0.045%, 0.130%]) | 233 (0.364%, [0.296%, 0.451%]) |
| CNN | 11,585 (3.686%, [3.560%, 3.808%]) | 3,954 (6.173%, [5.141%, 7.512%]) |
| Majority | 244 (0.078%, [0.041%, 0.127%]) | 173 (0.270%, [0.227%, 0.317%]) |
| Weighted | 290 (0.092%, [0.056%, 0.141%]) | 137 (0.214%, [0.176%, 0.255%]) |
| Unanimous | 171 (0.054%, [0.022%, 0.102%]) | 4,108 (6.414%, [5.372%, 7.736%]) |
| Any | 11,665 (3.712%, [3.584%, 3.834%]) | 22 (0.034%, [0.021%, 0.050%]) |

False-alarm rates of XGBoost, RF, majority, weighted and unanimous are indistinguishable at this resolution (intervals overlap). The CNN and `any` raise about 48 times as many false alarms as XGBoost (11,665 for `any` against 241). `Any` does miss fewer attacks than any other rule (22 against XGBoost's 116), so the trade-off is real: a defender who values recall over false alarms has a reason to use it. Unanimous misses the most attacks (4,108) because it requires the CNN to agree. The unhardened set (cov0) shows the same ordering of false-alarm cost (XGBoost 242, CNN 14,332, `any` 14,397 false positives).

## 4.3 Padded-probe detection, no agent

Padded final-test probes (13,569 flows from 72 groups per cell). The key results:

- **Full hardening (cov3).** Every rule flags 13,569 of 13,569 at 64, 128, 256 and 512 B. At random sizes the CNN and unanimous flag 13,564 of 13,569, and every other rule flags all.
- **One band per detector (cov1, model seed 42).** At 64 B: XGBoost and `any` 13,569 of 13,569 (interval [1.000, 1.000]); RF 7,258 [0.338, 0.711]; CNN 7,246 [0.337, 0.711]; majority 7,258; weighted 8,556 [0.441, 0.802]; unanimous 7,246. At 128 B the CNN and unanimous drop to 7,333 and the others stay complete; at 256 B they reach 11,254. The wide intervals reflect that these rates are driven by a handful of groups.
- **Two bands per detector (cov2).** RF flags 13,016 of 13,569 at 64 B and the CNN 13,564 at 128 and 256 B; majority, weighted and `any` flag everything.
- **Unhardened (cov0).** At 64 B, XGBoost, RF, majority, weighted and unanimous each flag 0 of 13,569 padded probes, reproducing the original shortcut that padded scan flows look benign. Only the CNN, and `any` which inherits it, flag any (7,836 of 13,569).

Across the four cov1 training runs (Part B), XGBoost is complete in all four; the RF's 64 B detection ranges from 7,258 to 13,460 of 13,569 (6,311 to 109 missed) and the CNN's is 7,246 in model seed 42 and 0 in the other three, so *the size of a weak member's gap depends on the training run*. The interval for any one run describes group sampling only; the spread between runs is a separate and larger source of variation.

## 4.4 Trained agents, Part A (agent seeds)

Evaded probes out of 2,000 per run; five agent seeds; reward-only and feature-observing agents (ranges over seeds shown as min-max; per-seed values in `v5_results.md`):

| Set | XGBoost | RF | CNN | Majority | Weighted | Unanimous | Any |
|---|---|---|---|---|---|---|---|
| cov0 | 2,000 (all seeds, both agents) | 2,000 | 847-2,000 / 910-2,000 | 2,000 | 2,000 | 2,000 / 1,981-2,000 | 847-2,000 / 910-2,000 |
| cov1 | 0 | 922 (all seeds, both agents) | 916-924 / 715-920 | 922 | 729 / 366-729 | 916-924 / 592-920 | 0 |
| cov2 | 0 | 0-74 / 0-1 | 0-1 / 0 | 0 | 0 | 0-74 / 0-28 | 0 |
| cov3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

(Each cell with two values is reward-only / feature-observing.)

- **Unhardened detectors (cov0)** are evaded completely by both agent types for the tree-based rules, in every seed. The CNN and `any` vary by seed because some agents settle on a payload size the CNN partly catches (a local optimum).
- **Full hardening (cov3)** is not evaded by any agent in any of the 70 runs.
- **Partial hardening (cov1)**: XGBoost and `any` are never evaded. Majority is evaded 922 of 2,000 times in every seed, equal to the RF, because the RF and the CNN both leave a gap at 64 B and majority needs two of three. Weighted voting is evaded somewhat less (729 for the reward-only agent). Unanimous is evaded as often as the CNN, whose blind sizes it inherits.
- **Two bands per detector (cov2)** leaves only small gaps (at most 74 of 2,000), exploited by some seeds.
- **Feature observation did not help the agent.** In all 15 cells where any run evaded, the feature-observing agent's total evasion over the five seeds was at most the reward-only agent's, and its results varied more between seeds.

## 4.5 Trained agents, Part B (detector training seeds)

Four training runs of the cov1 set (model seeds 42, 1, 2, 3), attacked with agent seed 0; evaded of 2,000, reward-only / feature-observing:

| Rule | Model seed 42 | 1 | 2 | 3 |
|---|---|---|---|---|
| XGBoost | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| RF | 922 / 922 | 0 / 68 | 0 / 0 | 0 / 0 |
| CNN | 916 / 920 | 2,000 / 45 | 2,000 / 2,000 | 1,975 / 1,728 |
| Majority | 922 / 922 | 0 / 68 | 0 / 0 | 0 / 0 |
| Weighted | 729 / 508 | 3 / 3 | 0 / 0 | 0 / 0 |
| Unanimous | 916 / 920 | 2,000 / 282 | 2,000 / 1,123 | 1,725 / 1,973 |
| Any | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

The CNN is evaded almost completely in three of four runs and unanimous with it. Majority and weighted are robust in model seeds 2 and 3 and nearly so in seed 1, because the RF's gap at 64 B there is small (540, 759 and 109 probes missed of 13,569). In model seed 42 the RF's gap is large and majority is evaded 922 times. *Whether majority voting is robust under partial hardening therefore depends on the training run*, which is the point of Part B: majority is only as robust as the second-strongest member at the sizes where the strongest is not enough. XGBoost is never evaded in any of the four runs. Part A and Part B are separate experiments; no combination of several agent seeds with several detector seeds was run.

## 4.6 Fixed-payload comparison

Because only the payload size changes the detectors' input, a policy that always sends one payload size at the smallest delay is a natural comparator. The size is chosen on the dev bank and evaluated on the final-test bank; all five sizes are reported as a reference (`fixed_payload_summary.md`).

- In all 28 set-and-rule pairs the size chosen on dev equalled the best of the five on final test. The dev bank has 14 groups, so this is encouraging but weak evidence of a stable choice.
- In the 15 cells where anything evades, the trained agent's mean evasion never exceeds the best fixed choice by more than one point. It is below it in most: for example, against the cov0 CNN a fixed 256 B payload evades all 13,569 final-test probes while the agents average 79% (a local optimum), and against cov2's RF a fixed 64 B payload evades 553 of 13,569 (4.1%) while the agents average 0.7%.
- The fixed policy earns a higher reward wherever evasion is possible, largely because it uses the smallest delay while the agents choose delays from 0.6 to 37 s (the DDQN reward is an estimate from its evasion rate and median delay).

So, in this simulation, the attacker's problem reduces to choosing among five payload sizes, and learning adds nothing over the best fixed choice. The agent's value is as an attacker model that finds the weak size without being told which it is.

## 4.7 The PSH diagnostic

Before the corrected protocol, hardened detectors appeared robust at every payload size. Setting `PSH Flag Count` to 0 on padded probes collapsed detection in every hardened set, and forcing it to 1 on benign traffic raised false alarms (for example cov1 XGBoost, 0.07% to 2.4%); `Fwd PSH Flags` had no effect (`09_psh_diagnostic_design.md`). Retraining without the column kept full hardening intact and removed the apparent transfer of partial hardening for RF and the CNN. The PSH-visible results are kept only as this diagnostic comparison; whether adversarial training introduced the dependence is suggestive, not shown.

## 4.8 CNN band rotation

Training the CNN on band A or B instead of C (two model seeds each; XGBoost and RF fixed): **the CNN's blind sizes follow its assigned band in both seeds.** Hardened on C it flags 0 of 13,569 probes at 64 B; hardened on A it flags 13,565 and 13,569 at 64 B but 0 at 256 B and at 512 B; hardened on B its gaps are small (64 B: 873 missed in seed 1, 37 in seed 2). The reward-only agent finds them: against the band-A CNN it evades 2,000 of 2,000 probes in both seeds by choosing 256 B or 512 B, against band C it evades 2,000 of 2,000 at 64 B, and against band B it evades 0 and 5. Unanimous inherits the CNN's weakness every time. With the CNN on band A, majority, weighted and `any` flag all 13,569 at every agent size in both seeds: an ensemble covers a weak member's blind sizes when two other members are strong there. The mechanism (why the CNN transfers hardening less than the trees) is not investigated and is left to future work.

## 4.9 Inference cost

On the development machine (Apple M5 Max, single-threaded, batch size 1, inference only; `13_benchmark_results.md`), relative to standalone XGBoost (p50 0.387 ms, peak memory about 163 MB): the three-model ensemble runs at about 7.2 times the p50 latency (7.1 times at p99) and uses about 3.8 times the memory, regardless of voting rule. The random forest accounts for most of it: alone it is about 5 times XGBoost's latency and 2.6 times its memory, while the CNN is about 1.1 times. The four voting rules differ by microseconds. Unhardened and fully hardened sets cost the same to within 2%. Scaling a flow takes about 0.32 ms, which is most of the time for XGBoost and the CNN alone. These measure inference only; packet capture and flow-feature extraction are not included. They come from one machine and do not establish edge-device cost: absolute latencies do not transfer, and slowdown ratios transfer better but remain measurements on that hardware.

## 4.10 What the results support

1. Against detectors hardened on padded probes, the strongest observed standalone comparator (XGBoost) was as hard to evade as any ensemble rule, at lower cost, in every set tested.
2. Majority and weighted voting were the best-behaved ensemble rules on clean traffic and under full hardening, but under partial hardening their robustness depended on which members covered which sizes and varied with the detector's training run.
3. `Any` matched XGBoost's robustness with the highest recall but about 48 times the false alarms; unanimous was the least robust, inheriting its weakest member's blind sizes.
4. A trained agent added nothing over the best fixed payload in this simulation, and observing flow features did not help it.
5. Conclusions about edge-device cost wait on measurements from an edge device.

Limits are in `14_methodology.md` §3.11: one dataset, a constrained simulation, grouping as a proxy for host separation, few groups, one training seed for most sets, and an unexplained transfer mechanism.
