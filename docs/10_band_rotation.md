# 10 — CNN Band Rotation (bounded diagnostic)

Written before any rotated model was trained. Frozen protocol v5 still applies (`12_frozen_protocol.md`); this analysis was listed there and edits none of the frozen files. Code: `code/train_v5_rotation.py`.

## Question

In cov1 the CNN is hardened on band C (363-1460 B) and leaves a gap at small payloads. Does the gap change with the band the CNN was assigned? Dr. Saha asked for this as a bounded diagnostic, not a full explanation of CNN and tree behaviour.

## Design

| | XGBoost | RF | CNN |
|---|---|---|---|
| cov1 baseline | A (1-90 B) | B (91-362 B) | **C** (363-1460 B) |
| rotation 1 | A | B | **A** |
| rotation 2 | A | B | **B** |

Only the CNN changes. XGBoost and RF are the matching cov1 models (symlinks), so within a model seed the CNN's band is the only difference. Each rotation is trained with model seeds 1 and 2 and compared with the baselines `cov1_ms1` and `cov1_ms2`, which already exist. That is 4 new CNNs. Training data, scaler, split, hyperparameters, the inner-validation epoch selection and the padded-copy sources are those of the frozen pipeline. The CNN's payloads are drawn uniformly within its band with the same generator as the baseline.

Two seeds, because the baseline gap already varies by training run: the CNN caught 0 of 13,569 probes at 64 B in three of four cov1 runs but 197 to 11,174 at 128 B. One seed per band could not separate a band effect from a seed effect.

## Measurements (final-test bank, exact counts)

1. Detection of padded probes at 64, 128, 256 and 512 B and random sizes, for the CNN alone and for each rule.
2. Where the gap sits (which sizes fall below full detection) and how wide, per band and seed.
3. Whether the unanimous rule still inherits the CNN's gap and whether majority, weighted and `any` still cover it.
4. Clean-traffic false positives of each rotated CNN.

An agent experiment (CNN and unanimous, both agent types, agent seed 0) runs only for a rotated set that shows a gap.

## Pre-stated reading

- **The gap follows the band** if the CNN's weak sizes move toward the sizes outside its new band in both seeds.
- **The gap does not follow the band** if the weak sizes stay near 64 B regardless of band, which would point to the CNN itself, not the band assignment.
- **Inconclusive** if the two seeds disagree.

Anything beyond this (why the trees transfer and the CNN does not) stays in limitations and future work, unless the result contradicts the main findings.

---

# Results (detection rates; agent runs follow)

Tables: `band_rotation_summary.md` / `.csv`, generated from the final-test evaluation of the four rotated sets and the two cov1 baselines. Final-test probe bank, 13,569 flows per cell.

## CNN alone, flagged of 13,569

| Model seed | CNN band | 64 B | 128 B | 256 B | 512 B | random 1-1460 B |
|---|---|---|---|---|---|---|
| 1 | C (baseline) | 0 | 11,174 | 13,563 | 13,569 | 12,580 |
| 1 | A | 13,565 | 11,793 | 0 | 0 | 1,626 |
| 1 | B | 12,696 | 13,569 | 13,569 | 13,563 | 11,406 |
| 2 | C (baseline) | 0 | 6,041 | 13,563 | 13,569 | 12,417 |
| 2 | A | 13,569 | 13,566 | 0 | 0 | 2,769 |
| 2 | B | 13,532 | 13,569 | 13,569 | 13,564 | 13,069 |

## Reading, against the pre-stated criteria

1. **The gap follows the band, in both seeds.** A CNN hardened on C flags 0 of 13,569 probes at 64 B (both seeds). Hardened on A, it flags 64 B almost completely (13,565 and 13,569) but flags 0 of 13,569 at 256 B and at 512 B (both seeds). Hardened on B, the middle band, its gaps are small: 64 B misses 873 (seed 1) and 37 (seed 2), and the other agent sizes stay at or above 13,563; at random sizes it flags 11,406 and 13,069. The blind sizes are the ones farthest from the assigned band. The partial zone beside a blind size varies by seed: the band-C CNN flags 11,174 (seed 1) and 6,041 (seed 2) at 128 B.
2. **Unanimous inherits the CNN's blind sizes exactly.** It flags 0 of 13,569 wherever the CNN does (C at 64 B, A at 256 B and 512 B).
3. **Majority depends on which two members cover each size.** With the CNN on C, majority is limited at 64 B by RF's gap there (13,029 and 12,810 flagged). With the CNN on A, the CNN covers 64 B and majority, weighted and `any` flag all 13,569 at every agent size in both seeds. A rule that needs two of three members is only as robust as the weakest size where fewer than two members are strong.
4. **Cost.** The rotated CNNs have somewhat higher clean false-positive rates (4.2% to 4.8%) than the baseline CNNs (3.9% in seed 1, 3.7% in seed 2). The baseline itself moves 0.27 points between the two seeds, so with two seeds this is suggestive, not established.
5. **Main findings.** Not contradicted. XGBoost is unchanged (it is the same model in every set) and no rule beats it.

Limits: two model seeds per band and one dataset; the mechanism (why the CNN generalises across payload size less than the trees do) is not studied here and stays in future work.
