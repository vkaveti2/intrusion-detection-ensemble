# Thesis documentation

*Intrusion Detection: Evaluating Adversarial Resilience and Convergence
Dynamics of Multi-Model Ensemble Systems on Edge Hardware*

Read the numbered documents in order. Together they form the methodology and
results narrative. `PROJECT_LOG.txt` is the dated record of every run,
problem and fix behind them.

| File | What it contains | Use it for |
|---|---|---|
| `01_current_state.md` | Pipeline, dataset, detectors, the 7 defences, simulator, agent, evaluation protocol, runs v1–v3 with results, findings, limitations, file map | Methodology chapter; background for results |
| `02_voting_rule_metrics.md` | Step 1: every decision rule scored on held-out traffic (recall, false alarms, F1, per class), robustness to padded probes without an agent, detector diversity | RQ2 cost side; explains the ensemble collapse |
| `03_diversity_experiment_design.md` | v4 reasoning, hypotheses and design (written before results): partitioned adversarial training, coverage ladder cov0–cov3 | Methodology for the main RQ1/RQ2 experiment |
| `04_diversity_results.md` | v4 results, checked against the hypotheses in 03 | Results and discussion chapters |
| `05_multiseed_design.md` | Step 3 rationale and design (written before results): agent seeds, detector seeds, pre-specified comparisons | Methodology: replication and statistics |
| `06_multiseed_results.md` | Step 3 results and verdicts on the predictions in 05 | Results: robustness of the findings |
| `results_summary.md` / `.csv` | Every defence × coverage level in one table, seed 0 (generated) | Thesis tables |
| `multiseed_summary.md`, `multiseed_runs.csv`, `multiseed_model_seeds.csv` | Step 3 tables over 5 agent seeds and 4 cov1 training runs (generated) | Thesis tables with spread |
| `figures/` | Generated figures: `coverage_ladder_evasion.png`, `coverage_matrix.png`, `multiseed_convergence_cov0.png`, `multiseed_convergence_cov1.png`, `multiseed_learning_speed.png`, `detector_seed_coverage.png` | Thesis figures |
| `PROJECT_LOG.txt` | Dated log. `[auto]` entries are written by the pipeline scripts themselves | Audit trail, appendix |
| `logs/PROJECT_LOG_step3_auto.txt` | The ~500 `[auto]` entries from step 3, kept out of the main log | Audit trail |
| `BUILD_NOTES_2026-09-24.md` | Notes from the first mock-ensemble build (historical) | Background only |

The proposal itself is `../New Thesis Proposal.docx`.

## Regenerating the numbers

From `code/`, with the project virtualenv (`../ids_env/bin/python`). Each
detector set is selected with `IDS_MODEL_DIR` (default `code/models`):

```bash
# detector + decision-rule metrics, diversity  -> <set>/detector_eval.json, rule_eval.json
IDS_MODEL_DIR=$PWD/models_cov1 python evaluate_detectors.py

# detector sets for the diversity experiment (trees and CNN run as separate processes)
python train_adversarial.py --design cov1 --stage trees
python train_adversarial.py --design cov1 --stage cnn

# DDQN experiments + evaluation, run from the results directory
cd ../results_v4_cov1
IDS_MODEL_DIR=$PWD/../code/models_cov1 ../ids_env/bin/python ../code/train_ensemble.py
IDS_MODEL_DIR=$PWD/../code/models_cov1 ../ids_env/bin/python ../code/evaluate_agents.py

# cross-set tables and figures (repository root)
cd .. && ids_env/bin/python summarize_results.py

# step 3: agent seeds 1–4 for every set + cov1 model seeds 1–3, then aggregate
./code/run_step3.sh
ids_env/bin/python summarize_multiseed.py
```

Step 3 results live in `results_multiseed/<set>/seed_<n>/` (agent seeds
1–4) and `results_multiseed/cov1_ms<s>/` (cov1 model seeds 1–3). Seed 0
is the original runs in the directories below.

| Coverage level | Detector set | Results directory |
|---|---|---|
| cov0 (v2, original) | `code/models/` | repository root |
| cov1 (v4, disjoint bands) | `code/models_cov1/` | `results_v4_cov1/` |
| cov2 (v4, overlapping bands) | `code/models_cov2/` | `results_v4_cov2/` |
| cov3 (v3, full hardening) | `code/models_adv/` | `results_v3_adversarial/` |
