#!/usr/bin/env bash
# run_v5_sweep.sh — the agent sweep on the frozen v5 protocol (docs/12_frozen_protocol.md).
#
#   caffeinate -i ./code/run_v5_sweep.sh            # start, or resume after any interruption
#   SMOKE=1 ./code/run_v5_sweep.sh                  # one batch only, for a first check
#
# Part A (agent seeds): cov0-cov3 x agent seeds 0-4, 14 experiments per batch.
# Part B (detector seeds): cov1_ms1-3 x agent seed 0.   Separate experiments, not crossed.
# Agents train on the agent-train bank and are evaluated on the final-test bank.
#
# Resumable: a batch with a `.done` marker is skipped. An unfinished batch is rerun with
# --fresh-incomplete, which keeps agents that already finished and restarts the others from
# scratch (a checkpoint resume would restart with an empty replay buffer and differ from an
# uninterrupted run). Output: results_v5/agents/<set>/seed_<n>/.  Order puts the headline sets
# (cov1, cov3) first so the most informative results exist earliest.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/ids_env/bin/python"
OUT_ROOT="$ROOT/results_v5/agents"
export ROOT PY OUT_ROOT
export PROJECT_LOG_PATH="$ROOT/docs/logs/PROJECT_LOG_v5_sweep_auto.txt"
mkdir -p "$OUT_ROOT" "$ROOT/docs/logs"

batch() {
    local set="$1" seed="$2"
    local dir="$OUT_ROOT/$set/seed_$seed"
    mkdir -p "$dir"
    if [ -f "$dir/.done" ]; then echo "[skip] $(date +%H:%M:%S) $set seed $seed (done)"; return 0; fi
    echo "[run ] $(date +%H:%M:%S) $set seed $seed"
    (
        cd "$dir" || exit 1
        export IDS_MODEL_DIR="$ROOT/code/models_v5/$set"
        "$PY" "$ROOT/code/train_ensemble.py" --seed "$seed" --max-parallel 7 --fresh-incomplete \
            >> train_run.log 2>&1 || exit 1
        # every one of the 14 agents must exist before the batch is evaluated
        n=$(ls models/ddqn_*.zip 2>/dev/null | wc -l | tr -d ' ')
        [ "$n" -eq 14 ] || { echo "only $n of 14 agents present" >> train_run.log; exit 1; }
        "$PY" "$ROOT/code/evaluate_agents.py" >> evaluate_agents.log 2>&1 || exit 1
        [ -f eval_results.json ] || exit 1
        touch .done
    ) && echo "[done] $(date +%H:%M:%S) $set seed $seed" || echo "[FAIL] $(date +%H:%M:%S) $set seed $seed (rerun this script to resume)"
}
export -f batch

if [ "${SMOKE:-0}" = "1" ]; then
    batch cov1 0
    exit 0
fi

echo "v5 sweep started $(date +%H:%M:%S)"
{
    for s in 0 1 2 3 4; do echo "cov1 $s"; echo "cov3 $s"; done
    for s in 1 2 3; do echo "cov1_ms$s 0"; done
    for s in 0 1 2 3 4; do echo "cov0 $s"; echo "cov2 $s"; done
} | xargs -P 2 -L 1 bash -c 'batch "$0" "$1"'
echo "v5 sweep finished $(date +%H:%M:%S)"
total=0; done_n=0
for d in "$OUT_ROOT"/*/seed_*; do total=$((total + 1)); [ -f "$d/.done" ] && done_n=$((done_n + 1)); done
echo "batches complete: $done_n of $total"
