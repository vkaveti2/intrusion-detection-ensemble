#!/usr/bin/env bash
# run_v5_rotation_agents.sh — agent runs for the rotated CNN sets (docs/10_band_rotation.md).
# Pre-stated: CNN and unanimous only, both agent types, agent seed 0, for rotated sets that show a gap.
# Output: results_v5/rotation/agents/<tag>/.   Resumable: a set with .done is skipped.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/ids_env/bin/python"
export PROJECT_LOG_PATH="$ROOT/docs/logs/PROJECT_LOG_v5_rotation_auto.txt"
mkdir -p "$ROOT/docs/logs"
run() {
    local tag="$1" dir="$ROOT/results_v5/rotation/agents/$1"
    mkdir -p "$dir"
    if [ -f "$dir/.done" ]; then echo "[skip] $tag"; return; fi
    echo "[run ] $(date +%H:%M:%S) $tag"
    (
        cd "$dir" || exit 1
        export IDS_MODEL_DIR="$ROOT/code/models_v5/$tag"
        "$PY" "$ROOT/code/train_ensemble.py" --seed 0 --voting cnn unanimous --knowledge black white \
            --max-parallel 4 --fresh-incomplete >> train_run.log 2>&1 || exit 1
        [ "$(ls models/ddqn_*.zip 2>/dev/null | wc -l | tr -d ' ')" -eq 4 ] || exit 1
        "$PY" "$ROOT/code/evaluate_agents.py" >> evaluate_agents.log 2>&1 || exit 1
        [ -f eval_results.json ] || exit 1
        touch .done
    ) && echo "[done] $(date +%H:%M:%S) $tag" || echo "[FAIL] $(date +%H:%M:%S) $tag"
}
echo "rotation agents started $(date +%H:%M:%S)"
run cov1_cnnA_ms1 & run cov1_cnnB_ms1 & run cov1_cnnA_ms2 & run cov1_cnnB_ms2 & wait
echo "rotation agents finished $(date +%H:%M:%S)"
