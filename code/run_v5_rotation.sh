#!/usr/bin/env bash
# run_v5_rotation.sh — CNN band rotation (docs/10_band_rotation.md): CNN on band A or B, model seeds 1 and 2.
# Resumable: a set with models_v5/<tag>/cnn_training.json is skipped. Output: results_v5/rotation/.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/ids_env/bin/python"
export PROJECT_LOG_PATH="$ROOT/docs/logs/PROJECT_LOG_v5_rotation_auto.txt"
OUT="$ROOT/results_v5/rotation"; mkdir -p "$OUT" "$ROOT/docs/logs"
cd "$ROOT/code"
run() {   # $1 band, $2 model seed
    local tag="cov1_cnn$1_ms$2"
    if [ -f "$ROOT/code/models_v5/$tag/cnn_training.json" ]; then echo "[skip] $tag"; return; fi
    echo "[run ] $(date +%H:%M:%S) $tag"
    "$PY" train_v5_rotation.py --band "$1" --model-seed "$2" > "$OUT/train_$tag.log" 2>&1 \
        && echo "[done] $(date +%H:%M:%S) $tag" || echo "[FAIL] $(date +%H:%M:%S) $tag"
}
echo "rotation started $(date +%H:%M:%S)"
run A 1 & run B 1 & run A 2 & run B 2 & wait
echo "rotation finished $(date +%H:%M:%S)"
