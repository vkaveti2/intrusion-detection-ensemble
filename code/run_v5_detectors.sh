#!/usr/bin/env bash
# run_v5_detectors.sh — retrain every detector set under the corrected protocol (train_v5.py).
# Sets: cov0, cov1 (model seeds 42, 1, 2, 3), cov2, cov3. Output: code/models_v5/<set>/,
# logs in results_v5/. Trees one at a time (each uses all cores); CNNs three at a time on the GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/ids_env/bin/python"
export PROJECT_LOG_PATH="$ROOT/docs/logs/PROJECT_LOG_v5_auto.txt"
OUT="$ROOT/results_v5"; mkdir -p "$OUT" "$ROOT/docs/logs"
cd "$ROOT/code"

# $1 = design, $2 = stage, $3 = model seed or "-" for the default seed 42
train() {
    local tag="$1"; local seedargs=()
    if [ "$3" != "-" ]; then seedargs=(--model-seed "$3"); tag="${1}_ms$3"; fi
    "$PY" train_v5.py --design "$1" --stage "$2" ${seedargs[@]+"${seedargs[@]}"} > "$OUT/train_${tag}_$2.log" 2>&1
}
echo "v5 detector training started $(date +%H:%M:%S)"
(train cov0 trees -; train cov1 trees -; train cov2 trees -; train cov3 trees -;
 train cov1 trees 1; train cov1 trees 2; train cov1 trees 3) &
(train cov0 cnn -; train cov1 cnn 1; train cov1 cnn 3) &
(train cov1 cnn -; train cov2 cnn -) &
(train cov3 cnn -; train cov1 cnn 2) &
wait
for tag in cov0 cov1 cov2 cov3 cov1_ms1 cov1_ms2 cov1_ms3; do
    if grep -q "Epoch 30/30" "$OUT/train_${tag}_cnn.log" && [ -f "$ROOT/code/models_v5/$tag/rf_model.pkl" ]; then
        echo "complete $tag"
    else
        echo "INCOMPLETE $tag"
    fi
done
echo "v5 detector training finished $(date +%H:%M:%S)"
