#!/usr/bin/env bash
# run_psh_retrain_rest.sh — remaining feature-masked (no PSH Flag Count) detector sets:
# cov2, and cov1 retrained with model seeds 1-3. Companion to run_psh_retrain.sh
# (cov1 seed 42 and full hardening). Design: docs/09_psh_diagnostic_design.md.
# Outputs: code/models_cov2_nopsh/, code/models_cov1_nopsh_ms{1,2,3}/, logs in results_psh_masked/.
# Trees one at a time (each uses all cores); two CNNs at a time on the GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/ids_env/bin/python"
export PROJECT_LOG_PATH="$ROOT/docs/logs/PROJECT_LOG_psh_auto.txt"
OUT="$ROOT/results_psh_masked"; mkdir -p "$OUT"
MASK=(--mask-feature "PSH Flag Count")
cd "$ROOT/code"

# $1 = design, $2 = stage, $3 = model seed or "-" for the default seed 42
train() {
    local tag="$1"; local seedargs=()
    [ "$3" != "-" ] && { seedargs=(--model-seed "$3"); tag="${1}_ms$3"; }
    "$PY" train_adversarial.py --design "$1" --stage "$2" ${seedargs[@]+"${seedargs[@]}"} "${MASK[@]}" \
        > "$OUT/train_${tag}_$2.log" 2>&1
}
echo "psh retrain (rest) started $(date +%H:%M:%S)"
(train cov2 trees -; train cov1 trees 1; train cov1 trees 2; train cov1 trees 3) &
(train cov2 cnn -; train cov1 cnn 2) &
(train cov1 cnn 1; train cov1 cnn 3) &
wait
for pair in "cov2:-:models_cov2_nopsh" "cov1:1:models_cov1_nopsh_ms1" "cov1:2:models_cov1_nopsh_ms2" "cov1:3:models_cov1_nopsh_ms3"; do
    IFS=: read -r d s m <<< "$pair"
    tag="$d"; [ "$s" != "-" ] && tag="${d}_ms$s"
    grep -q "Epoch 30/30" "$OUT/train_${tag}_cnn.log" || { echo "CNN $tag did not complete"; continue; }
    IDS_MODEL_DIR="$ROOT/code/$m" "$PY" evaluate_detectors.py > "$OUT/evaluate_detectors_$m.log" 2>&1
    IDS_MODEL_DIR="$ROOT/code/$m" "$PY" psh_sensitivity.py > "$OUT/psh_sensitivity_$m.log" 2>&1
    echo "evaluated $m $(date +%H:%M:%S)"
done
echo "psh retrain (rest) finished $(date +%H:%M:%S)"
