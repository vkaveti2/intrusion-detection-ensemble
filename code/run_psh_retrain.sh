#!/usr/bin/env bash
# run_psh_retrain.sh — retrain cov1 and full (v3) detector sets without PSH Flag Count.
# Design and decision rule: docs/09_psh_diagnostic_design.md. Outputs: code/models_cov1_nopsh/,
# code/models_adv_nopsh/, logs in results_psh_masked/. Trees run one at a time (each uses all
# cores); CNNs run on the GPU alongside.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/ids_env/bin/python"
export PROJECT_LOG_PATH="$ROOT/docs/logs/PROJECT_LOG_psh_auto.txt"
OUT="$ROOT/results_psh_masked"; mkdir -p "$OUT"
MASK=(--mask-feature "PSH Flag Count")
cd "$ROOT/code"
train() { "$PY" train_adversarial.py --design "$1" --stage "$2" "${MASK[@]}" > "$OUT/train_$1_$2.log" 2>&1; }
echo "psh retrain started $(date +%H:%M:%S)"
(train cov1 trees; train full trees) &
(train cov1 cnn; train full cnn) &
wait
for pair in "cov1:models_cov1_nopsh" "full:models_adv_nopsh"; do
    d="${pair%%:*}"; m="${pair##*:}"
    grep -q "Epoch 30/30" "$OUT/train_${d}_cnn.log" || { echo "CNN $d did not complete"; continue; }
    IDS_MODEL_DIR="$ROOT/code/$m" "$PY" evaluate_detectors.py > "$OUT/evaluate_detectors_$m.log" 2>&1
    IDS_MODEL_DIR="$ROOT/code/$m" "$PY" psh_sensitivity.py > "$OUT/psh_sensitivity_$m.log" 2>&1
    echo "evaluated $m $(date +%H:%M:%S)"
done
echo "psh retrain finished $(date +%H:%M:%S)"
