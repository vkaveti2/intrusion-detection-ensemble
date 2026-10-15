#!/usr/bin/env bash
# run_step3.sh — Step 3: multi-seed replication (docs/05_multiseed_design.md).
#
#   ./code/run_step3.sh
#
# Part A (agent seeds): the 14 DDQN experiments for every detector set
#   (cov0 models/, cov1 models_cov1/, cov2 models_cov2/, cov3 models_adv/) with
#   agent seeds 1–4. Seed 0 already exists (repo root, results_v4_cov1/,
#   results_v4_cov2/, results_v3_adversarial/) and reproduces exactly.
#   Output: results_multiseed/<set>/seed_<n>/
# Part B (detector seeds): cov1 retrained with model seeds 1–3 (same data,
#   different training randomness), then evaluated and attacked with agent seed 0.
#   Output: code/models_cov1_ms<s>/, results_multiseed/cov1_ms<s>/
#
# [auto] log entries go to docs/logs/PROJECT_LOG_step3_auto.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/ids_env/bin/python"
export ROOT PY
export PROJECT_LOG_PATH="$ROOT/docs/logs/PROJECT_LOG_step3_auto.txt"
mkdir -p "$ROOT/docs/logs" "$ROOT/results_multiseed"

model_dir() {
    case "$1" in
        cov0) echo "$ROOT/code/models" ;;
        cov1) echo "$ROOT/code/models_cov1" ;;
        cov2) echo "$ROOT/code/models_cov2" ;;
        cov3) echo "$ROOT/code/models_adv" ;;
        cov1_ms*) echo "$ROOT/code/models_${1}" ;;
    esac
}

# One batch = 14 experiments for one detector set and one agent seed, then evaluation
agent_batch() {
    local set="$1" seed="$2" out="$3"
    mkdir -p "$out"
    (cd "$out" &&
        IDS_MODEL_DIR="$(model_dir "$set")" "$PY" "$ROOT/code/train_ensemble.py" \
            --seed "$seed" --max-parallel 7 > train_run.log 2>&1 &&
        IDS_MODEL_DIR="$(model_dir "$set")" "$PY" "$ROOT/code/evaluate_agents.py" \
            > evaluate_agents.log 2>&1) \
        && echo "[A] $(date +%H:%M:%S) done $set seed $seed" \
        || echo "[A] $(date +%H:%M:%S) FAILED $set seed $seed"
}
export -f model_dir agent_batch

part_a() {
    # Most informative sets first; two batches (14 processes) at a time
    { for s in 1 2 3 4; do echo "cov0 $s"; echo "cov1 $s"; done
      for s in 1 2 3 4; do echo "cov2 $s"; echo "cov3 $s"; done; } |
        xargs -P 2 -L 1 bash -c 'agent_batch "$0" "$1" "$ROOT/results_multiseed/$0/seed_$1"'
}

part_b() {
    local s
    for s in 1 2 3; do mkdir -p "$ROOT/results_multiseed/cov1_ms$s"; done
    train() {  # $1 = stage, $2 = model seed
        (cd "$ROOT/code" && "$PY" train_adversarial.py --design cov1 --stage "$1" --model-seed "$2" \
            > "$ROOT/results_multiseed/cov1_ms$2/train_detectors_$1.log" 2>&1)
    }
    # trees one at a time (each uses every core); CNNs two at a time on the GPU
    (train trees 1; train trees 2; train trees 3) &
    (train cnn 1; train cnn 3) &
    (train cnn 2) &
    wait
    for s in 1 2 3; do
        grep -q "Epoch 30/30" "$ROOT/results_multiseed/cov1_ms$s/train_detectors_cnn.log" \
            || { echo "[B] CNN model seed $s did not complete"; continue; }
        (cd "$ROOT/code" && IDS_MODEL_DIR="$(model_dir "cov1_ms$s")" "$PY" evaluate_detectors.py \
            > "$ROOT/results_multiseed/cov1_ms$s/evaluate_detectors.log" 2>&1)
        agent_batch "cov1_ms$s" 0 "$ROOT/results_multiseed/cov1_ms$s"
        echo "[B] $(date +%H:%M:%S) done cov1 model seed $s"
    done
}

echo "step 3 started $(date +%H:%M:%S)"
part_b &
part_a
wait
echo "step 3 finished $(date +%H:%M:%S)"
