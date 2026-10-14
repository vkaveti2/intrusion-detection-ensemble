#!/usr/bin/env bash
# run_pi_benchmark.sh — RQ3: run benchmark_edge.py on the Raspberry Pi from the Mac.
#
#   ./code/run_pi_benchmark.sh pi@172.16.105.41
#
# Copies only what the benchmark needs (code + both detector sets), builds a
# venv on the Pi with requirements-pi.txt, benchmarks the original (models/)
# and adversarially trained (models_adv/) detectors, copies the JSON results
# back next to this repo, and appends a summary to PROJECT_LOG.txt.
set -euo pipefail

PI="${1:?usage: $0 user@pi-host}"
SAMPLES="${SAMPLES:-2000}"
REMOTE_DIR="ids_bench"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"

echo "==> Copying code and detector artifacts to $PI:~/$REMOTE_DIR"
ssh "$PI" "mkdir -p ~/$REMOTE_DIR/code/models ~/$REMOTE_DIR/code/models_adv"
rsync -av "$HERE"/{ensemble.py,cnn_arch.py,benchmark_edge.py,project_log.py,requirements-pi.txt} \
    "$PI:~/$REMOTE_DIR/code/"
for set in models models_adv; do
    rsync -av "$HERE/$set"/{xgboost_model.pkl,rf_model.pkl,cnn_model.pth,cnn_arch.json,scaler.pkl,label_encoder.pkl,ensemble_weights.json} \
        "$PI:~/$REMOTE_DIR/code/$set/"
done
rsync -av "$HERE/models/probe_bank.npz" "$PI:~/$REMOTE_DIR/code/models/"

echo "==> Building venv and running benchmark on the Pi (first run installs torch; slow)"
ssh "$PI" bash -s <<EOF
set -euo pipefail
cd ~/$REMOTE_DIR
rm -f PROJECT_LOG.txt   # Pi-side log holds only this run's entries
python3 -m venv venv
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r code/requirements-pi.txt
venv/bin/python code/benchmark_edge.py --samples $SAMPLES
IDS_MODEL_DIR=\$HOME/$REMOTE_DIR/code/models_adv venv/bin/python code/benchmark_edge.py --samples $SAMPLES
EOF

echo "==> Fetching results"
rsync -av "$PI:~/$REMOTE_DIR/benchmark_*.json" "$REPO/"
ssh "$PI" "cat ~/$REMOTE_DIR/PROJECT_LOG.txt" >> "$REPO/PROJECT_LOG.txt"
echo "Done. Results: $REPO/benchmark_*.json (Pi entries appended to PROJECT_LOG.txt)"
