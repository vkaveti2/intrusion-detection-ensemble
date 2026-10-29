#!/usr/bin/env bash
# package_benchmark.sh — bundle everything benchmark_edge.py needs into one folder that can be copied to
# another machine (an SD-card, scp, a USB stick). Nothing is trained or evaluated here.
#
#   ./code/package_benchmark.sh [output-dir]            # default: ./benchmark_bundle
#
# The bundle holds the benchmark code, the detector sets (cov0 unhardened, cov3 fully hardened), the dev probe
# bank, the pinned requirements, a checksum list, and run_on_target.sh, which sets up and runs everything on the
# target. Models are the frozen v5 models (config/protocol_v5.json); checksums are verified on the target.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/benchmark_bundle}"
SETS="cov3 cov0"
rm -rf "$OUT"; mkdir -p "$OUT/code/models_v5" "$OUT/code/splits_v5"

cp "$ROOT"/code/{benchmark_edge.py,ensemble.py,cnn_arch.py,masked_scaler.py,flow_features.py,project_log.py,requirements-pi.txt} "$OUT/code/"
for s in $SETS; do
    mkdir -p "$OUT/code/models_v5/$s"
    for f in xgboost_model.pkl rf_model.pkl cnn_model.pth cnn_arch.json scaler.pkl ensemble_weights.json label_encoder.pkl; do
        cp "$ROOT/code/models_v5/$s/$f" "$OUT/code/models_v5/$s/$f"
    done
done
# only the dev flows are needed; copying the whole bank would also ship final-test flows
"$ROOT/ids_env/bin/python" - "$ROOT/code/splits_v5/probe_bank.npz" "$OUT/code/splits_v5/probe_bank.npz" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1], allow_pickle=False)
names = [str(n) for n in z["label_names"]]
m = z["label"] == names.index("dev")
np.savez_compressed(sys.argv[2], flows=z["flows"][m], label=np.full(int(m.sum()), names.index("dev"), np.int8),
                    label_names=z["label_names"], feature_names=z["feature_names"])
print(f"dev bank: {int(m.sum())} flows")
PY
cp "$ROOT/docs/13_edge_benchmark.md" "$OUT/METHOD.md"

cat > "$OUT/run_on_target.sh" <<'EOS'
#!/usr/bin/env bash
# run_on_target.sh — set up and run the benchmark on the machine this folder was copied to.
#   ./run_on_target.sh <label>      e.g. ./run_on_target.sh edge-device-1
# Needs python3 (3.9-3.12) and internet access for the first run (pip installs the pinned libraries, about 1 GB
# with torch). Keep the machine otherwise idle, plugged in, and with a heatsink/fan as normally used.
set -euo pipefail
cd "$(dirname "$0")"
LABEL="${1:?usage: $0 <label>}"
sha256sum -c SHA256SUMS --quiet 2>/dev/null || shasum -a 256 -c SHA256SUMS --quiet
echo "checksums OK"
[ -d venv ] || python3 -m venv venv
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r code/requirements-pi.txt
export OMP_NUM_THREADS=1
echo "== machine"; uname -a; (cat /proc/device-tree/model 2>/dev/null; echo) || true
(vcgencmd measure_temp; vcgencmd get_throttled) 2>/dev/null || true
for set in cov3 cov0; do
    echo "== $set"
    venv/bin/python code/benchmark_edge.py --set "$set" --label "$LABEL" --repeats 5 --samples 2000 \
        --bank code/splits_v5/probe_bank.npz --out-dir results
done
echo "Done. Copy the 'results' folder back (benchmark_${LABEL}_cov3.json and benchmark_${LABEL}_cov0.json)."
EOS
chmod +x "$OUT/run_on_target.sh"
( cd "$OUT" && find code -type f | sort | xargs shasum -a 256 > SHA256SUMS )
echo "bundle: $OUT ($(du -sh "$OUT" | cut -f1)), $(wc -l < "$OUT/SHA256SUMS") files"
