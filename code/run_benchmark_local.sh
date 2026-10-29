#!/usr/bin/env bash
# run_benchmark_local.sh — benchmark_edge.py for the unhardened (cov0) and fully hardened (cov3) detector sets
# on this machine. Output: results_v5/benchmark/benchmark_<label>_<set>.json.   Method: docs/13_edge_benchmark.md
#   LABEL=mac-m5-max caffeinate -i ./code/run_benchmark_local.sh
# Keep the machine otherwise idle while it runs.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/ids_env/bin/python"
LABEL="${LABEL:-$(hostname -s)}"
for set in cov3 cov0; do
    echo "== $set $(date +%H:%M:%S)"
    "$PY" "$ROOT/code/benchmark_edge.py" --set "$set" --label "$LABEL" --repeats 5 --samples 2000 || echo "FAILED $set"
done
echo "finished $(date +%H:%M:%S)"
