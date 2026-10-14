"""
benchmark_edge.py — RQ3: computational overhead of multi-model inference.

Run on the Raspberry Pi gateway (and on the Mac for comparison):
    python benchmark_edge.py --samples 2000

For each standalone detector and each ensemble rule, measures on real
held-out PortScan flows (models/probe_bank.npz):
  - load time and resident memory added by each detector
  - single-flow latency (mean, p50, p95, p99) — the per-packet decision path
  - throughput: single-stream flows/s, and batched flows/s (batch of 256)
  - CPU load: process CPU time / wall time during the latency run

Output: benchmark_<hostname>.json
"""

import argparse
import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
import platform
import resource
import socket
import sys
import time

import numpy as np


def rss_mb():
    """Current resident set size in MB (Linux /proc; peak RSS fallback elsewhere)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def latency_run(fn, flows, warmup=50):
    for f in flows[:warmup]:
        fn(f)
    lat = np.empty(len(flows))
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for i, f in enumerate(flows):
        t = time.perf_counter()
        fn(f)
        lat[i] = time.perf_counter() - t
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    ms = lat * 1e3
    return {
        "mean_ms": float(ms.mean()), "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)), "p99_ms": float(np.percentile(ms, 99)),
        "single_stream_flows_per_s": float(len(flows) / wall),
        "cpu_load_pct": float(100 * cpu / wall),
    }


def batched_throughput(det, flows, batch=256, repeats=5):
    X = det.scaler.transform(flows[:batch]) if det.scaler is not None else flows[:batch]
    X = np.asarray(X, dtype=np.float32)
    if det.name == "cnn":
        import torch
        x = torch.from_numpy(X)
        run = lambda: det.model(x)
        ctx = torch.no_grad()
    else:
        run = lambda: det.model.predict_proba(X)
        ctx = None
    if ctx:
        ctx.__enter__()
    run()
    t = time.perf_counter()
    for _ in range(repeats):
        run()
    elapsed = time.perf_counter() - t
    if ctx:
        ctx.__exit__(None, None, None)
    return float(batch * repeats / elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ensemble import (MODEL_DIR, STANDALONE_RULES, VOTING_RULES, CNNDetector,
                          EnsembleIDS, RandomForestDetector, XGBoostDetector)
    # Loaded directly (not via ids_env) so the Pi doesn't need scapy/gymnasium
    bank = np.load(os.path.join(MODEL_DIR, "probe_bank.npz"))["flows"]
    flows = bank[np.random.default_rng(0).integers(len(bank), size=args.samples)]

    report = {
        "host": socket.gethostname(), "machine": platform.machine(),
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "samples": args.samples,
        "baseline_rss_mb": rss_mb(), "detectors": {}, "rules": {},
    }

    detectors = []
    for cls in (XGBoostDetector, RandomForestDetector, CNNDetector):
        before, t = rss_mb(), time.perf_counter()
        det = cls()
        load_s, added = time.perf_counter() - t, rss_mb() - before
        if det.is_mock:
            sys.exit(f"{det.name} failed to load real artifacts — benchmark would be meaningless.")
        detectors.append(det)
        ext = "pth" if det.name == "cnn" else "pkl"
        report["detectors"][det.name] = {
            "load_s": load_s, "added_rss_mb": added,
            "model_file_mb": os.path.getsize(os.path.join(MODEL_DIR, f"{det.name}_model.{ext}")) / 2**20,
            "batched_flows_per_s": batched_throughput(det, flows),
        }
        print(f"[{det.name}] loaded in {load_s:.2f}s, +{added:.0f} MB", flush=True)
    report["total_rss_mb"] = rss_mb()

    for rule in VOTING_RULES:
        ens = EnsembleIDS(voting=rule, detectors=detectors)
        report["rules"][rule] = latency_run(ens.classify, flows)
        r = report["rules"][rule]
        print(f"[{rule:9s}] p50={r['p50_ms']:.2f}ms p99={r['p99_ms']:.2f}ms "
              f"{r['single_stream_flows_per_s']:.0f} flows/s cpu={r['cpu_load_pct']:.0f}%", flush=True)

    ens_p50 = np.mean([report["rules"][r]["p50_ms"] for r in VOTING_RULES if r not in STANDALONE_RULES])
    single_p50 = np.mean([report["rules"][r]["p50_ms"] for r in STANDALONE_RULES])
    report["ensemble_vs_standalone_latency_ratio"] = float(ens_p50 / single_p50)

    out = f"benchmark_{report['host']}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTotal RSS with all detectors: {report['total_rss_mb']:.0f} MB")
    print(f"Ensemble / mean standalone p50 latency: {report['ensemble_vs_standalone_latency_ratio']:.2f}x")
    print(f"Saved → {out}")

    from project_log import log_event
    lines = [f"host={report['host']}  machine={report['machine']}  cpus={report['cpu_count']}  "
             f"samples={args.samples}",
             f"memory with all detectors: {report['total_rss_mb']:.0f} MB  ("
             + ", ".join(f"{n} +{d['added_rss_mb']:.0f} MB" for n, d in report["detectors"].items()) + ")",
             "defence     p50 ms   p99 ms   flows/s   cpu%"]
    for rule, r in report["rules"].items():
        lines.append(f"{rule:10s}  {r['p50_ms']:6.2f}   {r['p99_ms']:6.2f}   "
                     f"{r['single_stream_flows_per_s']:7.0f}   {r['cpu_load_pct']:4.0f}")
    lines.append(f"ensemble / standalone p50 latency: "
                 f"{report['ensemble_vs_standalone_latency_ratio']:.2f}x   output: {out}")
    log_event("Edge benchmark (benchmark_edge.py)", lines)
