"""
benchmark_edge.py — computational cost of each detector and decision rule (RQ3). Method: docs/13_edge_benchmark.md.

    python benchmark_edge.py --set cov3 --label my-host            # 7 configurations x 5 repeats

Configurations: the standalone detectors (xgboost, rf, cnn) and the four voting rules (majority, weighted,
unanimous, any). Every voting rule runs all three detectors on every flow, so their inference cost is the same
up to the vote.  A pseudo-configuration "features" times the payload-to-feature recomputation the simulator uses.

Procedure (identical on every host):
  - each configuration runs in its own fresh Python process, so memory is not affected by load order;
  - the configurations are run in a different random order in every repeat, so drift (thermal, background load)
    is not tied to one configuration;
  - per process: load the models, warm up, then time the single-flow path (batch size 1) end to end, then a
    separate staged pass (scaling / model / vote), then a batched pass (batch 256);
  - the numbers reported are the median over repeats of each statistic.

What is timed: inference only, on 77-feature flow records that already exist. Packet capture and flow-feature
extraction (CICFlowMeter) are NOT measured; "features" times only the payload recomputation (a proxy).

Workload: flows from the dev probe bank (splits_v5/probe_bank.npz), each padded to one of the agent's five
payload sizes, seeded. Nothing here selects or evaluates a model, so no final-test data is involved.

Slowdown is relative to a named baseline (--baseline, default xgboost): latency and memory as config / baseline,
throughput as baseline / config, so a value above 1.00x always means worse than the baseline.

Output: <out-dir>/benchmark_<label>_<set>.json  (default results_v5/benchmark/)
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STANDALONE = ("xgboost", "rf", "cnn")
CONFIGS = STANDALONE + ("majority", "weighted", "unanimous", "any")
PAYLOADS = (0, 64, 128, 256, 512)
LAT_KEYS = ("mean", "p50", "p90", "p95", "p99", "max")
BATCH = 256
MARK = "BENCH_RESULT_JSON:"
MODEL_FILES = ("xgboost_model.pkl", "rf_model.pkl", "cnn_model.pth", "cnn_arch.json", "scaler.pkl",
               "ensemble_weights.json")


# ── Measurement helpers ───────────────────────────────────────────────────────
def rss_mb():
    """Current resident set size in MB."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        try:   # macOS: current RSS in KB
            return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 1024
        except Exception:
            return peak_rss_mb()


def peak_rss_mb():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def latency_stats(seconds):
    ms = np.asarray(seconds) * 1e3
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p90": float(np.percentile(ms, 90)),
            "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)), "max": float(ms.max())}


def load_workload(bank_path, samples, seed=0):
    """Dev-bank flows, each padded to one of the agent's payload sizes. Returns (flows, raw_flows, sizes)."""
    from flow_features import set_forward_payload
    z = np.load(bank_path, allow_pickle=False)
    names = [str(n) for n in z["label_names"]]
    pool = z["flows"][z["label"] == names.index("dev")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(len(pool), size=samples)
    sizes = rng.choice(PAYLOADS, size=samples)
    raw = pool[idx]
    flows = np.empty_like(raw, dtype=np.float32)
    for s in PAYLOADS:
        m = sizes == s
        if m.any():
            flows[m] = set_forward_payload(raw[m], s)
    return flows, raw, sizes


def apply_rule(cfg, P, weights):
    """P: (k, n) attack probabilities in detector order -> (n,) bool, as ensemble.EnsembleIDS.decide."""
    votes = P >= 0.5
    if cfg in STANDALONE:
        return votes[0]
    n = votes.sum(axis=0)
    if cfg == "majority":
        return n >= 2
    if cfg == "unanimous":
        return n == 3
    if cfg == "any":
        return n >= 1
    return (weights[:, None] * P).sum(axis=0) >= 0.5


# ── Worker: one configuration in a fresh process ──────────────────────────────
def worker_features(a):
    from flow_features import set_forward_payload
    rss0 = rss_mb()
    _, raw, sizes = load_workload(a.bank, a.samples)
    for i in range(min(a.warmup, len(raw))):
        set_forward_payload(raw[i:i + 1], int(sizes[i]))
    gc.collect()
    lat = np.empty(len(raw))
    w0 = time.perf_counter()
    for i in range(len(raw)):
        t = time.perf_counter()
        set_forward_payload(raw[i:i + 1], int(sizes[i]))
        lat[i] = time.perf_counter() - t
    wall = time.perf_counter() - w0
    return {"config": "features", "latency_ms": latency_stats(lat), "single_stream_flows_per_s": len(raw) / wall,
            "memory_mb": {"libs": rss0, "loaded": rss0, "peak": peak_rss_mb()}}


def worker_detectors(cfg, a):
    flows, _, _ = load_workload(a.bank, a.samples)
    from ensemble import EnsembleIDS, XGBoostDetector, RandomForestDetector, CNNDetector
    rss_libs = rss_mb()
    make = {"xgboost": XGBoostDetector, "rf": RandomForestDetector, "cnn": CNNDetector}
    names = [cfg] if cfg in STANDALONE else list(STANDALONE)
    dets, load_s = [], {}
    for n in names:
        t = time.perf_counter()
        d = make[n]()
        load_s[n] = time.perf_counter() - t
        if d.is_mock:
            sys.exit(f"{n} loaded as a mock, not the real model; the benchmark would be meaningless")
        dets.append(d)
    rss_loaded = rss_mb()
    ens = EnsembleIDS(voting=cfg, detectors=dets)
    torch = None
    if "cnn" in names:
        import torch
    n = len(flows)

    # 1. end-to-end single-flow path (batch size 1)
    for i in range(min(a.warmup, n)):
        ens.classify(flows[i])
    gc.collect()
    lat, labels = np.empty(n), [None] * n
    cpu0, w0 = time.process_time(), time.perf_counter()
    for i in range(n):
        t = time.perf_counter()
        r = ens.classify(flows[i])
        lat[i] = time.perf_counter() - t
        labels[i] = r["label"]
    wall, cpu = time.perf_counter() - w0, time.process_time() - cpu0

    # 2. staged pass: where does the time go (scaling / model / vote)?
    m = min(a.staged, n)
    scale_t, model_t, vote_t, staged_labels = np.zeros(m), np.zeros(m), np.zeros(m), [None] * m
    pc = time.perf_counter
    for i in range(m):
        f, probs = flows[i], {}
        for d in dets:
            t0 = pc()
            X = d.scale_features(f)
            t1 = pc()
            if d.name == "cnn":
                with torch.no_grad():
                    probs[d.name] = torch.softmax(d.model(torch.tensor(X, dtype=torch.float32)), dim=1)[0].numpy()
            else:
                probs[d.name] = d.model.predict_proba(X)[0]
            t2 = pc()
            scale_t[i] += t1 - t0
            model_t[i] += t2 - t1
        t3 = pc()
        votes, confs = {}, {}
        for d in dets:
            p_scan = 1.0 - probs[d.name][0]
            votes[d.name] = "scanning" if p_scan >= 0.5 else "normal"
            confs[d.name] = p_scan if p_scan >= 0.5 else probs[d.name][0]
        staged_labels[i] = ens.decide(votes, confs)["label"]
        vote_t[i] = pc() - t3

    # 3. batched pass (scale the batch once, then every active detector, then the rule)
    import pandas as pd
    scaler = dets[0].scaler
    Xb_raw = flows[:min(BATCH, n)]

    def batch_run():
        Xs = scaler.transform(pd.DataFrame(Xb_raw, columns=scaler.feature_names_in_)).astype(np.float32)
        P = []
        for d in dets:
            if d.name == "cnn":
                with torch.no_grad():
                    P.append(1.0 - torch.softmax(d.model(torch.from_numpy(Xs)), dim=1).numpy()[:, 0])
            else:
                P.append(1.0 - d.model.predict_proba(Xs)[:, 0])
        return apply_rule(cfg, np.stack(P), ens.weights)

    decisions = batch_run()
    t = pc()
    for _ in range(a.batch_repeats):
        batch_run()
    batched = len(Xb_raw) * a.batch_repeats / (pc() - t)
    single_scanning = np.array([lbl == "scanning" for lbl in labels[:len(Xb_raw)]])

    return {
        "config": cfg, "latency_ms": latency_stats(lat),
        "single_stream_flows_per_s": n / wall, "cpu_load_pct": 100 * cpu / wall, "batched_flows_per_s": batched,
        "stages_ms_mean": {"scale": float(scale_t.mean() * 1e3), "model": float(model_t.mean() * 1e3),
                           "vote": float(vote_t.mean() * 1e3)},
        "memory_mb": {"libs": rss_libs, "loaded": rss_loaded, "peak": peak_rss_mb()},
        "load_s": load_s, "torch_threads": (torch.get_num_threads() if torch else None),
        "agreement": {"staged_vs_end_to_end": [int(sum(x == y for x, y in zip(staged_labels, labels[:m]))), m],
                      "batched_vs_single_flow": [int((decisions == single_scanning).sum()), int(len(single_scanning))]},
    }


# ── Host description ──────────────────────────────────────────────────────────
def sh(cmd):
    env = dict(os.environ, PATH=os.environ.get("PATH", "") + ":/usr/sbin:/sbin:/usr/bin:/opt/vc/bin")  # sysctl, vcgencmd
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10, env=env).decode().strip()
    except Exception:
        return None


def read(path):
    try:
        with open(path) as f:
            return f.read().strip().replace("\x00", "")
    except OSError:
        return None


def thermal_sample():
    """Temperature and throttle flags where the platform exposes them (None elsewhere)."""
    out = {"time": time.strftime("%H:%M:%S")}
    t = sh(["vcgencmd", "measure_temp"])
    out["temp"] = t
    out["throttled"] = sh(["vcgencmd", "get_throttled"])
    freq = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    out["cpu0_mhz"] = int(freq) // 1000 if freq else None
    return out


def describe_host():
    import numpy, pandas, scipy, sklearn, xgboost, joblib, torch
    h = {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
         "python": platform.python_version(), "cpu_count": os.cpu_count(),
         "libraries": {m.__name__: m.__version__ for m in (numpy, pandas, scipy, sklearn, xgboost, joblib, torch)},
         "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    if sys.platform == "darwin":
        h["cpu"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        mem = sh(["sysctl", "-n", "hw.memsize"])
        h["memory_gb"] = round(int(mem) / 2**30, 1) if mem else None
        h["physical_cores"] = int(sh(["sysctl", "-n", "hw.physicalcpu"]) or 0) or None
        h["power"] = (sh(["pmset", "-g", "batt"]) or "").split("\n")[0]
    else:
        info = read("/proc/cpuinfo") or ""
        pick = lambda key: next((l.split(":", 1)[1].strip() for l in info.split("\n") if l.lower().startswith(key)), None)
        h["cpu"] = pick("model name") or pick("hardware") or pick("model")
        h["device_model"] = read("/proc/device-tree/model")
        mem = next((l for l in (read("/proc/meminfo") or "").split("\n") if l.startswith("MemTotal")), None)
        h["memory_gb"] = round(int(re.findall(r"\d+", mem)[0]) / 2**20, 1) if mem else None
        h["governor"] = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        h["os"] = next((l.split("=", 1)[1].strip('"') for l in (read("/etc/os-release") or "").split("\n")
                        if l.startswith("PRETTY_NAME")), None)
    return h


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Orchestration ─────────────────────────────────────────────────────────────
def run_worker(cfg, a, model_dir):
    env = dict(os.environ, IDS_MODEL_DIR=model_dir, OMP_NUM_THREADS="1")
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", cfg, "--bank", a.bank, "--samples", str(a.samples),
           "--warmup", str(a.warmup), "--staged", str(a.staged), "--batch-repeats", str(a.batch_repeats)]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=7200)
    line = next((l for l in reversed(p.stdout.split("\n")) if l.startswith(MARK)), None)
    if p.returncode != 0 or line is None:
        sys.exit(f"worker '{cfg}' failed (exit {p.returncode}):\n{p.stderr[-1500:]}\n{p.stdout[-500:]}")
    return json.loads(line[len(MARK):])


def med(values):
    return float(np.median(values))


def aggregate(runs):
    r0 = runs[0]
    out = {"latency_ms": {k: med([r["latency_ms"][k] for r in runs]) for k in LAT_KEYS},
           "latency_ms_range": {k: [min(r["latency_ms"][k] for r in runs), max(r["latency_ms"][k] for r in runs)]
                                for k in ("p50", "p99")},
           "single_stream_flows_per_s": med([r["single_stream_flows_per_s"] for r in runs]),
           "memory_mb": {k: med([r["memory_mb"][k] for r in runs]) for k in ("libs", "loaded", "peak")}}
    out["memory_mb"]["added_by_models"] = out["memory_mb"]["loaded"] - out["memory_mb"]["libs"]
    if "batched_flows_per_s" in r0:
        out["batched_flows_per_s"] = med([r["batched_flows_per_s"] for r in runs])
        out["cpu_load_pct"] = med([r["cpu_load_pct"] for r in runs])
        st = {k: med([r["stages_ms_mean"][k] for r in runs]) for k in ("scale", "model", "vote")}
        st["unattributed"] = out["latency_ms"]["mean"] - sum(st.values())
        out["stages_ms_mean"] = st
        out["load_s"] = {k: med([r["load_s"][k] for r in runs]) for k in r0["load_s"]}
        out["torch_threads"] = r0["torch_threads"]
        out["agreement"] = {k: [sum(r["agreement"][k][0] for r in runs), sum(r["agreement"][k][1] for r in runs)]
                            for k in r0["agreement"]}
    return out


def slowdowns(agg, baseline):
    b = agg[baseline]
    out = {}
    for cfg, x in agg.items():
        if cfg == "features":
            continue
        out[cfg] = {"p50_latency_x": x["latency_ms"]["p50"] / b["latency_ms"]["p50"],
                    "p99_latency_x": x["latency_ms"]["p99"] / b["latency_ms"]["p99"],
                    "single_stream_throughput_x": b["single_stream_flows_per_s"] / x["single_stream_flows_per_s"],
                    "batched_throughput_x": b["batched_flows_per_s"] / x["batched_flows_per_s"],
                    "peak_memory_x": x["memory_mb"]["peak"] / b["memory_mb"]["peak"]}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="cov3", help="detector set under models_v5/ (default cov3, full hardening)")
    ap.add_argument("--model-dir", default=None, help="explicit detector directory (overrides --set)")
    ap.add_argument("--bank", default=os.path.join(HERE, "splits_v5", "probe_bank.npz"))
    ap.add_argument("--samples", type=int, default=2000, help="flows timed per repeat")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--staged", type=int, default=500, help="flows in the staged (scale/model/vote) pass")
    ap.add_argument("--batch-repeats", type=int, default=5)
    ap.add_argument("--baseline", default="xgboost", choices=CONFIGS)
    ap.add_argument("--label", default=None, help="name for this host in the output file (default: hostname)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(HERE), "results_v5", "benchmark"))
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.worker:
        res = worker_features(a) if a.worker == "features" else worker_detectors(a.worker, a)
        print(MARK + json.dumps(res))
        return

    model_dir = os.path.abspath(a.model_dir or os.path.join(HERE, "models_v5", a.set))
    for f in MODEL_FILES:
        if not os.path.exists(os.path.join(model_dir, f)):
            sys.exit(f"missing {os.path.join(model_dir, f)}")
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", a.label or socket.gethostname())
    host = describe_host()
    print(f"host: {host.get('device_model') or host.get('cpu')} | {host['machine']} | {host['cpu_count']} logical cpus | "
          f"python {host['python']} | detector set {os.path.basename(model_dir)}", flush=True)

    rng = np.random.default_rng(1)
    names = ("features",) + CONFIGS
    runs, thermal = {c: [] for c in names}, [thermal_sample()]
    for rep in range(a.repeats):
        order = [names[i] for i in rng.permutation(len(names))]
        for cfg in order:
            runs[cfg].append(run_worker(cfg, a, model_dir))
        thermal.append(thermal_sample())
        print(f"repeat {rep + 1}/{a.repeats} done (order: {', '.join(order)})", flush=True)

    agg = {c: aggregate(r) for c, r in runs.items()}
    report = {
        "label": label, "detector_set": os.path.basename(model_dir), "baseline": a.baseline,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"), "host": host,
        "settings": {"flows_per_repeat": a.samples, "repeats": a.repeats, "warmup_flows": a.warmup,
                     "staged_flows": a.staged, "single_flow_batch_size": 1, "batched_pass_batch_size": BATCH,
                     "batched_pass_repeats": a.batch_repeats, "configuration_order": "shuffled in every repeat (seed 1)",
                     "process": "fresh Python process per configuration per repeat",
                     "workload": "dev probe-bank flows padded to 0/64/128/256/512 B, seed 0",
                     "statistic": "median over repeats",
                     "excluded": "packet capture and flow-feature extraction are not measured"},
        "bank_sha256": sha256(a.bank),
        "model_files": {f: {"sha256": sha256(os.path.join(model_dir, f)),
                            "mb": round(os.path.getsize(os.path.join(model_dir, f)) / 2**20, 3)} for f in MODEL_FILES},
        "thermal": thermal, "results": agg, "slowdown_vs_baseline": slowdowns(agg, a.baseline),
    }
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"benchmark_{label}_{os.path.basename(model_dir)}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'config':10s} {'p50 ms':>8s} {'p99 ms':>8s} {'flows/s':>9s} {'batched/s':>10s} {'peak MB':>8s}   vs {a.baseline} (p50 / p99 / mem)")
    sd = report["slowdown_vs_baseline"]
    for c in CONFIGS:
        x = agg[c]
        print(f"{c:10s} {x['latency_ms']['p50']:8.3f} {x['latency_ms']['p99']:8.3f} {x['single_stream_flows_per_s']:9.0f} "
              f"{x['batched_flows_per_s']:10.0f} {x['memory_mb']['peak']:8.0f}   {sd[c]['p50_latency_x']:.2f}x / "
              f"{sd[c]['p99_latency_x']:.2f}x / {sd[c]['peak_memory_x']:.2f}x")
    print(f"features (payload recomputation): p50 {agg['features']['latency_ms']['p50']:.4f} ms")
    bad = [c for c in CONFIGS if agg[c]["agreement"]["staged_vs_end_to_end"][0] != agg[c]["agreement"]["staged_vs_end_to_end"][1]]
    print("staged pass agrees with the end-to-end decisions in every configuration" if not bad else f"DISAGREEMENT in {bad}")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
