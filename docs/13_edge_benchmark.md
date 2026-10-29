# 13 — Inference Cost Benchmark (RQ3): Method

Code: `code/benchmark_edge.py` (measurement), `code/run_benchmark_local.sh` (this machine), `code/package_benchmark.sh` (bundle for another machine). Results are written to `results_v5/benchmark/benchmark_<label>_<set>.json` and summarised in `docs/13_benchmark_results.md` by `summarize_benchmark.py`.

RQ3 asks what multi-model inference costs: latency, throughput and memory, for the ensemble compared with each standalone detector.

## What is measured, and what is not

| Measured | Not measured |
|---|---|
| Inference on one 77-feature flow record: feature scaling, the model call(s), and the voting rule | Packet capture |
| The payload-to-feature recomputation the simulator uses (`features`), reported as its own line | Flow-feature extraction from packets (CICFlowMeter). `features` is a cheap proxy for the arithmetic only, not for extraction |
| Model load time and resident memory | Network or storage I/O |

Every figure in `13_benchmark_results.md` is therefore **inference time**. It is a lower bound on the per-flow cost of a deployed system, which must also capture packets and extract features.

## Configurations

Seven, plus the feature line. Standalone: `xgboost`, `rf`, `cnn`. Ensembles: `majority`, `weighted`, `unanimous`, `any`. Each ensemble runs all three detectors on every flow and applies its rule afterwards, so the four ensembles cost the same to within the vote (microseconds). The detector sets timed are `cov3` (full hardening, the robustness reference) and `cov0` (unhardened). Their models differ in content but not in architecture, so costs are expected to match closely; both are timed so that is checked, not assumed.

## Named baseline and slowdown

The baseline is **standalone XGBoost** (`--baseline xgboost`): the strongest observed standalone comparator in the robustness results and the fastest standalone model. Every slowdown is `configuration / baseline` for latency and memory, and `baseline / configuration` for throughput, so a value above 1.00x always means worse than XGBoost. XGBoost against itself reads exactly 1.00x.

## Procedure

Identical on every machine.

1. **Workload.** 2,000 flows per repeat drawn (seeded) from the dev probe bank, each padded to one of the five agent payload sizes (0, 64, 128, 256, 512 B). No model is selected or scored here, and no final-test data is involved.
2. **One process per configuration per repeat.** Each timed configuration starts in a fresh Python process, so memory is independent of load order and nothing is cached from a previous configuration.
3. **Randomised order.** The configurations are run in a different random order in every repeat, so drift (temperature, background load) is not tied to one configuration.
4. **Warm-up.** 200 flows are classified and discarded before timing.
5. **Single-flow path (batch size 1).** Each flow is timed individually with `time.perf_counter()` through `EnsembleIDS.classify`, which is the path the simulator and a live gateway would use. This gives the latency distribution (mean, p50, p90, p95, p99, max), single-stream throughput (flows per second) and CPU load (process CPU time over wall time).
6. **Staged pass.** 500 flows are timed again stage by stage: scaling, model call(s), vote. This shows where the time goes. The stages are compared with the end-to-end mean; the remainder is reported as `unattributed`.
7. **Batched pass (batch size 256).** The batch is scaled once and each active detector scores it. This gives batched throughput for a gateway that buffers flows.
8. **Consistency checks, on every run.** The staged decisions must equal the end-to-end decisions (500 of 500), and the batched decisions must equal the single-flow decisions (256 of 256). Both counts are stored in the result file.
9. **Repeats.** Five repeats; every reported statistic is the median over repeats, with the range stored for p50 and p99.

## Settings recorded with every result

Hostname, device model, CPU, physical and logical cores, memory, operating system, Python and library versions, `OMP_NUM_THREADS` (set to 1 so inference is single-threaded and comparable across machines), CPU governor and frequency where the system exposes them, temperature and throttle flags before and after every repeat where available, the SHA-256 of every model file and of the probe bank, the workload description, and the timestamp.

## Memory

Three numbers per configuration, from the process's resident set size: `libs` (Python and libraries imported, models not yet loaded), `loaded` (after loading the models), `peak` (the maximum during the run). `added_by_models = loaded - libs`. Because each configuration runs in its own process, the ensemble figure is what a single process holding all three models actually needs, not a sum of standalone figures.

## Reading the results

- Latency is judged by p50 for the typical case and p99 for the worst case that matters to a gateway.
- The single-stream throughput is roughly the reciprocal of the mean latency. The batched figure is **not** a reliable measure of the CNN: in a direct check on the development machine (one thread, no oneDNN), the CNN's forward pass cost 82 microseconds per flow at batch 1 and 40 at batch 4, then about 1,290 at batch 16 and above, so batching made it roughly 16 times slower per flow. That is a property of PyTorch's CPU convolution path on that build, not of the model, and it can differ on other hardware and builds. The tree models behave normally (batching speeds them up). Treat the single-flow (batch size 1) figures as the primary result and the CNN's batched figure as platform-specific.
- Absolute numbers belong to one machine. Slowdown ratios against XGBoost transfer much better between machines than absolute latencies do, but they are still measurements on specific hardware and should be quoted with the host they came from.

## Running on another machine

`./code/package_benchmark.sh` builds a self-contained folder (code, the two detector sets, the dev flows only, pinned requirements, checksums, `METHOD.md` and `run_on_target.sh`). On the target, `./run_on_target.sh <label>` verifies the checksums, creates a virtual environment, installs the pinned libraries, runs both detector sets, and writes the result files to `results/`. Copy those two JSON files back into `results_v5/benchmark/` and rerun `summarize_benchmark.py`; the tables then include the new host. The library versions are pinned because the model files are pickles and are only guaranteed to load under the versions they were saved with.

## Limits

- Inference only (see the table above).
- Single-threaded by design. A multi-core run would show a different, usually better, picture for the random forest.
- A short run on a machine that throttles under heat can drift; the temperature and throttle samples are stored so that can be seen.
- One training run of each detector set; the cost of a model does not depend on its training seed to first order, but this was not separately verified.
