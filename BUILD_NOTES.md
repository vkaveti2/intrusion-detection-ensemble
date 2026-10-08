# Thesis Ensemble Build — 2026-09-24

## What was built
- `src/ensemble.py` — EnsembleIDS with BaseDetector interface, 3 mock detectors
  (XGBoost/timing-sensitive, CNN/pattern-sensitive, RF/balanced-noisy), and 3
  voting rules (majority, unanimous, weighted). Standalone smoke test included.
- `src/ids_env.py` — added `ensemble` param to IDSEvasionEnv; ensemble mode
  builds 154-dim features locally from probe params, skips Pi socket + Scapy
  I/O + wall-clock delay sleep. Fixed missing EPISODE_LENGTH import.
- `src/train_ensemble.py` — DDQN training vs ensemble (mirrors dqn.py:
  checkpointing, resume, convergence JSON). Trains per (voting, knowledge).

## Bugs fixed
1. `dqn.py` imported EPISODE_LENGTH which `ids_env.py` never defined → added
   EPISODE_LENGTH=100 (matches target_ports range(1,101)).
2. Ensemble-mode invariant checker saw all-"unknown" port results → every probe
   violated failed-ratio, episodes never terminated → mock port states now
   (22/80/443 open, rest closed).
3. Wall-clock `time.sleep(delay)` (up to 100s/probe) made local training
   infeasible → skipped in ensemble mode (delay still in features + reward).

## Verification (all passing)
- ensemble.py smoke: majority/unanimous/weighted behave as designed;
  slow+stealthy probes evade majority (the learnable exploit).
- 250-step DDQN vs majority/black: ep1 59% evasion → ep2 99% evasion,
  convergence JSON + model zip saved.
- Unanimous easiest to evade (86% ep1) — matches theory (fool 1 of 3).

## Still needed (next steps toward real ensemble)
1. Replace mock detectors with real models: train XGBoost/CNN/RF on CIC IDS
   2017, export (ONNX/TFLite), subclass BaseDetector with load().
2. Pi-side: run ensemble on the Pi 5 or keep local inference; wire real
   154-dim features (currently synthetic from probe params).
3. Calibrate: real models will be harder to evade; re-tune reward/threshold.
4. Gemini API was at 503 capacity all session — retry design review later.
5. Google Docs not connected — couldn't read "New Thesis Proposal" doc.
   Thesis_Context_file.pdf + Drive file listing used instead.

## Notion (Sem 7, Thursday)
- Thesis task: "see if we can develop script for ensemble w what we have" ✓ done
- Short-term: "get logans fucking script" (still open)

## New Thesis Proposal (read 2026-09-24 via Google Docs)
- Working title: "Intrusion Detection: Evaluating Adversarial Resilience and
  Convergence Dynamics of Multi Model Ensemble Systems on Edge Hardware"
- RQ1: DDQN evasion success vs ensemble compared to standalone models.
- RQ2: which voting rule (majority, unanimous, confidence-weighted) resists
  the agent best.
- RQ3: Pi overhead — latency, memory footprint, throughput of multi-model
  inference on the gateway.
- Plan: CICIDS2017 preprocessing -> standalone XGBoost/1D-CNN/RF -> gym-style
  env for DDQN attacks -> ensemble voting -> convergence analysis -> Pi bench.
- Deliverables: 50-page honors thesis + conference manuscript with
  Dr. Saha and Dr. Lee.
- AI policy (agreed w/ Dr. Saha): AI may build detectors + DDQN agent; must NOT
  automate/synthesize empirical evaluations. All Pi benchmarks collected via
  direct hardware testing, logged manually. -> our mock results are pipeline
  validation ONLY, never thesis evidence.
