"""
evaluate_agents.py — Untrained baseline vs trained DDQN evasion, per defence.

1. Untrained baseline: uniform-random probes, scored under every rule on the
   *same* probes (paired), broken down by payload size. This is the evasion
   rate an attacker gets with no learning, the reference point for RQ1/RQ2.
2. Trained agents: each models/ddqn_{rule}_{knowledge}.zip run greedily
   (no exploration) on fresh seeds.

Run from the directory train_ensemble.py was run from.
Output: eval_results.json
"""

import argparse
import json
import os
os.environ["OMP_NUM_THREADS"] = "1"
from collections import Counter

import numpy as np
from stable_baselines3 import DQN

from ensemble import MODEL_DIR, VOTING_RULES, EnsembleIDS
from ids_env import (EPISODE_LENGTH, N_ACTIONS, PKTSIZE_OPTIONS, IDSEvasionEnv,
                     decode_action)
from project_log import log_event

EVAL_SEED = 12345  # distinct from the training seed


def untrained_baseline(env, rules, n_probes):
    rng = np.random.default_rng(EVAL_SEED)
    env.reset(seed=EVAL_SEED)
    detectors = env.ensemble.detectors
    deciders = {r: EnsembleIDS(voting=r, detectors=detectors) for r in rules}
    evaded = {r: [] for r in rules}
    sizes = []
    for i in range(n_probes):
        _, _, pkt_size = decode_action(int(rng.integers(N_ACTIONS)))
        f = env._probe_features(1001 + i % EPISODE_LENGTH, pkt_size)
        votes, confs = {}, {}
        for det in detectors:
            votes[det.name], confs[det.name] = det.predict(f)
        for r in rules:
            evaded[r].append(deciders[r].decide(votes, confs)["label"] == "normal")
        sizes.append(pkt_size)

    sizes = np.array(sizes)
    out = {}
    for r in rules:
        e = np.array(evaded[r])
        out[r] = {
            "evasion": float(e.mean()),
            "by_payload": {str(s): float(e[sizes == s].mean()) for s in PKTSIZE_OPTIONS},
        }
    return out


def evaluate_policy(env, model, episodes):
    evasions, sizes, delays = 0, Counter(), []
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + 1 + ep)
        for _ in range(EPISODE_LENGTH):
            action, _ = model.predict(obs, deterministic=True)
            delay, _, pkt_size = decode_action(int(action))
            obs, _, done, _, info = env.step(action)
            evasions += info.get("classification") == "normal"
            sizes[pkt_size] += 1
            delays.append(delay)
            if done:
                break
    n = sum(sizes.values())
    return {
        "evasion": evasions / n,
        "payload_choice": {str(k): v / n for k, v in sorted(sizes.items())},
        "median_delay_s": float(np.median(delays)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=int, default=5000, help="Probes for the untrained baseline")
    parser.add_argument("--episodes", type=int, default=20, help="Greedy episodes per trained agent")
    parser.add_argument("--models-dir", default="./models")
    args = parser.parse_args()

    rules = list(VOTING_RULES)
    env = IDSEvasionEnv(knowledge="black", ensemble_rule="majority")
    shared = env.ensemble.detectors

    print(f"Untrained baseline ({args.probes} random probes, paired across rules)...")
    results = {"untrained": untrained_baseline(env, rules, args.probes), "trained": {}}

    for knowledge in ("black", "white"):
        env.knowledge = knowledge
        for rule in rules:
            path = os.path.join(args.models_dir, f"ddqn_{rule}_{knowledge}.zip")
            if not os.path.exists(path):
                continue
            env.ensemble = EnsembleIDS(voting=rule, detectors=shared)
            model = DQN.load(path, device="cpu")  # inference only; same network as DoubleDQN
            results["trained"][f"{rule}_{knowledge}"] = evaluate_policy(env, model, args.episodes)
            print(f"  {rule:9s} {knowledge}: {results['trained'][f'{rule}_{knowledge}']['evasion']:.1%}")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n| Defence | Untrained | Trained black-box | Trained white-box |")
    print("|---|---|---|---|")
    for r in rules:
        b = results["trained"].get(f"{r}_black", {}).get("evasion")
        w = results["trained"].get(f"{r}_white", {}).get("evasion")
        fmt = lambda x: "—" if x is None else f"{x:.1%}"
        print(f"| {r} | {results['untrained'][r]['evasion']:.1%} | {fmt(b)} | {fmt(w)} |")
    print("\nUntrained evasion by payload size (bytes):")
    for r in rules:
        print(f"  {r:9s}", {k: f"{v:.0%}" for k, v in results["untrained"][r]["by_payload"].items()})
    print("\nSaved → eval_results.json")

    pct = lambda x: "  —  " if x is None else f"{x:6.1%}"
    lines = [f"detector set={MODEL_DIR}  results dir={os.getcwd()}",
             f"untrained: {args.probes} random probes (paired across rules); "
             f"trained: {args.episodes} greedy episodes each, eval seed {EVAL_SEED}",
             "defence     untrained  trained-black  trained-white"]
    for r in rules:
        lines.append(f"{r:10s}  {pct(results['untrained'][r]['evasion'])}     "
                     f"{pct(results['trained'].get(f'{r}_black', {}).get('evasion'))}         "
                     f"{pct(results['trained'].get(f'{r}_white', {}).get('evasion'))}")
    lines.append("output: eval_results.json")
    log_event("Agent evaluation (evaluate_agents.py)", lines)
