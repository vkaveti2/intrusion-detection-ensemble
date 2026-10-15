"""
train_ensemble.py — DDQN Thesis Experiment Runner

Trains a Double DQN agent against every defence configuration — each
standalone detector (RQ1 baseline) and each ensemble voting rule (RQ2) —
at each knowledge level, one process per experiment. Each process:
  - Tracks evasion rate + cumulative reward per episode via CustomCallback
  - Saves a convergence JSON after every episode (thesis evidence for RQ1/RQ2)
  - Checkpoints the model every CHECKPOINT_FREQ steps (crash-safe)
  - Resumes from the latest checkpoint if one exists (safe to re-run)

Output files (all in /code directory):
  convergence_ddqn_{voting}_{knowledge}.json  ← thesis data (RQ1, RQ2)
  checkpoints_ensemble/ddqn_{voting}_{knowledge}_{N}_steps.zip
  models/ddqn_{voting}_{knowledge}.zip        ← final trained agent brain
"""

import json
import multiprocessing
import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)

from ensemble import MODEL_DIR, STANDALONE_RULES, VOTING_RULES
from ids_env import IDSEvasionEnv, EPISODE_LENGTH
from project_log import log_event

# ── Experiment matrix ─────────────────────────────────────────────────────────
# RQ1: standalone detectors (xgboost, rf, cnn) vs ensembles
# RQ2: voting rules, from most permissive (unanimous) to most defensive (any)
KNOWLEDGE_LEVELS = ("black", "white")

TIMESTEPS       = 50_000   # Steps per experiment (≈500 episodes @ 100 steps/ep)
CHECKPOINT_FREQ = 1_000    # Save a zip every N steps
CHECKPOINT_DIR  = "./checkpoints_ensemble"
MODELS_DIR      = "./models"
DEFAULT_SEED    = 0        # agent init, exploration and probe-flow sampling; --seed overrides


# ── Double DQN ────────────────────────────────────────────────────────────────
class DoubleDQN(DQN):
    """
    SB3's DQN bootstraps with max_a Q_target(s', a), which is vanilla DQN.
    Double DQN (van Hasselt et al., 2016) selects a' with the online network
    and evaluates it with the target network, reducing overestimation bias.
    Only the TD target differs from DQN.train().
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_actions = self.q_net(replay_data.next_observations).argmax(dim=1, keepdim=True)
                next_q_values = th.gather(
                    self.q_net_target(replay_data.next_observations), dim=1, index=next_actions
                )
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = th.gather(
                self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long()
            )
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# ── Exploration rate recovery after checkpoint resume ─────────────────────────
def get_exploration_rate(model, steps_done, total_timesteps):
    """
    SB3 resets exploration_rate on DQN.load(). We restore it deterministically
    from the fixed linear schedule so the agent continues annealing correctly.
    """
    progress_remaining = 1.0 - steps_done / total_timesteps
    return float(model.exploration_schedule(progress_remaining))


# ── Checkpoint resume helper ──────────────────────────────────────────────────
def find_latest_checkpoint(voting_rule, knowledge_level):
    """
    Scans CHECKPOINT_DIR for the highest-step checkpoint for this experiment.
    Returns (path_str_or_None, steps_done_int).
    """
    ckpt_dir = Path(CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        return None, 0
    prefix = f"ddqn_{voting_rule}_{knowledge_level}_"
    checkpoints = sorted(
        ckpt_dir.glob(f"{prefix}*_steps.zip"),
        key=lambda p: int(p.stem.replace(prefix, "").replace("_steps", "")),
    )
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    try:
        steps = int(latest.stem.replace(prefix, "").replace("_steps", ""))
    except ValueError:
        steps = 0
    return str(latest), steps


# ── Per-episode convergence tracking callback ─────────────────────────────────
class ConvergenceCallback(BaseCallback):
    """
    Fires on every environment step. At episode end it:
      1. Computes evasion_rate = successful_evasions / total_probes
      2. Appends [total_reward, evasion_rate] to the convergence dict
      3. Writes the dict to convergence_ddqn_{voting}_{knowledge}.json
      4. Prints a progress line with ETA

    The JSON is written incrementally so a crash never loses prior episodes.
    Supports resume: pass existing_convergence loaded from a prior run.
    """

    def __init__(self, voting_rule, knowledge_level, total_timesteps,
                 existing_convergence=None, verbose=0):
        super().__init__(verbose)
        self.voting_rule       = voting_rule
        self.knowledge_level   = knowledge_level
        self.total_timesteps   = total_timesteps

        # Per-episode accumulators
        self.total_reward        = 0.0
        self.total_probes        = 0
        self.successful_evasions = 0

        # Convergence log (may be pre-populated from a prior run)
        self.convergence    = existing_convergence or {}
        self.episode_number = max((int(k) for k in self.convergence), default=0)
        self._start_time    = None

        # File path written after every episode
        self._conv_path = (
            f"convergence_ddqn_{voting_rule}_{knowledge_level}.json"
        )

    def _on_training_start(self) -> None:
        self._start_time = time.time()
        exp = f"ddqn_{self.voting_rule}/{self.knowledge_level}"
        print(f"[{exp}] Training started — {self.total_timesteps} timesteps")

    def _on_step(self) -> bool:
        info   = self.locals["infos"][0]
        reward = self.locals["rewards"][0]

        self.total_reward += float(reward)
        self.total_probes += 1

        if info.get("classification") == "normal":
            self.successful_evasions += 1

        # Episode boundary
        if self.locals["dones"][0]:
            evasion_rate        = self.successful_evasions / max(self.total_probes, 1)
            self.episode_number += 1

            # Record to convergence dict
            self.convergence[self.episode_number] = [
                float(self.total_reward),
                float(evasion_rate),
            ]

            # ETA calculation
            elapsed      = time.time() - self._start_time
            steps_done   = self.num_timesteps
            steps_left   = self.total_timesteps - steps_done
            eps_done     = self.episode_number
            eps_per_sec  = eps_done / elapsed if elapsed > 0 else 0
            eps_per_min  = eps_per_sec * 60
            steps_per_ep = steps_done / eps_done if eps_done > 0 else EPISODE_LENGTH
            eps_left     = steps_left / steps_per_ep if steps_per_ep > 0 else 0
            eta_secs     = eps_left / eps_per_sec if eps_per_sec > 0 else float("inf")
            eta_str      = f"{eta_secs / 3600:.1f}h" if eta_secs != float("inf") else "?"

            print(
                f"[ddqn_{self.voting_rule}/{self.knowledge_level}] "
                f"ep={self.episode_number} "
                f"steps={steps_done}/{self.total_timesteps} "
                f"R={self.total_reward:.1f} "
                f"evasion={evasion_rate:.1%} "
                f"eps/min={eps_per_min:.1f} "
                f"ETA={eta_str}"
            )

            # Reset accumulators
            self.total_reward        = 0.0
            self.total_probes        = 0
            self.successful_evasions = 0

            # Flush to disk immediately (crash-safe)
            self._flush_convergence()

        return True

    def _on_training_end(self) -> None:
        self._flush_convergence()
        ev = [v[1] for _, v in sorted(self.convergence.items(), key=lambda kv: int(kv[0]))]
        if ev:
            pct = lambda xs: 100 * sum(xs) / len(xs)
            log_event(
                f"Experiment finished: ddqn_{self.voting_rule}_{self.knowledge_level}",
                [f"episodes={len(ev)}  wall-clock={(time.time() - self._start_time) / 60:.1f} min",
                 f"evasion: first 10 eps {pct(ev[:10]):.1f}%  last 50 eps {pct(ev[-50:]):.1f}%  "
                 f"mean {pct(ev):.1f}%",
                 f"convergence file: {self._conv_path}"],
            )
        print(
            f"[ddqn_{self.voting_rule}/{self.knowledge_level}] "
            f"Convergence JSON saved → {self._conv_path}"
        )

    def _flush_convergence(self):
        serialisable = {
            k: [float(v[0]), float(v[1])]
            for k, v in self.convergence.items()
        }
        with open(self._conv_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)


# ── Single experiment runner (runs in its own process) ────────────────────────
def run_experiment(voting_rule, knowledge_level, total_timesteps=TIMESTEPS, seed=DEFAULT_SEED):
    """
    Trains one DDQN agent against ensemble_{voting_rule} with {knowledge_level}
    knowledge. Handles fresh start and checkpoint resume transparently.
    Output files carry no seed in their names: run each seed in its own directory.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    exp_name = f"ddqn_{voting_rule}_{knowledge_level}"
    print(f"\n{'='*60}")
    print(f" Experiment: {exp_name}")
    print(f" Timesteps : {total_timesteps}  |  Episode length: {EPISODE_LENGTH}")
    print(f"{'='*60}")
    log_event(f"Experiment started: {exp_name}",
              [f"timesteps={total_timesteps}  seed={seed}  agent=DoubleDQN  simulator=real-flow probe bank"])

    # ── Environment ───────────────────────────────────────────────────────────
    env = IDSEvasionEnv(
        knowledge=knowledge_level,
        ensemble_rule=voting_rule,
        use_pi=False,           # Local simulation (Mac Pro speed)
    )

    # ── Load existing convergence data if resuming ────────────────────────────
    conv_path = f"convergence_{exp_name}.json"
    existing_convergence = {}
    if os.path.exists(conv_path):
        with open(conv_path, "r") as f:
            existing_convergence = json.load(f)
        print(f"[{exp_name}] Resuming — {len(existing_convergence)} episodes already logged")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    conv_callback = ConvergenceCallback(
        voting_rule=voting_rule,
        knowledge_level=knowledge_level,
        total_timesteps=total_timesteps,
        existing_convergence=existing_convergence,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix=f"ddqn_{voting_rule}_{knowledge_level}",
        verbose=1,
    )
    callbacks = CallbackList([conv_callback, checkpoint_callback])

    # ── Find latest checkpoint (resume) or start fresh ───────────────────────
    latest_ckpt, steps_done = find_latest_checkpoint(voting_rule, knowledge_level)
    remaining = total_timesteps - steps_done

    if latest_ckpt and remaining > 0:
        print(f"[{exp_name}] Resuming from {latest_ckpt} ({steps_done} steps done, {remaining} left)")
        model = DoubleDQN.load(latest_ckpt, env=env)
        model.exploration_rate = get_exploration_rate(model, steps_done, total_timesteps)
        print(f"[{exp_name}] Restored exploration_rate={model.exploration_rate:.4f}")
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            reset_num_timesteps=False,
        )

    elif remaining <= 0:
        print(f"[{exp_name}] Already complete ({steps_done} >= {total_timesteps} steps). Skipping.")
        env.close()
        return

    else:
        print(f"[{exp_name}] Fresh training run — {total_timesteps} timesteps")
        env.reset(seed=seed)
        model = DoubleDQN(
            "MlpPolicy",
            env,
            verbose=0,                      # Silenced — ConvergenceCallback handles logging
            learning_rate=1e-4,
            buffer_size=100_000,
            learning_starts=2_000,          # Explore before learning
            batch_size=128,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1_000,   # Key DDQN stabilisation parameter
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            policy_kwargs=dict(net_arch=[256, 256]),
            device="cpu",                   # CPU faster than MPS for small MLP inference
            seed=seed,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
        )

    # ── Save final agent brain ────────────────────────────────────────────────
    final_path = os.path.join(MODELS_DIR, exp_name)
    model.save(final_path)
    print(f"[{exp_name}] ✓ Final model saved → {final_path}.zip")
    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DDQN Thesis Ensemble Experiments")
    parser.add_argument(
        "--voting", nargs="+",
        choices=list(VOTING_RULES),
        default=list(VOTING_RULES),
        help=f"Voting rules / standalone detectors to run (default: all; "
             f"standalone = {', '.join(STANDALONE_RULES)})",
    )
    parser.add_argument(
        "--knowledge", nargs="+",
        choices=["black", "gray", "white"],
        default=list(KNOWLEDGE_LEVELS),
        help="Knowledge levels to run (default: black white)",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=7,
        help="Max experiments running at once (each loads all three detectors)",
    )
    parser.add_argument(
        "--timesteps", type=int,
        default=TIMESTEPS,
        help=f"Timesteps per experiment (default: {TIMESTEPS})",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run experiments one-by-one instead of in parallel (lower RAM usage)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Agent/environment seed (default: {DEFAULT_SEED}); use a separate directory per seed",
    )
    args = parser.parse_args()

    # Build experiment list from CLI args
    experiments = [
        (v, k)
        for v in args.voting
        for k in args.knowledge
    ]

    print(f"\n[THESIS] Starting {len(experiments)} DDQN experiment(s)")
    print(f"  Voting rules : {args.voting}")
    print(f"  Knowledge    : {args.knowledge}")
    print(f"  Timesteps    : {args.timesteps}")
    print(f"  Episodes est.: ~{args.timesteps // EPISODE_LENGTH} per experiment")
    print(f"  Seed         : {args.seed}")
    print(f"  Mode         : {'sequential' if args.sequential else 'parallel'}\n")
    log_event(f"Training batch started: {len(experiments)} experiment(s)",
              [f"rules={args.voting}  knowledge={args.knowledge}  timesteps={args.timesteps}  seed={args.seed}",
               f"mode={'sequential' if args.sequential else f'parallel (max {args.max_parallel})'}",
               f"detector set={MODEL_DIR}  output dir={os.getcwd()}"])

    if args.sequential:
        for voting_rule, knowledge_level in experiments:
            run_experiment(voting_rule, knowledge_level, args.timesteps, args.seed)
    else:
        with multiprocessing.Pool(processes=args.max_parallel, maxtasksperchild=1) as pool:
            pool.starmap(
                run_experiment,
                [(v, k, args.timesteps, args.seed) for v, k in experiments],
                chunksize=1,
            )

    log_event(f"Training batch finished: {len(experiments)} experiment(s)",
              [f"seed={args.seed}  detector set={MODEL_DIR}  output dir={os.getcwd()}"])
    print("\n[THESIS] All experiments complete.")
    print("Convergence data → convergence_ddqn_*.json")
    print("Trained agents  → models/ddqn_*.zip")