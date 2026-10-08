"""
train_ensemble.py — DDQN training against the multi-model ensemble IDS.

Mirrors dqn.py but targets EnsembleIDS instead of single Pi models.
Trains one agent per (voting_rule, knowledge) combination and logs
convergence for the thesis comparison matrix (Experiment Series 3).

Usage:
    python train_ensemble.py --voting majority unanimous weighted --knowledge black gray white --timesteps 15000
"""

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, CallbackList,
)

from ids_env import IDSEvasionEnv, EPISODE_LENGTH
from ensemble import EnsembleIDS, VOTING_RULES

TOTAL_TIMESTEPS = 15_000
CHECKPOINT_FREQ = 500
CHECKPOINT_DIR = "./checkpoints_ensemble"


def get_exploration_rate(model, steps_done, total_timesteps):
    progress_remaining = 1.0 - steps_done / total_timesteps
    return float(model.exploration_schedule(progress_remaining))


class EnsembleCallback(BaseCallback):
    def __init__(self, voting, knowledge, total_timesteps, verbose=0,
                 existing_convergence=None):
        super().__init__(verbose)
        self.voting = voting
        self.knowledge = knowledge
        self.total_timesteps = total_timesteps
        self.total_reward = 0.0
        self.total_probes = 0
        self.successful_evasions = 0
        self.convergence = existing_convergence or {}
        self.episode_number = max((int(k) for k in self.convergence), default=0)
        self._start_time = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        self.total_reward += float(reward)
        self.total_probes += 1

        if info.get("classification") == "normal":
            self.successful_evasions += 1

        if self.locals["dones"][0]:
            success_rate = self.successful_evasions / max(self.total_probes, 1)
            self.episode_number += 1
            self.convergence[self.episode_number] = [
                float(self.total_reward), float(success_rate),
            ]

            elapsed = time.time() - self._start_time
            steps_done = self.num_timesteps
            eps_per_min = (self.episode_number / elapsed * 60) if elapsed > 0 else 0

            print(
                f"[ENSEMBLE] {self.voting}/{self.knowledge} "
                f"ep={self.episode_number} steps={steps_done}/{self.total_timesteps} "
                f"R={self.total_reward:.1f} evasion={success_rate:.1%} "
                f"eps/min={eps_per_min:.1f}"
            )

            self.total_reward = 0.0
            self.total_probes = 0
            self.successful_evasions = 0
            self._save_convergence()
        return True

    def _save_convergence(self):
        path = f"convergence_ensemble_{self.voting}_{self.knowledge}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({k: [float(v[0]), float(v[1])]
                       for k, v in self.convergence.items()}, f, indent=2)

    def _on_training_end(self) -> None:
        self._save_convergence()
        print(f"[ENSEMBLE] Convergence saved for {self.voting}/{self.knowledge}")


def find_latest_checkpoint(voting, knowledge):
    ckpt_dir = Path(CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        return None, 0
    prefix = f"dqn_ensemble_{voting}_{knowledge}_"
    cks = sorted(
        ckpt_dir.glob(f"{prefix}*_steps.zip"),
        key=lambda p: int(p.stem.replace(prefix, "").replace("_steps", "")),
    )
    if not cks:
        return None, 0
    latest = cks[-1]
    try:
        steps = int(latest.stem.replace(prefix, "").replace("_steps", ""))
    except ValueError:
        steps = 0
    return str(latest), steps


def train_ensemble(voting, knowledge_list, total_timesteps=TOTAL_TIMESTEPS):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for knowledge in knowledge_list:
        tag = f"ensemble_{voting}_{knowledge}"
        print(f"[ENSEMBLE] Starting: voting={voting} knowledge={knowledge}")

        env = IDSEvasionEnv(model="ensemble", knowledge=knowledge,
                            ensemble=voting)

        conv_path = f"convergence_{tag}.json"
        existing = {}
        if os.path.exists(conv_path):
            with open(conv_path) as f:
                existing = json.load(f)
            print(f"[ENSEMBLE] Resuming — {len(existing)} episodes logged")

        callbacks = CallbackList([
            EnsembleCallback(voting, knowledge, total_timesteps,
                             verbose=1, existing_convergence=existing),
            CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=CHECKPOINT_DIR,
                               name_prefix=f"dqn_{tag}", verbose=1),
        ])

        latest_ckpt, steps_done = find_latest_checkpoint(voting, knowledge)
        remaining = total_timesteps - steps_done

        if latest_ckpt and remaining > 0:
            print(f"[ENSEMBLE] Resuming from {latest_ckpt}")
            model = DQN.load(latest_ckpt, env=env)
            model.exploration_rate = get_exploration_rate(
                model, steps_done, total_timesteps)
            model.learn(total_timesteps=remaining, callback=callbacks,
                        reset_num_timesteps=False)
        elif remaining <= 0:
            print(f"[ENSEMBLE] Already complete")
            env.close()
            continue
        else:
            print(f"[ENSEMBLE] Fresh run — {total_timesteps} timesteps")
            model = DQN("MlpPolicy", env, verbose=1, learning_rate=1e-4,
                        buffer_size=10_000, learning_starts=100, batch_size=32,
                        gamma=0.99, exploration_fraction=0.3,
                        exploration_final_eps=0.05, train_freq=4,
                        target_update_interval=100)
            model.learn(total_timesteps=total_timesteps, callback=callbacks)

        model.save(f"dqn_{tag}")
        print(f"[ENSEMBLE] Done: dqn_{tag}.zip saved")
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDQN vs Ensemble IDS")
    parser.add_argument("--voting", nargs="+", choices=list(VOTING_RULES),
                        default=["majority"])
    parser.add_argument("--knowledge", nargs="+",
                        choices=["black", "gray", "white"],
                        default=["black"])
    parser.add_argument("--timesteps", type=int, default=TOTAL_TIMESTEPS)
    args = parser.parse_args()

    print("[ENSEMBLE] Training config:")
    print(f"  Voting    : {args.voting}")
    print(f"  Knowledge : {args.knowledge}")
    print(f"  Timesteps : {args.timesteps}")

    processes = []
    for voting in args.voting:
        p = multiprocessing.Process(
            target=train_ensemble,
            args=(voting, args.knowledge, args.timesteps))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print("[ENSEMBLE] All training complete.")
