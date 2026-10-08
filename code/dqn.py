import gymnasium as gym
import time
import json
import multiprocessing
import os
from pathlib import Path
from stable_baselines3 import DQN
from ids_env import IDSEvasionEnv, EPISODE_LENGTH
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList

TOTAL_TIMESTEPS  = 15_000
CHECKPOINT_FREQ  = 500
CHECKPOINT_DIR   = "./checkpoints"


def get_exploration_rate(model, steps_done, total_timesteps):
    """
    Recalculate exploration rate from training progress.
    SB3's exploration_schedule maps progress_remaining (1.0 → 0.0)
    linearly from exploration_initial_eps to exploration_final_eps
    over exploration_fraction of total_timesteps. This is deterministic
    given the fixed schedule parameters so we can recover the exact
    value without reading from the checkpoint file.
    """
    progress_remaining = 1.0 - steps_done / total_timesteps
    return float(model.exploration_schedule(progress_remaining))


class CustomCallback(BaseCallback):
    def __init__(self, model_name, knowledge, total_timesteps, verbose=0, existing_convergence=None):
        super().__init__(verbose)
        self.model_name       = model_name
        self.knowledge        = knowledge
        self.total_timesteps  = total_timesteps
        self.total_reward     = 0.0
        self.total_probes     = 0
        self.successful_evasions = 0
        self.convergence      = existing_convergence or {}
        self.episode_number   = max((int(k) for k in self.convergence), default=0)
        self._start_time      = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        info   = self.locals["infos"][0]
        reward = self.locals["rewards"][0]
        self.total_reward  += float(reward)
        self.total_probes  += 1

        if info.get("classification") == "normal":
            self.successful_evasions += 1

        if self.locals["dones"][0]:
            success_rate        = self.successful_evasions / max(self.total_probes, 1)
            self.episode_number += 1

            self.convergence[self.episode_number] = [
                float(self.total_reward),
                float(success_rate),
            ]

            elapsed      = time.time() - self._start_time
            steps_done   = self.num_timesteps
            steps_left   = self.total_timesteps - steps_done
            eps_done     = self.episode_number
            eps_per_sec  = eps_done / elapsed if elapsed > 0 else 0
            eps_per_min  = eps_per_sec * 60
            steps_per_ep = steps_done / eps_done if eps_done > 0 else EPISODE_LENGTH
            eps_left     = steps_left / steps_per_ep if steps_per_ep > 0 else 0
            eta_secs     = eps_left / eps_per_sec if eps_per_sec > 0 else float("inf")
            eta_str      = f"{eta_secs/3600:.1f}h" if eta_secs != float("inf") else "?"

            print(
                f"[DDQN] {self.model_name}/{self.knowledge} "
                f"ep={self.episode_number} "
                f"steps={steps_done}/{self.total_timesteps} "
                f"R={self.total_reward:.1f} "
                f"evasion={success_rate:.1%} "
                f"eps/min={eps_per_min:.1f} "
                f"ETA={eta_str}"
            )

            self.total_reward        = 0.0
            self.total_probes        = 0
            self.successful_evasions = 0

            self._save_convergence()

        return True

    def _save_convergence(self):
        path = f"convergence_{self.model_name}_{self.knowledge}.json"
        serialisable = {
            k: [float(v[0]), float(v[1])] for k, v in self.convergence.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, indent=2)

    def _on_training_end(self) -> None:
        self._save_convergence()
        print(f"[DDQN] Convergence data saved for {self.model_name} / {self.knowledge}")


def find_latest_checkpoint(model_name, knowledge):
    ckpt_dir = Path(CHECKPOINT_DIR)
    if not ckpt_dir.exists():
        return None, 0
    prefix = f"dqn_{model_name}_{knowledge}_"
    checkpoints = sorted(
        ckpt_dir.glob(f"{prefix}*_steps.zip"),
        key=lambda p: int(p.stem.replace(prefix, "").replace("_steps", ""))
    )
    if not checkpoints:
        return None, 0
    latest = checkpoints[-1]
    try:
        steps = int(latest.stem.replace(prefix, "").replace("_steps", ""))
    except ValueError:
        steps = 0
    return str(latest), steps


def train_model(model_name, pi_ip, conditions, total_timesteps=15_000):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for knowledge in conditions:
        print(f"[DDQN] Starting: model={model_name} knowledge={knowledge} pi={pi_ip}")
        env = IDSEvasionEnv(pi_ip=pi_ip, model=model_name, knowledge=knowledge)

        convergence_path = f"convergence_{model_name}_{knowledge}.json"
        existing_convergence = {}
        if os.path.exists(convergence_path):
            with open(convergence_path, "r") as f:
                existing_convergence = json.load(f)
            print(f"[DDQN] Resuming — loaded {len(existing_convergence)} episodes from convergence log")

        conv_callback = CustomCallback(
            model_name=model_name,
            knowledge=knowledge,
            total_timesteps=total_timesteps,
            verbose=1,
            existing_convergence=existing_convergence,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=CHECKPOINT_DIR,
            name_prefix=f"dqn_{model_name}_{knowledge}",
            verbose=1,
        )

        callbacks = CallbackList([conv_callback, checkpoint_callback])

        latest_ckpt, steps_done = find_latest_checkpoint(model_name, knowledge)
        remaining = total_timesteps - steps_done

        if latest_ckpt and remaining > 0:
            print(f"[DDQN] Resuming from {latest_ckpt} ({steps_done} steps done, {remaining} remaining)")
            model = DQN.load(latest_ckpt, env=env)
            # Recalculate exploration rate from training progress —
            # SB3 resets it on load so we restore it mathematically
            # from the fixed schedule parameters.
            model.exploration_rate = get_exploration_rate(model, steps_done, total_timesteps)
            print(f"[DDQN] Restored exploration_rate={model.exploration_rate:.4f}")
            model.learn(
                total_timesteps=remaining,
                callback=callbacks,
                reset_num_timesteps=False,
            )
        elif remaining <= 0:
            print(f"[DDQN] Already complete: {steps_done} >= {total_timesteps} timesteps")
            env.close()
            continue
        else:
            print(f"[DDQN] Fresh training run — {total_timesteps} timesteps")
            model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=1e-4,
                buffer_size=10_000,
                learning_starts=100,
                batch_size=32,
                gamma=0.99,
                exploration_fraction=0.3,
                exploration_final_eps=0.05,
                train_freq=4,
                target_update_interval=100,
            )

            timesteps_done = 0
            while timesteps_done < total_timesteps:
                remaining = total_timesteps - timesteps_done
                try:
                    model.learn(
                        total_timesteps=remaining,
                        callback=callbacks,
                        reset_num_timesteps=False,
                    )
                    timesteps_done = total_timesteps
                except Exception as e:
                    print(f"[DDQN] Training error: {e}")
                    print(f"[DDQN] Saving checkpoint and restarting in 15 seconds...")
                    try:
                        model.save(f"dqn_{model_name}_{knowledge}_recovery")
                    except Exception as save_err:
                        print(f"[DDQN] Save failed: {save_err}")
                    time.sleep(15)
                    timesteps_done = model.num_timesteps
                    print(f"[DDQN] Resuming from timestep {timesteps_done}")

        model.save(f"dqn_{model_name}_{knowledge}")
        print(f"[DDQN] Done: dqn_{model_name}_{knowledge}.zip saved")
        env.close()


training_options = {
    "xgboost": {"ip": "10.0.0.18", "conditions": ["gray", "white"]},
    "cnn":     {"ip": "10.0.0.18", "conditions": ["gray", "white"]},
    "mlp":     {"ip": "10.0.0.18", "conditions": ["gray", "white"]},
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DDQN IDS Evasion Training")
    parser.add_argument(
        "--models", nargs="+",
        choices=["xgboost", "cnn", "mlp"],
        default=["xgboost", "cnn", "mlp"],
    )
    parser.add_argument(
        "--knowledge", nargs="+",
        choices=["black", "gray", "white"],
        default=["gray", "white"],
    )
    parser.add_argument(
        "--timesteps", type=int,
        default=TOTAL_TIMESTEPS,
    )
    args = parser.parse_args()

    print(f"[DDQN] Training config:")
    print(f"  Models      : {args.models}")
    print(f"  Knowledge   : {args.knowledge}")
    print(f"  Timesteps   : {args.timesteps}")
    print(f"  Episode len : {EPISODE_LENGTH} probes")
    print(f"  Est episodes: {args.timesteps // EPISODE_LENGTH} per condition")

    processes = []
    for model_name in args.models:
        config = training_options[model_name]
        conditions = [k for k in args.knowledge if k in config["conditions"]]
        if not conditions:
            continue
        p = multiprocessing.Process(
            target=train_model,
            args=(model_name, config["ip"], conditions, args.timesteps)
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("[DDQN] All training complete.")
