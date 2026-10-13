import multiprocessing
import os
from stable_baselines3 import DQN
from ids_env import IDSEvasionEnv

# ── Thesis Experiment Matrix ──────────────────────────────────────────────
# We compare the two most important voting rules against 
# the two most important knowledge levels.
EXPERIMENTS = [
    ("majority", "white"),
    ("majority", "black"),
    ("unanimous", "white"),
    ("unanimous", "black"),
]

TIMESTEPS = 50000 # Increased for DDQN convergence

def run_ddqn_training(voting_rule, knowledge_level):
    """
    Trains a Double DQN agent against a specific ensemble configuration.
    """
    exp_name = f"ddqn_{voting_rule}_{knowledge_level}"
    print(f"[PROCESS] Starting Double DQN Experiment: {exp_name}")
    
    # Initialize Environment
    env = IDSEvasionEnv(
        knowledge=knowledge_level, 
        ensemble_rule=voting_rule, 
        use_pi=False # Simulation mode for Mac Pro speed
    )
    
    # Configure Double DQN
    # Stable-Baselines3 DQN implementation is DDQN by default.
    model = DQN(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=1e-4,     # Standard for DDQN stability
        buffer_size=100000,     # Large buffer for complex network states
        learning_starts=2000,   # Allow the agent to explore before learning
        batch_size=128,         # Larger batch for Mac Pro memory
        gamma=0.99,             # Future reward discount
        train_freq=4, 
        gradient_steps=1, 
        target_update_interval=1000, # Delay target update (Key for DDQN)
        exploration_fraction=0.2, 
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256]), # Deeper network for 78 features
        device="cpu"            # CPU is usually faster for small vectors
    )

    # Train
    model.learn(total_timesteps=TIMESTEPS)
    
    # Save the 'Brain'
    model_save_path = f"models/{exp_name}"
    model.save(model_save_path)
    print(f"[PROCESS] Saved DDQN model to {model_save_path}")

if __name__ == "__main__":
    # Utilize the Mac Pro's cores to run the 4 experiments in parallel
    processes = []
    for rule, know in EXPERIMENTS:
        p = multiprocessing.Process(target=run_ddqn_training, args=(rule, know))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    print("\n[FINISH] All Double DQN Thesis Experiments Complete.")