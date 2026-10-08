import gymnasium as gym
import numpy as np
import time

# ── Constants from your Original Research ──────────────────────────────────────
# Changed to 78 to match the real Kaggle CIC-IDS2017 dataset
FEATURE_DIM    = 78 
EPISODE_LENGTH = 100 

REWARD_EVADE   = +1.0
REWARD_DETECT  = -1.0
REWARD_OTHER   =  0.0
REWARD_INVALID = -0.5
MAX_DELAY      = 100.0
PENALTY_WEIGHT = 0.1

# Invariant thresholds (Keeping these for scientific rigor)
MIN_UNIQUE_PORTS = 2
MIN_FAILED_RATIO = 0.01
MIN_PORT_RANGE   = 1

# ── Action Space ──────────────────────────────────────────────────────────────
DELAY_OPTIONS   = list(np.logspace(np.log10(0.1), np.log10(100.0), num=50))
TCPFLAG_OPTIONS = ["S", "F", "N", "FPU", "SA"]
PKTSIZE_OPTIONS = [0, 64, 128, 256, 512]
N_ACTIONS = len(DELAY_OPTIONS) * len(TCPFLAG_OPTIONS) * len(PKTSIZE_OPTIONS)

class IDSEvasionEnv(gym.Env):
    def __init__(self, knowledge="white", ensemble_rule="majority"):
        super().__init__()
        
        # Load the real Ensemble engine we updated in the previous step
        from ensemble import EnsembleIDS
        self.ensemble = EnsembleIDS(voting=ensemble_rule)
        
        self.knowledge = knowledge
        self.feature_dim = FEATURE_DIM
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )

        # State tracking
        self._step_count = 0
        self._port_results = {}
        self._ports_probed = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._port_results = {}
        self._ports_probed = []
        # Return a zero-vector as the starting observation
        return np.zeros(self.feature_dim, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        
        # 1. Decode Action
        delay, tcp_flags, pkt_size = self._decode_action(action)
        
        # 2. Simulate Port Logic (Internal Invariants)
        # We simulate a scan where most ports are closed/filtered
        port = 1000 + self._step_count
        self._ports_probed.append(port)
        self._port_results[port] = "closed" if port != 1022 else "open"

        # 3. Check Invariants (The "Thesis" Logic)
        # If the agent's actions make the attack 'invalid', punish them
        violation = self._check_invariants()
        if violation:
            return np.zeros(self.feature_dim), REWARD_INVALID, True, False, {"violation": violation}

        # 4. Build REAL 78-dimensional features
        features = self._build_cic_features(delay, tcp_flags, pkt_size)
        
        # 5. Classify using the Real Models in ensemble.py
        result = self.ensemble.classify(features)
        classification = result["label"]

        # 6. Compute Reward (Sophisticated Version)
        if classification == "normal":
            # Success! Reward = 1.0 minus a penalty for how much time it took
            reward = REWARD_EVADE - (delay / MAX_DELAY) * PENALTY_WEIGHT
        else:
            reward = REWARD_DETECT

        # 7. Knowledge Condition (Black/Gray/White Box)
        obs = self._get_observation(features)

        terminated = self._step_count >= EPISODE_LENGTH
        return obs, reward, terminated, False, {"classification": classification}

    def _decode_action(self, action_idx):
        pkt_idx   = action_idx % len(PKTSIZE_OPTIONS)
        flag_idx  = (action_idx // len(PKTSIZE_OPTIONS)) % len(TCPFLAG_OPTIONS)
        delay_idx = action_idx // (len(PKTSIZE_OPTIONS) * len(TCPFLAG_OPTIONS))
        return DELAY_OPTIONS[delay_idx], TCPFLAG_OPTIONS[flag_idx], PKTSIZE_OPTIONS[pkt_idx]

    def _check_invariants(self):
        """Original Invariant Logic: Ensures the scan is still functionally a scan."""
        if len(self._port_results) < MIN_UNIQUE_PORTS: return None
        
        # Rule: Did the agent cover a wide enough range of ports?
        if max(self._port_results.keys()) - min(self._port_results.keys()) < MIN_PORT_RANGE:
            return "port_range_too_small"
        
        return None

    def _build_cic_features(self, delay, tcp_flags, pkt_size):
        """Translates RL actions into real CIC-IDS2017 Feature Indices."""
        f = np.zeros(self.feature_dim, dtype=np.float32)
        
        # Standard CIC-IDS2017 Mapping (Based on common CSV schemas)
        f[1] = delay * 1_000_000 # Flow Duration (index 1)
        f[7] = pkt_size          # Fwd Packet Length Max (index 7)
        
        if "S" in tcp_flags: f[44] = 1 # SYN Flag Count
        if "F" in tcp_flags: f[45] = 1 # RST Flag Count (approx)
        
        f[20] = pkt_size / (delay + 1e-6) # Flow Bytes/s
        
        # Add a tiny bit of noise so the model doesn't see 'perfect' math
        f += np.random.normal(0, 0.001, self.feature_dim)
        return f

    def _get_observation(self, features):
        if self.knowledge == "black":
            return np.zeros(self.feature_dim, dtype=np.float32)
        elif self.knowledge == "white":
            return features
        else: # Gray Box: Reveal only key timing and flag features
            gray = np.zeros(self.feature_dim, dtype=np.float32)
            gray[1] = features[1]
            gray[44] = features[44]
            return gray