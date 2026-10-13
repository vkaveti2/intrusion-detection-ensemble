"""
ids_env.py — Gymnasium environment for adversarial RL against live IDS

This environment represents the MDP (State, Action, Reward). 
It supports both 'Hardware-in-the-loop' (sending Scapy packets to a Pi)
and 'Local Simulation' (training against the ensemble.py library).
"""

import gymnasium as gym
import numpy as np
import socket
from scapy.all import IP, TCP, Raw, sr1, conf as scapy_conf
from scapy.volatile import RandShort
import time
import json
from pathlib import Path

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURE_DIM    = 77  # CIC-IDS2017 features after removing the Label column
EPISODE_LENGTH = 100 
REWARD_EVADE   = +1.0
REWARD_DETECT  = -1.0
REWARD_OTHER   =  0.0
REWARD_INVALID = -0.5
MAX_DELAY      = 100.0
PENALTY_WEIGHT = 0.1

# Invariant thresholds for research validity
MIN_UNIQUE_PORTS = 2
MIN_FAILED_RATIO = 0.01
MIN_PORT_RANGE   = 1

# Pi communication defaults
DEFAULT_PI_IP       = "172.16.105.41"
DEFAULT_REWARD_PORT = 9999
SOCKET_TIMEOUT      = 120.0

# ── Action Space ──────────────────────────────────────────────────────────────
DELAY_OPTIONS   = list(np.logspace(np.log10(0.1), np.log10(100.0), num=50))
TCPFLAG_OPTIONS = ["S", "F", "N", "FPU", "SA"]
PKTSIZE_OPTIONS = [0, 64, 128, 256, 512]
N_DELAY, N_FLAGS, N_PKTSIZE = len(DELAY_OPTIONS), len(TCPFLAG_OPTIONS), len(PKTSIZE_OPTIONS)
N_ACTIONS = N_DELAY * N_FLAGS * N_PKTSIZE

def decode_action(action_idx):
    pkt_idx   = action_idx % N_PKTSIZE
    flag_idx  = (action_idx // N_PKTSIZE) % N_FLAGS
    delay_idx = action_idx // (N_PKTSIZE * N_FLAGS)
    return DELAY_OPTIONS[delay_idx], TCPFLAG_OPTIONS[flag_idx], PKTSIZE_OPTIONS[pkt_idx]

# ── Environment ────────────────────────────────────────────────────────────────
class IDSEvasionEnv(gym.Env):
    def __init__(self, pi_ip=DEFAULT_PI_IP, pi_port=DEFAULT_REWARD_PORT, 
                 knowledge="white", ensemble_rule="majority", use_pi=False):
        super().__init__()

        self.pi_ip = pi_ip
        self.pi_port = pi_port
        self.use_pi = use_pi # Toggle between Hardware and Simulation
        self.knowledge = knowledge
        self.feature_dim = FEATURE_DIM
        self.iface = "en0"

        # Initialize the Defense Engine
        from ensemble import EnsembleIDS
        self.ensemble = EnsembleIDS(voting=ensemble_rule)

        # Gym Spaces
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(FEATURE_DIM,), dtype=np.float32
        )

        # State Variables
        self._step_count = 0
        self._ports_probed = []
        self._port_results = {}
        self._socket = None
        self._last_features = np.zeros(FEATURE_DIM, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._ports_probed = []
        self._port_results = {}
        
        if self.use_pi:
            self._connect_socket()
            
        return np.zeros(self.feature_dim, dtype=np.float32), {"step": 0}

    def step(self, action):
        self._step_count += 1
        delay, tcp_flags, pkt_size = decode_action(int(action))

        # Port Logic
        port = 1000 + self._step_count
        self._ports_probed.append(port)

        # Invariant Check (Thesis Rigor)
        violation = self._check_invariants()
        if violation:
            return np.zeros(self.feature_dim), REWARD_INVALID, True, False, {"violation": violation}

        # The "Probe" (Real Scapy vs Simulation)
        if self.use_pi:
            classification, features = self._send_real_probe(port, tcp_flags, delay, pkt_size)
        else:
            classification, features = self._simulate_probe(
                port, delay, tcp_flags, pkt_size
            )

        reward = self._compute_reward(classification, delay)
        self._last_features = features
        obs = self._get_observation(features)

        terminated = self._step_count >= EPISODE_LENGTH
        return obs, reward, terminated, False, {"classification": classification}

    def _simulate_probe(self, port, delay, tcp_flags, pkt_size):
        """Simulates traffic for the Mac Pro to train the agent quickly."""
        # Match the feature order used by the saved CIC-IDS2017 scaler.
        f = np.zeros(self.feature_dim, dtype=np.float32)
        f[0] = port
        f[1] = delay * 1_000_000
        f[6] = pkt_size
        f[14] = pkt_size / (delay + 1e-6)
        if "S" in tcp_flags:
            f[44] = 1
        if "F" in tcp_flags:
            f[43] = 1
        if "P" in tcp_flags:
            f[46] = 1
        if "A" in tcp_flags:
            f[47] = 1
        if "U" in tcp_flags:
            f[48] = 1
        
        # Real Inference from ensemble.py
        result = self.ensemble.classify(f)
        return result["label"], f

    def _send_real_probe(self, port, tcp_flags, delay, pkt_size):
        """Actual Scapy Packet Logic for the Pi Hardware Benchmarks."""
        scapy_conf.verb = 0
        pkt = IP(dst=self.pi_ip)/TCP(dport=port, sport=RandShort(), flags=tcp_flags)
        if pkt_size > 0: pkt = pkt / Raw(load=b"A"*pkt_size)
        
        try:
            # We send the packet but don't wait for response in RL training
            # unless we need the Pi's specific detector result.
            sr1(pkt, timeout=1, verbose=0, iface=self.iface)
        except: pass

        time.sleep(delay)
        return self._recv_pi_classification()

    def _check_invariants(self):
        """Original scientific constraints on the attack behavior."""
        if len(self._ports_probed) < MIN_UNIQUE_PORTS: return None
        if (max(self._ports_probed) - min(self._ports_probed)) < MIN_PORT_RANGE:
            return "port_range_too_small"
        return None

    def _compute_reward(self, classification, delay):
        if classification == "normal":
            return REWARD_EVADE - (delay / MAX_DELAY) * PENALTY_WEIGHT
        if classification == "scanning":
            return REWARD_DETECT
        return REWARD_OTHER

    def _get_observation(self, features):
        if self.knowledge == "black":
            return np.zeros(self.feature_dim, dtype=np.float32)
        if self.knowledge == "white":
            return np.asarray(features, dtype=np.float32)
        # Gray: Timing + Port + Size only
        gray = np.zeros(self.feature_dim, dtype=np.float32)
        gray[[0, 1, 6, 14]] = features[[0, 1, 6, 14]]
        return gray

    def _connect_socket(self):
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(SOCKET_TIMEOUT)
            self._socket.connect((self.pi_ip, self.pi_port))
        except: self._socket = None

    def _recv_pi_classification(self):
        """Communicates with the Pi's reward socket."""
        if not self._socket:
            return "unknown", np.zeros(self.feature_dim, dtype=np.float32)
        try:
            data = self._socket.recv(4096).decode().strip()
            msg = json.loads(data)
            features = np.asarray(msg["features"], dtype=np.float32)
            if features.shape != (self.feature_dim,):
                raise ValueError(
                    f"Pi returned {features.size} features; expected {self.feature_dim}"
                )
            return msg["classification"], features
        except: return "unknown", np.zeros(self.feature_dim)

    def close(self):
        if self._socket: self._socket.close()