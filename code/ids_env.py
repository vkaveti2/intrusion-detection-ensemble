"""
ids_env.py — Gymnasium environment for adversarial RL against live IDS

The agent crafts port scan parameters, sends real packets to the Pi via Scapy,
and receives a reward based on the IDS classification result.

MDP:
  S: 154-dimensional feature vector (IDS observation space)
  A: Scan parameters (inter-probe delay, TCP flags, packet size)
  R: +1 * time_penalty evaded, -1 detected, 0 other class, -0.5 invariant violation
  γ: 0.99

Usage:
    env = IDSEvasionEnv(pi_ip="172.16.105.41", model="xgboost", knowledge="black")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
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

FEATURE_DIM    = 154
EPISODE_LENGTH = 100  # probes per episode (matches default target_ports range(1, 101))
REWARD_EVADE   = +1.0
REWARD_DETECT  = -1.0
REWARD_OTHER   =  0.0
REWARD_INVALID = -0.5
MAX_DELAY      = 100.0
PENALTY_WEIGHT = 0.1

# Invariant thresholds
MIN_UNIQUE_PORTS = 2
MIN_FAILED_RATIO = 0.01
MIN_PORT_RANGE   = 1

# Pi communication
DEFAULT_PI_IP       = "172.16.105.41"
DEFAULT_REWARD_PORT = 9999
SOCKET_TIMEOUT      = 120.0

# ── Action space (hybrid discrete) ────────────────────────────────────────────
#
# Double DQN requires discrete actions. Fine-grained grid gives the agent
# effective continuous control while staying compatible with DDQN.
#
#   delay     : 50 values log-spaced from 0.1s to 100.0s
#   tcp_flags : [S, F, N, FPU, SA] — SYN, FIN, NULL, Xmas, SYN-ACK  (5)
#   pkt_size  : [0, 64, 128, 256, 512] bytes payload                  (5)
#
# Total: 50 × 5 × 5 = 1250 discrete actions
#
# Priya: update DELAY_OPTIONS, TCPFLAG_OPTIONS, PKTSIZE_OPTIONS as needed

DELAY_OPTIONS   = list(np.logspace(np.log10(0.1), np.log10(100.0), num=50))
TCPFLAG_OPTIONS = ["S", "F", "N", "FPU", "SA"]
PKTSIZE_OPTIONS = [0, 64, 128, 256, 512]

N_DELAY   = len(DELAY_OPTIONS)
N_FLAGS   = len(TCPFLAG_OPTIONS)
N_PKTSIZE = len(PKTSIZE_OPTIONS)
N_ACTIONS = N_DELAY * N_FLAGS * N_PKTSIZE  # 1250

def decode_action(action_idx):
    """Decode flat action index into (delay, tcp_flags, pkt_size)."""
    pkt_idx   = action_idx % N_PKTSIZE
    flag_idx  = (action_idx // N_PKTSIZE) % N_FLAGS
    delay_idx = action_idx // (N_PKTSIZE * N_FLAGS)
    return (
        DELAY_OPTIONS[delay_idx],
        TCPFLAG_OPTIONS[flag_idx],
        PKTSIZE_OPTIONS[pkt_idx],
    )


# ── Environment ────────────────────────────────────────────────────────────────

class IDSEvasionEnv(gym.Env):
    """
    Adversarial RL environment against a live IDS on Raspberry Pi.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        pi_ip        = DEFAULT_PI_IP,
        pi_port      = DEFAULT_REWARD_PORT,
        model        = None,
        target_ports = range(1, 101),
        feature_dim  = FEATURE_DIM,
        knowledge    = None,
        ensemble     = None,
    ):
        assert model in ("xgboost", "cnn", "mlp", "ensemble"), \
            "model must be one of: xgboost, cnn, mlp, ensemble"
        assert knowledge in ("black", "gray", "white"), \
            "knowledge must be one of: black, gray, white"
        super().__init__()

        self.pi_ip        = pi_ip
        self.pi_port      = pi_port
        self.model        = model
        self.target_ports = list(target_ports)
        self.feature_dim  = feature_dim
        self.knowledge    = knowledge
        self.iface        = "en0"  # network interface for Scapy

        # Ensemble mode: classify locally instead of querying the Pi.
        # Pass an EnsembleIDS instance, a voting-rule string, or True
        # (defaults to majority voting).
        if ensemble is True:
            ensemble = "majority"
        if isinstance(ensemble, str):
            from ensemble import EnsembleIDS
            ensemble = EnsembleIDS(voting=ensemble)
        self.ensemble = ensemble
        self._last_probe = None  # (delay, tcp_flags, pkt_size) for feature build
        self.np_random = np.random.default_rng()

        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        self.observation_space = gym.spaces.Box(
            low   = -np.inf,
            high  =  np.inf,
            shape = (feature_dim,),
            dtype = np.float32
        )

        self._step_count   = 0
        self._ports_probed = []
        self._port_results = {}
        self._current_obs  = np.zeros(feature_dim, dtype=np.float32)
        self._socket       = None

        self._last_features = np.zeros(feature_dim, dtype=np.float32)

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count   = 0
        self._ports_probed = []
        self._port_results = {}
        self._current_obs  = np.zeros(self.feature_dim, dtype=np.float32)

        self._port_queue = self.target_ports.copy()
        if seed is not None:
            rng = np.random.default_rng(seed)
            rng.shuffle(self._port_queue)
        else:
            np.random.shuffle(self._port_queue)

        self._connect_socket()

        info = {
            "step":         0,
            "ports_probed": 0,
            "knowledge":    self.knowledge,
            "model":        self.model,
        }

        return self._current_obs.copy(), info

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1
        delay, tcp_flags, pkt_size = decode_action(int(action))

        if self._port_queue:
            port = self._port_queue.pop(0)
        else:
            self._port_queue = self.target_ports.copy()
            np.random.shuffle(self._port_queue)
            port = self._port_queue.pop(0)

        self._ports_probed.append(port)
        print(f"[IDSEnv] step={self._step_count} port={port} delay={delay:.2f}s flags={tcp_flags} pkt={pkt_size}B")

        # Check invariants — Priya
        violation = self._check_invariants()
        if violation:
            return (
                self._current_obs.copy(),
                REWARD_INVALID,
                False,
                False,
                {"violation": violation, "port": port, "action": action}
            )

        classification,engineered_features = self._send_probe(port, tcp_flags, delay, pkt_size)
        reward         = self._compute_reward(classification, delay)
        new_obs        = self._get_observation()
        self._current_obs = new_obs

        terminated = len(set(self._ports_probed)) >= len(self.target_ports)
        truncated  = False

        info = {
            "step":           self._step_count,
            "port":           port,
            "delay":          delay,
            "tcp_flags":      tcp_flags,
            "pkt_size":       pkt_size,
            "classification": classification,
            "features": engineered_features,
            "reward":         reward,
            "ports_probed":   len(self._ports_probed),
            "port_result":    self._port_results.get(port, "unknown"),
        }

        return new_obs.copy(), reward, terminated, truncated, info

    # ── Reward — Priya ─────────────────────────────────────────────────────────

    def _compute_reward(self, classification, delay):
        if classification == "normal":
            return REWARD_EVADE - (delay / MAX_DELAY) * PENALTY_WEIGHT
        elif classification == "scanning":
            return REWARD_DETECT
        else:
            return REWARD_OTHER

    # ── Invariant checker — Priya (bug fixed) ──────────────────────────────────

    def _check_invariants(self):
        """
        Ensure traffic remains a valid port scan.
        Returns violation string or None if valid.

        Fix: define failed=0 before conditional block to avoid NameError
        when port_results is empty, and use elif chain correctly.
        """
        failed = 0
        if len(self._port_results) > 0:
            values = list(self._port_results.values())
            failed = values.count("closed") + values.count("filtered")

        if len(self._port_results) < MIN_UNIQUE_PORTS:
            return None  # not enough data yet — don't penalize early probes

        if max(self._port_results.keys()) - min(self._port_results.keys()) < MIN_PORT_RANGE:
            return "port_range_too_small"

        if len(self._port_results) > 0 and failed / len(self._port_results) < MIN_FAILED_RATIO:
            return "failed_ratio_too_small"

        return None

    # ── Probe sender ───────────────────────────────────────────────────────────

    def _send_probe(self, port, tcp_flags, delay, pkt_size):
        """
        Craft and send a raw TCP packet to the Pi using Scapy.
        In ensemble (local) mode, skip packet I/O entirely — the probe
        parameters alone drive feature construction and classification.
        """
        if self.ensemble is not None:
            # Mock port state: a few well-known ports open, rest closed.
            # Keeps the invariant checker (failed-ratio, port-range)
            # exercised with realistic scan results.
            self._port_results[port] = "open" if port in (22, 80, 443) else "closed"
            self._last_probe = (delay, tcp_flags, pkt_size)
            classification, features = self._recv_classification()
            return classification, features

        scapy_conf.verb = 0

        pkt = IP(dst=self.pi_ip) / TCP(
            dport=port,
            sport=RandShort(),
            flags=tcp_flags,
        )
        if pkt_size > 0:
            pkt = pkt / Raw(load=b"A" * pkt_size)

        try:
            response = sr1(pkt, timeout=2, verbose=0, iface=self.iface)
        except Exception as e:
            print(f"[IDSEnv] Scapy error: {e}")
            return "error",None

        if response and response.haslayer(TCP):
            flags = response[TCP].flags
            if flags & 0x12:
                self._port_results[port] = "open"
                rst = IP(dst=self.pi_ip) / TCP(
                    dport=port, sport=response[TCP].dport,
                    flags="R", seq=response[TCP].ack
                )
                sr1(rst, timeout=1, verbose=0, iface=self.iface)
            elif flags & 0x04:
                self._port_results[port] = "closed"
            else:
                self._port_results[port] = "unknown"
        else:
            self._port_results[port] = "filtered"

        # In ensemble (local) mode the delay is encoded in the feature
        # vector and penalized in the reward — no need to burn wall-clock
        # time. Only sleep when probing the real Pi.
        if self.ensemble is None:
            time.sleep(delay)

        self._last_probe = (delay, tcp_flags, pkt_size)
        classification,engineered_features = self._recv_classification()
        return classification,engineered_features

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_observation(self):
        """
        Knowledge conditions:
          black: zeros — agent infers state from reward signal alone
          gray:  TODO — top-k features from XGBoost importance via Pi socket
          white: TODO — full 154-feature vector from Pi socket
        """
        if self.knowledge == "black":
            return np.zeros(self.feature_dim, dtype=np.float32)

        elif self.knowledge == "white":
            return self._last_features.copy()
        
        elif self.knowledge == "gray":
            #take top k important features from feature_columns.csv (here choosing k=20)
            GRAY_BOX_FEATURE_INDICES = [2, 43, 49, 55, 63, 82, 83, 85, 86, 94, 95, 97, 98, 101, 29, 36, 107, 126, 118, 133]
            masked = np.zeros(self.feature_dim, dtype=np.float32)
            masked[GRAY_BOX_FEATURE_INDICES] = self._last_features[GRAY_BOX_FEATURE_INDICES]
            return masked

    # ── Socket communication ───────────────────────────────────────────────────

    def _connect_socket(self):
        try:
            if self._socket:
                self._socket.close()
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(SOCKET_TIMEOUT)
            self._socket.connect((self.pi_ip, self.pi_port))
        except Exception as e:
            print(f"[IDSEnv] Warning: could not connect to Pi socket: {e}")
            print(f"[IDSEnv] Running in offline mode — classifications will be 'unknown'")
            self._socket = None

    def _build_local_features(self, delay, tcp_flags, pkt_size):
        """
        Build a 154-dim feature vector from probe parameters for
        ensemble (local) mode. Interpretable layout shared with the
        mock detectors in ensemble.py:

          idx 0 : scan aggressiveness = 1 - log-normalized delay [0,1]
                  (fast probes look like scanning)
          idx 1 : SYN flag indicator
          idx 2 : stealth flag indicator (FIN/NULL/Xmas)
          idx 3 : packet size normalized [0,1]
          idx 4 : ports probed so far (fraction of target set)
          idx 5 : unique-port ratio
          idx 6-9   : jitter
          idx 10-29 : flag one-hot band + size (legacy compat)
          idx 30+   : background noise
        """
        f = np.zeros(self.feature_dim, dtype=np.float32)
        norm_delay = (np.log10(max(delay, 0.1)) - np.log10(0.1)) / 2.0  # [0,1]
        f[0] = np.float32(1.0 - np.clip(norm_delay, 0, 1))
        f[1] = np.float32(1.0 if tcp_flags == "S" else 0.0)
        f[2] = np.float32(1.0 if tcp_flags in ("F", "N", "FPU") else 0.0)
        f[3] = np.float32(pkt_size / 512.0)
        f[4] = np.float32(len(self._ports_probed) / max(len(self.target_ports), 1))
        f[5] = np.float32(len(set(self._ports_probed)) / max(len(self._ports_probed), 1))
        f[6:10] = self.np_random.normal(0, 0.05, 4).astype(np.float32)
        # legacy band for compat
        flag_idx = TCPFLAG_OPTIONS.index(tcp_flags) if tcp_flags in TCPFLAG_OPTIONS else 0
        f[10 + flag_idx] = 1.0
        f[20] = f[3]
        f[30] = f[4]
        f[31] = f[5]
        f[32:] = self.np_random.normal(0, 0.1, self.feature_dim - 32).astype(np.float32)
        return f

    def _recv_classification(self):
        """
        Receive classification from Pi reward socket, or from the local
        ensemble when ensemble mode is enabled.
        Expected JSON: {"classification": "scanning", "confidence": 0.98, "features": np.array of engineered features}
        Reads one byte at a time until newline to avoid consuming multiple messages.
        """
        if self.ensemble is not None:
            delay, tcp_flags, pkt_size = self._last_probe or (1.0, "S", 0)
            features = self._build_local_features(delay, tcp_flags, pkt_size)
            result = self.ensemble.classify(features)
            classification = result["label"]
            self._last_features = features
            print(f"[IDSEnv] ensemble({result['rule']}) ← {classification} "
                  f"votes={result['votes']}")
            return classification, features.tolist()

        if self._socket is None:
            return "unknown",None

        try:
            self._socket.settimeout(SOCKET_TIMEOUT)
            data = b""
            while True:
                chunk = self._socket.recv(1)
                if not chunk or chunk == b"\n":
                    break
                data += chunk
            msg = json.loads(data.decode().strip())
            classification = msg.get("classification", "unknown")
            features = msg.get("features")
            self._last_features = np.array(features, dtype=np.float32) if features is not None else np.zeros(self.feature_dim, dtype=np.float32)
            print(f"[IDSEnv] reward ← {classification}")
            return classification,features
        except Exception as e:
            print(f"[IDSEnv] Socket error: {e} — reconnecting")
            self._connect_socket()
            return "unknown",None

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def close(self):
        if self._socket:
            self._socket.close()
            self._socket = None

    def render(self):
        pass


# ── Smoke test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("IDSEvasionEnv smoke test (offline mode)")
    env = IDSEvasionEnv(pi_ip="172.16.105.41", model="xgboost", knowledge="black")
    obs, info = env.reset(seed=42)
    print(f"Reset OK — obs shape: {obs.shape}  info: {info}")

    for step in range(5):
        action = env.action_space.sample()
        delay, flags, pkt_size = decode_action(int(action))
        print(f"Step {step+1}: action={action} (delay={delay:.2f}s flags={flags} pkt={pkt_size})", end=" ")
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"→ reward={reward:.3f}  class={info.get('classification')}  port={info.get('port')}")
        if terminated or truncated:
            break

    env.close()
    print("Smoke test complete.")