"""
flow_features.py — CICFlowMeter feature arithmetic for single-probe flows.

Shared by the simulator (ids_env._probe_features) and adversarial training
(train_adversarial.py) so both apply exactly the same transformation.

A single-probe flow has 1 forward packet and 0 or 1 backward packet (the
target's reply). Setting the forward payload to `s` bytes changes only the
length/byte features below; every formula was validated against 116,998 real
CIC-IDS2017 single-packet flows (100% match), including CICFlowMeter quirks:
the first packet is counted twice in the flow length stats, Average Packet
Size = (2*fwd + bwd) / packets, and act_data_pkt_fwd is always 0.
"""

import numpy as np

# CICFlowMeter feature indices (CIC-IDS2017 MachineLearningCVE column order)
F_DST_PORT, F_FLOW_DURATION = 0, 1
F_TOT_FWD_PKTS, F_TOT_BWD_PKTS, F_TOT_LEN_FWD, F_TOT_LEN_BWD = 2, 3, 4, 5
F_FWD_LEN_MAX, F_FWD_LEN_MIN, F_FWD_LEN_MEAN, F_FWD_LEN_STD = 6, 7, 8, 9
F_FLOW_BYTES_S = 14
F_MIN_PKT_LEN, F_MAX_PKT_LEN, F_PKT_LEN_MEAN, F_PKT_LEN_STD, F_PKT_LEN_VAR = 38, 39, 40, 41, 42
F_AVG_PKT_SIZE, F_AVG_FWD_SEG = 52, 53
F_SUBFLOW_FWD_BYTES = 62
F_ACT_DATA_PKT_FWD = 67

MAX_PAYLOAD = 1460  # TCP MSS on a 1500-byte Ethernet MTU


def is_single_probe(flows):
    """Rows with 1 forward packet and at most 1 backward packet."""
    flows = np.atleast_2d(flows)
    return (flows[:, F_TOT_FWD_PKTS] == 1) & (flows[:, F_TOT_BWD_PKTS] <= 1)


def set_forward_payload(flows, payload):
    """
    Returns copies of single-probe `flows` (n, 77) with the forward packet's
    payload set to `payload` bytes (scalar or (n,)); dependent features recomputed.
    """
    f = np.array(np.atleast_2d(flows), dtype=np.float64, copy=True)
    s = np.broadcast_to(np.asarray(payload, dtype=np.float64), (len(f),))
    nb = f[:, F_TOT_BWD_PKTS]
    b = f[:, F_TOT_LEN_BWD]
    has_bwd = nb > 0

    for col in (F_TOT_LEN_FWD, F_FWD_LEN_MAX, F_FWD_LEN_MIN, F_FWD_LEN_MEAN,
                F_AVG_FWD_SEG, F_SUBFLOW_FWD_BYTES):
        f[:, col] = s
    f[:, F_FWD_LEN_STD] = 0.0
    f[:, F_ACT_DATA_PKT_FWD] = 0.0

    # Flow length stats over [s, s] (+ [b] if there is a reply), sample variance
    mean = np.where(has_bwd, (2 * s + b) / 3, s)
    var = np.where(has_bwd, (2 * (s - mean) ** 2 + (b - mean) ** 2) / 2, 0.0)
    f[:, F_MIN_PKT_LEN] = np.where(has_bwd, np.minimum(s, b), s)
    f[:, F_MAX_PKT_LEN] = np.where(has_bwd, np.maximum(s, b), s)
    f[:, F_PKT_LEN_MEAN] = mean
    f[:, F_PKT_LEN_VAR] = var
    # sqrt of the float32-rounded variance, matching the stored feature precision
    f[:, F_PKT_LEN_STD] = np.sqrt(var.astype(np.float32))
    f[:, F_AVG_PKT_SIZE] = (2 * s + b) / (1 + nb)

    duration_us = f[:, F_FLOW_DURATION]
    safe_duration = np.where(duration_us > 0, duration_us, 1.0)
    f[:, F_FLOW_BYTES_S] = np.where(
        duration_us > 0, (s + b) / (safe_duration / 1e6), f[:, F_FLOW_BYTES_S]
    )
    return f.astype(np.float32)
