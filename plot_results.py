"""
plot_results.py — Generates publication-ready figures for thesis analysis.

Run from the directory train_ensemble.py and evaluate_agents.py were run from.

Produces:
  - thesis_convergence_results.png:
      One panel per defence (standalone detectors, then ensemble rules):
      smoothed per-episode evasion rate for reward-only and feature-observing agents,
      with the untrained (random-probe) baseline as a dashed reference.
  - thesis_final_evasion.png:
      Untrained baseline vs greedy trained-agent evasion per defence
      (from eval_results.json).
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "code"))
from terminology import capital

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "axes.edgecolor": "#c3c2b7",
    "grid.color": "#e6e5e0",
})

DEFENCES = [
    ("xgboost",   "XGBoost (standalone)"),
    ("rf",        "Random Forest (standalone)"),
    ("cnn",       "1D-CNN (standalone)"),
    ("unanimous", "Unanimous (all 3 must flag)"),
    ("majority",  "Majority (2 of 3)"),
    ("weighted",  "Confidence-weighted"),
    ("any",       "Any (1 of 3 flags)"),
]
KNOWLEDGE = [("black", capital("black"), "#2a78d6"), ("white", capital("white"), "#eb6834")]
BASELINE_COLOR = "#7a7974"
INK = "#3a3a38"
WINDOW = 15  # rolling window (episodes) for smoothing

evals = {}
if os.path.exists("eval_results.json"):
    with open("eval_results.json") as f:
        evals = json.load(f)

# ── Figure 1: convergence small multiples ─────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True)
axes = axes.ravel()
for ax, (rule, title) in zip(axes, DEFENCES):
    for knowledge, label, color in KNOWLEDGE:
        path = f"convergence_ddqn_{rule}_{knowledge}.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        eps = sorted((int(k), v[1] * 100.0) for k, v in data.items())
        x = [e for e, _ in eps]
        y = pd.Series([v for _, v in eps]).rolling(WINDOW, min_periods=1).mean()
        ax.plot(x, y, color=color, linewidth=2.0, label=label)
    base = evals.get("untrained", {}).get(rule, {}).get("evasion")
    if base is not None:
        ax.axhline(base * 100, color=BASELINE_COLOR, linestyle="--", linewidth=1.2,
                   label="Untrained (random probes)")
    ax.set_title(title, color=INK)
    ax.set_ylim(0, 103)

axes[-1].axis("off")
handles, labels = axes[0].get_legend_handles_labels()
axes[-1].legend(handles, labels, loc="center", frameon=False)
for ax in axes[4:7]:
    ax.set_xlabel("Episode (100 probes each)")
for ax in (axes[0], axes[4]):
    ax.set_ylabel("Evasion rate (%)")
fig.suptitle("DDQN evasion rate during training, by defence", color=INK)
fig.tight_layout()
fig.savefig("thesis_convergence_results.png", dpi=300)
print("Saved thesis_convergence_results.png")

# ── Figure 2: final greedy evasion vs untrained baseline ──────────────────────
if evals:
    fig2, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(DEFENCES))
    width = 0.26
    series = [("Untrained (random probes)", BASELINE_COLOR,
               [evals["untrained"].get(r, {}).get("evasion", np.nan) for r, _ in DEFENCES])]
    for knowledge, label, color in KNOWLEDGE:
        series.append((f"Trained, {label.lower()}", color,
                       [evals["trained"].get(f"{r}_{knowledge}", {}).get("evasion", np.nan)
                        for r, _ in DEFENCES]))
    for i, (label, color, vals) in enumerate(series):
        # white edge = surface gap between adjacent bars
        ax.bar(xs + (i - 1) * width, np.array(vals) * 100, width, color=color,
               edgecolor="white", linewidth=2, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace(" (", "\n(") for _, t in DEFENCES])
    ax.set_ylabel("Evasion rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Evasion before vs after DDQN training (greedy policy, held-out seeds)", color=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig2.tight_layout()
    fig2.savefig("thesis_final_evasion.png", dpi=300)
    print("Saved thesis_final_evasion.png")
