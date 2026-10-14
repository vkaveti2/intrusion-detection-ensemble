"""
plot_comparison.py — Original vs adversarially trained detectors.

Reads eval_results.json (v2, original detectors) and
results_v3_adversarial/eval_results.json (v3, adversarially trained) and
plots evasion per defence for: random probes (no learning), the trained
black-box agent and the trained white-box agent.

Output: thesis_defence_comparison.png
"""

import json

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 11,
                     "axes.edgecolor": "#c3c2b7", "grid.color": "#e6e5e0"})

DEFENCES = [("xgboost", "XGBoost"), ("rf", "RF"), ("cnn", "CNN"), ("unanimous", "Unanimous"),
            ("majority", "Majority"), ("weighted", "Weighted"), ("any", "Any")]
SETS = [("eval_results.json", "Original detectors", "#7a7974"),
        ("results_v3_adversarial/eval_results.json", "Adversarially trained detectors", "#1baf7a")]
PANELS = [("untrained", None, "Random probes (no learning)"),
          ("trained", "black", "Trained black-box agent"),
          ("trained", "white", "Trained white-box agent")]
INK = "#3a3a38"

results = []
for path, _, _ in SETS:
    with open(path) as f:
        results.append(json.load(f))

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
xs = np.arange(len(DEFENCES))
width = 0.38
for ax, (kind, knowledge, title) in zip(axes, PANELS):
    for i, ((_, label, color), res) in enumerate(zip(SETS, results)):
        if kind == "untrained":
            vals = [res["untrained"][r]["evasion"] for r, _ in DEFENCES]
        else:
            vals = [res["trained"].get(f"{r}_{knowledge}", {}).get("evasion", np.nan) for r, _ in DEFENCES]
        # white edge = surface gap between adjacent bars
        pos = xs + (i - 0.5) * width
        ax.bar(pos, np.array(vals) * 100, width, color=color,
               edgecolor="white", linewidth=2, label=label)
        # A near-0% bar has no visible height; label it so it doesn't read as missing data
        for x, v in zip(pos, vals):
            if v < 0.01:
                ax.text(x, 1.5, f"{v * 100:.2g}%", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_title(title, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([t for _, t in DEFENCES], rotation=30, ha="right")
    ax.set_ylim(0, 105)
axes[0].set_ylabel("Evasion rate (%)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Evasion against original vs adversarially trained detectors", color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("thesis_defence_comparison.png", dpi=300)
print("Saved thesis_defence_comparison.png")
