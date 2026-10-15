"""
summarize_results.py — One comparison across the coverage ladder (cov0–cov3).

cov k = number of detectors hardened against payload padding in each payload
band (docs/03_diversity_experiment_design.md). For every detector set reads:
  code/<set>/detector_eval.json, rule_eval.json      (evaluate_detectors.py)
  <results>/eval_results.json                        (evaluate_agents.py)
  <results>/convergence_ddqn_{rule}_{knowledge}.json (train_ensemble.py)

Writes:
  docs/results_summary.csv               one row per (set, defence)
  docs/results_summary.md                markdown tables of the same data
  docs/figures/coverage_ladder_evasion.png
  docs/figures/coverage_matrix.png
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
SETS = [  # key, label, detector dir, results dir, designed bands per detector
    ("cov0", "cov0: original (v2)", "code/models", ".",
     {"xgboost": "none", "rf": "none", "cnn": "none"}),
    ("cov1", "cov1: disjoint bands (v4)", "code/models_cov1", "results_v4_cov1",
     {"xgboost": "A", "rf": "B", "cnn": "C"}),
    ("cov2", "cov2: overlapping bands (v4)", "code/models_cov2", "results_v4_cov2",
     {"xgboost": "A+B", "rf": "B+C", "cnn": "C+A"}),
    ("cov3", "cov3: full hardening (v3)", "code/models_adv", "results_v3_adversarial",
     {"xgboost": "A+B+C", "rf": "A+B+C", "cnn": "A+B+C"}),
]
DEFENCES = [("xgboost", "XGBoost"), ("rf", "RF"), ("cnn", "CNN"), ("unanimous", "Unanimous"),
            ("majority", "Majority"), ("weighted", "Weighted"), ("any", "Any")]
DETECTORS = [("xgboost", "XGBoost"), ("rf", "RF"), ("cnn", "CNN")]
SIZES = ["0B", "64B", "128B", "256B", "512B", "random 1-1460B"]
SIZE_LABELS = ["0 B\n–", "64 B\nA", "128 B\nB", "256 B\nB", "512 B\nC", "random\nA–C"]
ROLL = 15  # episodes in the rolling mean used for convergence thresholds

# Sequential blue ramp (dataviz reference palette, steps 100–700)
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUES)
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.titlesize": 11,
                     "axes.edgecolor": "#c3c2b7", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def load(path):
    path = os.path.join(ROOT, path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def convergence_stats(path):
    data = load(path)
    if not data:
        return {}
    ev = np.array([v[1] for _, v in sorted(data.items(), key=lambda kv: int(kv[0]))])
    roll = pd.Series(ev).rolling(ROLL, min_periods=ROLL).mean().to_numpy()

    def first(thr):
        hit = np.flatnonzero(roll >= thr)
        return int(hit[0]) + 1 if len(hit) else None

    return {"ep_to_50": first(0.5), "ep_to_90": first(0.9),
            "probes_detected_in_training": int(round(((1 - ev) * 100).sum())),
            "training_evasion_last50": float(ev[-50:].mean())}


def top_payload(choice):
    if not choice:
        return None
    size, share = max(choice.items(), key=lambda kv: kv[1])
    return f"{size} B ({share:.0%})"


def collect():
    rows = []
    for key, label, det_dir, res_dir, _ in SETS:
        rule_eval = load(f"{det_dir}/rule_eval.json") or {"rules": {}}
        evals = load(f"{res_dir}/eval_results.json") or {"untrained": {}, "trained": {}}
        for rule, name in DEFENCES:
            r = rule_eval["rules"].get(rule, {})
            pad = r.get("padded_probe_detected_sim_ports", {})
            row = {"set": key, "defence": name,
                   "recall": r.get("binary_recall"), "false_pos_rate": r.get("binary_false_positive_rate"),
                   "f1": r.get("binary_f1"), "false_alarms": r.get("false_positives"),
                   **{f"padded_caught_{s}": pad.get(s) for s in SIZES},
                   "untrained_evasion": evals["untrained"].get(rule, {}).get("evasion")}
            for k in ("black", "white"):
                t = evals["trained"].get(f"{rule}_{k}", {})
                row[f"{k}_evasion"] = t.get("evasion")
                row[f"{k}_payload"] = top_payload(t.get("payload_choice"))
                for stat, v in convergence_stats(f"{res_dir}/convergence_ddqn_{rule}_{k}.json").items():
                    row[f"{k}_{stat}"] = v
            rows.append(row)
    return pd.DataFrame(rows)


def pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


def cell_pct(v, unit="%"):
    """58.5% stays 58.5%, 100.0% becomes 100%."""
    return f"{v * 100:.1f}".removesuffix(".0") + unit


def matrix(df, column):
    return df.pivot(index="defence", columns="set", values=column).reindex(
        index=[n for _, n in DEFENCES], columns=[s[0] for s in SETS])


def markdown(df):
    set_cols = [s[0] for s in SETS]
    out = ["# Results summary across the coverage ladder", "",
           "Generated by `summarize_results.py`. cov k = detectors hardened per payload band "
           "(A 1–90 B, B 91–362 B, C 363–1460 B).", ""]
    for k, title in (("black", "black-box"), ("white", "white-box")):
        m = matrix(df, f"{k}_evasion")
        out += [f"## Trained {title} agent: evasion rate (greedy, 2,000 probes)", "",
                "| Defence | " + " | ".join(set_cols) + " |", "|---|" + "---|" * len(set_cols)]
        out += [f"| {d} | " + " | ".join(pct(m.loc[d, s]) for s in set_cols) + " |" for d in m.index]
        out.append("")
    m = matrix(df, "untrained_evasion")
    out += ["## Untrained baseline: evasion rate (5,000 random probes, paired)", "",
            "| Defence | " + " | ".join(set_cols) + " |", "|---|" + "---|" * len(set_cols)]
    out += [f"| {d} | " + " | ".join(pct(m.loc[d, s], 2) for s in set_cols) + " |" for d in m.index]
    out.append("")
    m = matrix(df, "false_pos_rate")
    out += ["## Cost on clean held-out traffic: false-positive rate", "",
            "| Defence | " + " | ".join(set_cols) + " |", "|---|" + "---|" * len(set_cols)]
    out += [f"| {d} | " + " | ".join(pct(m.loc[d, s], 3) for s in set_cols) + " |" for d in m.index]
    out.append("")
    out += ["## Padded probes caught, no agent (simulator ports 1001–1100)", ""]
    for key, label, *_ in SETS:
        sub = df[df["set"] == key].set_index("defence")
        out += [f"**{label}**", "", "| Defence | " + " | ".join(SIZES) + " |", "|---|" + "---|" * len(SIZES)]
        out += [f"| {d} | " + " | ".join(pct(sub.loc[d, f"padded_caught_{s}"]) for s in SIZES) + " |"
                for d in sub.index]
        out.append("")
    out += ["## Training dynamics (seed 0, 500 episodes)", "",
            f"Episode at which the {ROLL}-episode rolling mean of training evasion first reached "
            "50% / 90% (— = never), and probes detected during training (of 50,000).", "",
            "| Set | Defence | black: ep≥50% | black: ep≥90% | black: detected | black: payload | "
            "white: ep≥50% | white: ep≥90% | white: detected | white: payload |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{int(v):,}"
    for _, r in df.iterrows():
        out.append(f"| {r['set']} | {r['defence']} | {fmt(r.get('black_ep_to_50'))} | "
                   f"{fmt(r.get('black_ep_to_90'))} | {fmt(r.get('black_probes_detected_in_training'))} | "
                   f"{r.get('black_payload') or '—'} | {fmt(r.get('white_ep_to_50'))} | "
                   f"{fmt(r.get('white_ep_to_90'))} | {fmt(r.get('white_probes_detected_in_training'))} | "
                   f"{r.get('white_payload') or '—'} |")
    return "\n".join(out) + "\n"


def annotate(ax, values, fmt):
    for (i, j), v in np.ndenumerate(values):
        if np.isnan(v):
            ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color=INK_2)
            continue
        # ink by fill luminance so the label always clears contrast
        ax.text(j, i, fmt(v), ha="center", va="center", fontsize=8.5,
                color="white" if v > 0.55 else INK)


def heatmap(ax, values, row_labels, col_labels, title, fmt):
    ax.pcolormesh(np.arange(values.shape[1] + 1) - 0.5, np.arange(values.shape[0] + 1) - 0.5,
                  np.ma.masked_invalid(values), cmap=CMAP, vmin=0, vmax=1,
                  edgecolors=SURFACE, linewidth=2)  # 2px surface gap between cells
    ax.set_xlim(-0.5, values.shape[1] - 0.5)
    ax.set_ylim(values.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0, colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, color=INK, loc="left")
    annotate(ax, values, fmt)


def figure_ladder(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    cols = ["cov0\noriginal", "cov1\ndisjoint", "cov2\noverlap", "cov3\nfull"]
    for ax, (k, title) in zip(axes, (("black", "Trained black-box agent"), ("white", "Trained white-box agent"))):
        m = matrix(df, f"{k}_evasion").to_numpy(dtype=float)
        heatmap(ax, m, [n for _, n in DEFENCES], cols, title, cell_pct)
        ax.axhline(2.5, color=INK_2, linewidth=0.8)  # standalone above, ensemble rules below
    axes[0].set_ylabel("Defence", color=INK_2)
    fig.suptitle("Evasion rate of the trained DDQN agent (share of probes labelled benign) "
                 "by designed detector coverage", color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Columns: designed number of detectors hardened per payload band "
             "(measured coverage was higher, see coverage_matrix.png). "
             "Rows above the line are standalone detectors, rows below are ensemble rules.",
             color=INK_2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = os.path.join(DOCS, "figures", "coverage_ladder_evasion.png")
    fig.savefig(path, dpi=300)
    return path


def figure_coverage():
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    for ax, (key, label, det_dir, _, bands) in zip(axes.ravel(), SETS):
        d = load(f"{det_dir}/detector_eval.json") or {}
        m = np.array([[d.get(det, {}).get("padded_probe_detected_sim_ports", {}).get(s, np.nan)
                       for s in SIZES] for det, _ in DETECTORS], dtype=float)
        heatmap(ax, m, [f"{n} ({bands[det]})" for det, n in DETECTORS], SIZE_LABELS, label,
                lambda v: cell_pct(v, ""))
        ax.tick_params(axis="x", labelsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("Padded payload size and its band", color=INK_2)
    fig.suptitle("Measured coverage: % of padded held-out scan probes each detector catches\n"
                 "(simulator ports 1001–1100; designed hardening bands in brackets)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(DOCS, "figures", "coverage_matrix.png")
    fig.savefig(path, dpi=300)
    return path


if __name__ == "__main__":
    os.makedirs(os.path.join(DOCS, "figures"), exist_ok=True)
    df = collect()
    df.to_csv(os.path.join(DOCS, "results_summary.csv"), index=False)
    with open(os.path.join(DOCS, "results_summary.md"), "w") as f:
        f.write(markdown(df))
    print("Saved docs/results_summary.csv, docs/results_summary.md")
    print("Saved", figure_ladder(df))
    print("Saved", figure_coverage())
