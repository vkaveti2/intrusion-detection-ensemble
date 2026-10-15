"""
summarize_multiseed.py — Step 3 aggregation (docs/05_multiseed_design.md).

Part A: every detector set × defence × knowledge level over agent seeds 0–4
        (seed 0 = the original runs; seeds 1–4 in results_multiseed/<set>/seed_<n>/).
Part B: cov1 retrained with model seeds 1–3 (original = 42), agent seed 0.

Writes:
  docs/multiseed_summary.md, docs/multiseed_runs.csv, docs/multiseed_model_seeds.csv
  docs/figures/multiseed_convergence_cov0.png, multiseed_convergence_cov1.png
  docs/figures/multiseed_learning_speed.png
  docs/figures/detector_seed_coverage.png
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from summarize_results import (DEFENCES, DETECTORS, DOCS, INK, INK_2, ROLL, SETS, SIZE_LABELS, SIZES,
                               cell_pct, convergence_stats, heatmap, load)

SEEDS = (0, 1, 2, 3, 4)
MODEL_SEEDS = (42, 1, 2, 3)
SEED0_DIR = {key: res for key, _, _, res, _ in SETS}
DET_DIR = {key: det for key, _, det, _, _ in SETS}
SET_LABEL = {key: label for key, label, *_ in SETS}
DEF_NAME = dict(DEFENCES)
KNOWLEDGE = (("black", "Black-box agent", "#2a78d6"), ("white", "White-box agent", "#eb6834"))
BASELINE = "#7a7974"
PAYLOAD_OPTIONS = ("0B", "64B", "128B", "256B", "512B")
NEVER = 501        # episodes-to-threshold for runs that never reached it (ranks last)
SHORTFALL = 0.05   # "found the best option" = final evasion within 5 points of it
COLLAPSE_PAIRS = (("unanimous", "xgboost"), ("majority", "rf"), ("weighted", "rf"), ("any", "cnn"))


def seed_dir(key, seed):
    return SEED0_DIR[key] if seed == 0 else f"results_multiseed/{key}/seed_{seed}"


def complete_seeds(key):
    """Seeds whose batch finished (evaluate_agents.py ran), so every curve is full length."""
    return [s for s in SEEDS if load(f"{seed_dir(key, s)}/eval_results.json")]


def model_seed_dirs(ms):
    if ms == 42:
        return "code/models_cov1", "results_v4_cov1"
    return f"code/models_cov1_ms{ms}", f"results_multiseed/cov1_ms{ms}"


def best_option(det_dir, rule):
    """Highest evasion a single payload option gets against this rule (no agent, simulator ports)."""
    r = (load(f"{det_dir}/rule_eval.json") or {}).get("rules", {}).get(rule)
    if not r:
        return np.nan
    return max(1 - r["padded_probe_detected_sim_ports"][s] for s in PAYLOAD_OPTIONS)


def curve(res_dir, rule, k):
    d = load(f"{res_dir}/convergence_ddqn_{rule}_{k}.json")
    if not d:
        return None
    return np.array([v[1] for _, v in sorted(d.items(), key=lambda kv: int(kv[0]))])


def top_payload(choice):
    return max(choice, key=choice.get) + " B"


# ── Data ──────────────────────────────────────────────────────────────────────
def collect_runs():
    rows = []
    for key in DET_DIR:
        best = {rule: best_option(DET_DIR[key], rule) for rule, _ in DEFENCES}
        for seed in SEEDS:
            res = seed_dir(key, seed)
            ev = load(f"{res}/eval_results.json")
            if not ev:
                continue
            for rule, name in DEFENCES:
                for k, *_ in KNOWLEDGE:
                    t = ev["trained"].get(f"{rule}_{k}")
                    if t is None:
                        continue
                    rows.append({"set": key, "defence": name, "rule": rule, "knowledge": k, "seed": seed,
                                 "final_evasion": t["evasion"], "payload": top_payload(t["payload_choice"]),
                                 "best_option": best[rule],
                                 **convergence_stats(f"{res}/convergence_ddqn_{rule}_{k}.json")})
    df = pd.DataFrame(rows)
    for col in ("ep_to_50", "ep_to_90"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def collect_model_seeds():
    rows = []
    for ms in MODEL_SEEDS:
        det, res = model_seed_dirs(ms)
        rule_eval, evals = load(f"{det}/rule_eval.json"), load(f"{res}/eval_results.json")
        if not (rule_eval and evals):
            continue
        for rule, name in DEFENCES:
            r = rule_eval["rules"][rule]
            row = {"model_seed": ms, "defence": name, "rule": rule,
                   "recall": r["binary_recall"], "false_pos_rate": r["binary_false_positive_rate"],
                   **{f"caught_{s}": r["padded_probe_detected_sim_ports"][s] for s in SIZES},
                   **{f"caught_recorded_ports_{s}": r["padded_probe_detected"][s] for s in SIZES}}
            for k, *_ in KNOWLEDGE:
                t = evals["trained"].get(f"{rule}_{k}", {})
                row[f"{k}_evasion"] = t.get("evasion", np.nan)
                row[f"{k}_payload"] = top_payload(t["payload_choice"]) if t else None
            rows.append(row)
    return pd.DataFrame(rows)


def evadable(df):
    """(set, rule) pairs where some payload option evades (best option >= 5%)."""
    sub = df[df["best_option"] >= SHORTFALL][["set", "rule"]].drop_duplicates()
    return [tuple(x) for x in sub.to_numpy()]


def select(df, key, rule, k):
    return df[(df["set"] == key) & (df["rule"] == rule) & (df["knowledge"] == k)].sort_values("seed")


# ── Statistics ────────────────────────────────────────────────────────────────
def mw_p(x, y):
    return float(mannwhitneyu(x, y, alternative="two-sided", method="exact").pvalue)


def describe(x, digits=0, pct=False):
    x = np.asarray(x, dtype=float)
    if pct:
        x = x * 100
    f = lambda v: f"{v:,.{digits}f}"
    unit = "%" if pct else ""
    sd = f" ± {f(x.std(ddof=1))}" if len(x) > 1 else ""
    return f"{f(x.mean())}{sd}{unit} [{f(x.min())}–{f(x.max())}]"


def comparison(df, a, b, metric):
    xa = select(df, *a)[metric].fillna(NEVER).to_numpy(dtype=float)
    xb = select(df, *b)[metric].fillna(NEVER).to_numpy(dtype=float)
    if len(xa) < 2 or len(xb) < 2:
        return None
    return {"a": a, "b": b, "metric": metric, "a_desc": describe(xa), "b_desc": describe(xb),
            "consistent": bool(xa.min() > xb.max() or xb.min() > xa.max()), "p": mw_p(xa, xb),
            "n": (len(xa), len(xb))}


def run_comparisons(df):
    out = []
    for metric in ("ep_to_90", "probes_detected_in_training"):            # C1, C2
        for other in (("cov0", "cnn", "black"), ("cov0", "rf", "black")):
            out.append(("C1" if metric == "ep_to_90" else "C2", comparison(df, ("cov1", "cnn", "black"), other, metric)))
    for key, rule in evadable(df):                                       # C3
        out.append(("C3", comparison(df, (key, rule, "white"), (key, rule, "black"),
                                     "probes_detected_in_training")))
    return [(c, r) for c, r in out if r]


def shortfalls(df):
    """C4: seeds whose final evasion is more than 5 points below the best available option."""
    rows = []
    for key, rule in evadable(df):
        for k, *_ in KNOWLEDGE:
            s = select(df, key, rule, k)
            short = s[s["final_evasion"] < s["best_option"] - SHORTFALL]
            rows.append({"set": key, "defence": DEF_NAME[rule], "knowledge": k, "seeds": len(s),
                         "short": len(short), "best": s["best_option"].iloc[0],
                         "short_detail": ", ".join(f"seed {r.seed}: {r.final_evasion:.1%} ({r.payload})"
                                                   for r in short.itertuples())})
    return pd.DataFrame(rows)


def collapse_check():
    rows = []
    for key in DET_DIR:
        for seed in complete_seeds(key):
            res = seed_dir(key, seed)
            for k, *_ in KNOWLEDGE:
                for a, b in COLLAPSE_PAIRS:
                    ca, cb = curve(res, a, k), curve(res, b, k)
                    if ca is None or cb is None:
                        continue
                    rows.append({"set": key, "seed": seed, "knowledge": k, "pair": f"{a} = {b}",
                                 "identical_episodes": int((ca == cb).sum()), "episodes": len(ca)})
    return pd.DataFrame(rows)


# ── Markdown ──────────────────────────────────────────────────────────────────
def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(map(str, r)) + " |" for r in rows] + [""]


def markdown(df, ms, comps, short, collapse):
    n_seeds = df.groupby(["set", "rule", "knowledge"])["seed"].nunique()
    out = ["# Step 3 — multi-seed summary", "",
           "Generated by `summarize_multiseed.py` (design: `05_multiseed_design.md`). "
           f"Agent seeds per configuration: {sorted(df['seed'].unique().tolist())}. "
           "Values: mean ± SD [min–max] over seeds.", ""]

    out += ["## A1. Final greedy evasion (2,000 probes per agent)", ""]
    for k, label, _ in KNOWLEDGE:
        rows = []
        for rule, name in DEFENCES:
            rows.append([name] + [describe(select(df, key, rule, k)["final_evasion"], 1, pct=True)
                                  if len(select(df, key, rule, k)) else "—" for key in DET_DIR])
        out += [f"**{label}**", ""] + table(["Defence"] + list(DET_DIR), rows)

    out += ["## A2. Did the agent find the best available payload option?", "",
            "Best option = highest evasion any single payload option achieves against the defence "
            "(no agent, simulator ports). Shortfall = final evasion more than 5 points below it.", ""]
    rows = [[r.set, r.defence, r.knowledge, f"{r.best:.1%}", f"{r.seeds - r.short}/{r.seeds}",
             r.short_detail or "—"] for r in short.itertuples()]
    out += table(["Set", "Defence", "Agent", "Best option", "Seeds that found it", "Shortfalls"], rows)

    out += ["## A3. Learning speed (configurations where evasion is possible)", "",
            f"Episodes until the {ROLL}-episode rolling mean of training evasion first reached 90% "
            f"(runs that never reached it counted as {NEVER}), and probes detected during training "
            "(of 50,000).", ""]
    rows = []
    for key, rule in evadable(df):
        for k, *_ in KNOWLEDGE:
            s = select(df, key, rule, k)
            reached = int(s["ep_to_90"].notna().sum())
            rows.append([key, DEF_NAME[rule], k, f"{reached}/{len(s)}", describe(s["ep_to_90"].fillna(NEVER)),
                         describe(s["probes_detected_in_training"])])
    out += table(["Set", "Defence", "Agent", "Reached 90%", "Episodes to 90%", "Probes detected in training"], rows)

    out += ["## A4. Pre-specified comparisons", "",
            "Consistent = the two groups' min–max ranges do not overlap. p = exact two-sided "
            "Mann–Whitney U (5 vs 5: smallest possible p = 0.008).", ""]
    rows = []
    for c, r in comps:
        fmt = lambda g: f"{g[0]} {DEF_NAME[g[1]]} {g[2]}"
        rows.append([c, r["metric"], fmt(r["a"]), r["a_desc"], fmt(r["b"]), r["b_desc"],
                     "yes" if r["consistent"] else "no", f"{r['p']:.3f}"])
    out += table(["#", "Measure", "Group A", "A", "Group B", "B", "Consistent", "p"], rows)

    out += ["## A5. Ensemble collapse in every seed", "",
            "Identical training curves for each rule and the member it collapsed onto in v2.", ""]
    if len(collapse):
        agg = collapse.groupby(["set", "knowledge", "pair"]).agg(
            seeds=("seed", "nunique"),
            fully_identical=("identical_episodes", lambda x: int((x == 500).sum())),
            min_identical=("identical_episodes", "min")).reset_index()
        rows = [[r.set, r.knowledge, r.pair, f"{r.fully_identical}/{r.seeds}", f"{r.min_identical}/500"]
                for r in agg.itertuples()]
        out += table(["Set", "Agent", "Pair", "Seeds with all 500 episodes identical", "Fewest identical"], rows)

    if len(ms):
        out += ["## B1. cov1 retrained with different model seeds: measured coverage", "",
                "Padded held-out probes caught at simulator ports (17,834 per cell). Model seed 42 = "
                "the original cov1 detectors.", ""]
        rows = []
        for (seed, rule), r in ms.set_index(["model_seed", "rule"]).iterrows():
            if rule in ("xgboost", "rf", "cnn", "unanimous"):
                rows.append([seed, DEF_NAME[rule]] + [f"{r[f'caught_{s}']:.1%}" for s in SIZES])
        out += table(["Model seed", "Defence", *SIZES], rows)
        out += ["## B2. cov1 model seeds: agent evasion (agent seed 0) and clean false-positive rate", ""]
        rows = [[r.model_seed, r.defence, f"{r.black_evasion:.1%} ({r.black_payload})",
                 f"{r.white_evasion:.1%} ({r.white_payload})", f"{r.false_pos_rate:.3%}"]
                for r in ms.itertuples()]
        out += table(["Model seed", "Defence", "Black-box evasion (payload)", "White-box evasion (payload)",
                      "False-positive rate"], rows)
    return "\n".join(out) + "\n"


# ── Figures ───────────────────────────────────────────────────────────────────
def figure_convergence(key):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True)
    axes = axes.ravel()
    untrained = (load(f"{SEED0_DIR[key]}/eval_results.json") or {}).get("untrained", {})
    for ax, (rule, name) in zip(axes, DEFENCES):
        for k, label, color in KNOWLEDGE:
            curves = [c for c in (curve(seed_dir(key, s), rule, k) for s in complete_seeds(key))
                      if c is not None]
            if not curves:
                continue
            rolled = np.array([pd.Series(c).rolling(ROLL, min_periods=1).mean() for c in curves]) * 100
            x = np.arange(1, rolled.shape[1] + 1)
            ax.fill_between(x, rolled.min(axis=0), rolled.max(axis=0), color=color, alpha=0.15, linewidth=0)
            ax.plot(x, rolled.mean(axis=0), color=color, linewidth=2.0,
                    label=f"{label} (mean of {len(curves)} seeds; band = min–max)")
        base = untrained.get(rule, {}).get("evasion")
        if base is not None:
            ax.axhline(base * 100, color=BASELINE, linestyle="--", linewidth=1.2, label="Untrained (random probes)")
        ax.set_title(name, color=INK)
        ax.set_ylim(0, 103)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", frameon=False, fontsize=9)
    for ax in axes[4:7]:
        ax.set_xlabel("Episode (100 probes each)")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("Evasion rate during training (%)")
    fig.suptitle(f"DDQN training evasion across agent seeds — {SET_LABEL[key]}", color=INK)
    fig.tight_layout()
    path = os.path.join(DOCS, "figures", f"multiseed_convergence_{key}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure_learning_speed(df):
    configs = evadable(df)
    if not configs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(configs) + 1.8), sharey=True)
    metrics = (("ep_to_90", "Episodes until training evasion reached 90%"),
               ("probes_detected_in_training", "Probes detected during training (thousands)"))
    for ax, (metric, xlabel) in zip(axes, metrics):
        for i, (key, rule) in enumerate(configs):
            for offset, (k, label, color) in zip((-0.17, 0.17), KNOWLEDGE):
                s = select(df, key, rule, k)
                v = s[metric].fillna(NEVER).to_numpy(dtype=float)
                if metric != "ep_to_90":
                    v = v / 1000
                ax.plot(v, np.full(len(v), i + offset), "o", color=color, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.8, alpha=0.9,
                        label=f"{label}: one seed" if i == 0 else None)
                ax.plot([v.mean()], [i + offset], "|", color=color, markersize=13, markeredgewidth=2.2,
                        label=f"{label}: mean" if i == 0 else None)
        ax.set_xlabel(xlabel, color=INK_2)
        ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    sub = df[df[["set", "rule"]].apply(tuple, axis=1).isin(configs)]
    if sub["ep_to_90"].isna().any():
        axes[0].axvline(NEVER - 1, color="#c3c2b7", linewidth=0.8)
        axes[0].text(NEVER - 1, len(configs) - 0.45, " never ", color=INK_2, fontsize=8, ha="left")
    else:
        axes[0].set_xlim(0, sub["ep_to_90"].max() * 1.1)
    axes[1].set_xlim(0, sub["probes_detected_in_training"].max() / 1000 * 1.08)
    axes[0].set_yticks(range(len(configs)), [f"{key} · {DEF_NAME[rule]}" for key, rule in configs])
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0, colors=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Learning speed per seed where evasion is possible (cov2 and cov3 have none)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path = os.path.join(DOCS, "figures", "multiseed_learning_speed.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def figure_model_seeds():
    present = [ms for ms in MODEL_SEEDS if load(f"{model_seed_dirs(ms)[0]}/detector_eval.json")]
    if not present:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    bands = SETS[1][4]
    for ax, ms in zip(axes.ravel(), MODEL_SEEDS):
        d = load(f"{model_seed_dirs(ms)[0]}/detector_eval.json") or {}
        m = np.array([[d.get(det, {}).get("padded_probe_detected_sim_ports", {}).get(s, np.nan)
                       for s in SIZES] for det, _ in DETECTORS], dtype=float)
        title = f"cov1, model seed {ms}" + (" (original)" if ms == 42 else "")
        heatmap(ax, m, [f"{n} ({bands[det]})" for det, n in DETECTORS], SIZE_LABELS, title,
                lambda v: cell_pct(v, ""))
        ax.tick_params(axis="x", labelsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("Padded payload size and its band", color=INK_2)
    fig.suptitle("cov1 retrained with different model seeds: % of padded held-out scan probes caught\n"
                 "(simulator ports 1001–1100; same training data, only training randomness differs)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(DOCS, "figures", "detector_seed_coverage.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(os.path.join(DOCS, "figures"), exist_ok=True)
    df = collect_runs()
    ms = collect_model_seeds()
    comps, short, collapse = run_comparisons(df), shortfalls(df), collapse_check()
    df.to_csv(os.path.join(DOCS, "multiseed_runs.csv"), index=False)
    ms.to_csv(os.path.join(DOCS, "multiseed_model_seeds.csv"), index=False)
    with open(os.path.join(DOCS, "multiseed_summary.md"), "w") as f:
        f.write(markdown(df, ms, comps, short, collapse))
    print("Saved docs/multiseed_summary.md, docs/multiseed_runs.csv, docs/multiseed_model_seeds.csv")
    for path in (figure_convergence("cov0"), figure_convergence("cov1"),
                 figure_learning_speed(df), figure_model_seeds()):
        if path:
            print("Saved", path)
