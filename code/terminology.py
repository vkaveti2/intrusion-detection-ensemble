"""
terminology.py — the one place thesis-facing names for the agent knowledge levels are defined.

The internal keys "black" and "white" are identifiers only. They appear in file names
(convergence_ddqn_<rule>_white.json, models/ddqn_<rule>_white.zip), result dictionaries and
command-line choices, and renaming them would orphan trained agents and archived results.
Anything a reader sees (tables, figures, log lines, docs) uses the labels below.

  reward-only        the agent sees nothing but the reward (detected / not detected); its
                     observation is always zeros
  feature-observing  the agent also observes the probe's 77 flow features, standardised with the
                     detectors' scaler. It sees flow features, not model parameters or gradients,
                     so "white-box" would overstate what it knows.
  partial feature    ("gray") a 4-feature subset; defined in ids_env.py but used in no reported
  observation        experiment

Use these everywhere a label is printed:  from terminology import KNOWLEDGE_LABEL, label
"""

KNOWLEDGE_LABEL = {
    "black": "reward-only",
    "white": "feature-observing",
    "gray": "partial feature observation",
}
KNOWLEDGE_KEYS = tuple(KNOWLEDGE_LABEL)


def label(key, noun="agent"):
    """label("white") -> "feature-observing agent";  label("black", None) -> "reward-only"."""
    base = KNOWLEDGE_LABEL[key]
    return f"{base} {noun}" if noun else base


def capital(key, noun="agent"):
    s = label(key, noun)
    return s[0].upper() + s[1:]


# Phrases the thesis uses for the same things every time
SIMULATION = "constrained flow-feature simulation"
GROUPING = "invariant-feature groups"
STRONGEST_COMPARATOR = "strongest observed standalone comparator"
