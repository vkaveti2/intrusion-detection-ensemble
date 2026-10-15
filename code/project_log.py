"""
project_log.py — Append timestamped entries to docs/PROJECT_LOG.txt.

Each entry is written with a single append so parallel experiment processes
don't interleave lines. Set PROJECT_LOG_PATH to send a large batch of runs
to its own file (e.g. docs/logs/) and keep the main log readable.
"""

import os
from datetime import datetime

LOG_PATH = os.environ.get("PROJECT_LOG_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "PROJECT_LOG.txt")


def log_event(title, lines=()):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n[{stamp}] [auto] {title}\n" + "".join(f"    {l}\n" for l in lines)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
