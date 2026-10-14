"""
project_log.py — Append timestamped entries to PROJECT_LOG.txt (repo root).

Each entry is written with a single append so parallel experiment processes
don't interleave lines.
"""

import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "PROJECT_LOG.txt")


def log_event(title, lines=()):
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n[{stamp}] [auto] {title}\n" + "".join(f"    {l}\n" for l in lines)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
