"""
check_terminology.py — fail if a retired term appears in a thesis-facing file (docs/00_terminology.md).

Scans docs/*.md (except the historical records listed in HISTORICAL), the top-level summarizer and
plot scripts, and the live scripts in code/. Prints every offending line and exits 1, so it can
run before each commit or before pasting text into the thesis.

Checked:
  - "white-box", "black-box" (also "white box", "whitebox", "black box")
  - "optimal" / "sub-optimal" describing a detector or a policy
  - "99.9-100%" style ranges as a robustness claim (report exact counts)
Not checked: the internal keys "white" / "black" (file names, dict keys, CLI choices).

    python check_terminology.py [--list-skipped]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RETIRED = [
    (re.compile(r"\b(white|black)[\s_-]?box", re.I), 'retired label; use "feature-observing" / "reward-only"'),
    (re.compile(r"\b(sub-?)?optimal(ly)?\b", re.I), 'avoid "optimal"; use "strongest observed ..." or "best available ..."'),
    (re.compile(r"99\.9\s*[–-]\s*100\s*%"), "report exact counts (k of n), not a range"),
]
# Historical records: dated logs and early build notes keep their original wording.
HISTORICAL = {"PROJECT_LOG.txt", "BUILD_NOTES_2026-09-24.md"}
SKIP_FILES = {"old_ids_env.py", "dqn.py", "check_terminology.py", "terminology.py"}  # legacy / this checker
SKIP_DIRS = {"logs", "figures", "__pycache__"}
SCRIPT_DIRS = [ROOT, os.path.join(ROOT, "code")]


def targets():
    docs = os.path.join(ROOT, "docs")
    for dirpath, dirs, files in os.walk(docs):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            if f.endswith(".md") and f not in HISTORICAL:
                yield os.path.join(dirpath, f)
    for d in SCRIPT_DIRS:
        for f in sorted(os.listdir(d)):
            if f.endswith(".py") and f not in SKIP_FILES:
                yield os.path.join(d, f)


def main():
    bad = 0
    for path in targets():
        with open(path, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                # a doc may quote a retired term to explain it: lines marked `retired:` or quoting the glossary row
                if "00_terminology" in line or "retired:" in line.lower():
                    continue
                if path.endswith("00_terminology.md"):    # the glossary quotes retired terms to define them
                    continue
                for pat, why in RETIRED:
                    for m in pat.finditer(line):
                        bad += 1
                        print(f"{os.path.relpath(path, ROOT)}:{n}: '{m.group(0)}' — {why}\n    {line.strip()[:160]}")
    if "--list-skipped" in sys.argv:
        print("historical (not checked):", ", ".join(sorted(HISTORICAL)), "| docs/logs/, docs/figures/")
    print(f"\n{bad} retired-term occurrence(s)" if bad else "terminology OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
