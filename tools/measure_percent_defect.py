#!/usr/bin/env python3
"""Re-measure the escaped-percent defect from history, not from memory.

The appendix entry in paper/*.tex quotes four figures for the defect as it
stood at 708a84d, the last commit with it live. This reproduces them by
lifting the FOUR `body = re.sub(...)` lines verbatim out of
`708a84d:paper/audit_paper.py` and executing them against
`708a84d:paper/neurips_main.tex`. It does not reimplement the stripper,
because a reimplementation would answer a question about the reimplementation.

Usage:  python tools/measure_percent_defect.py [--variant main.tex]

Exits 0 and prints a table. Exits 3 -- "could not check" -- when the history
it needs is not in this clone, which is the state of a shallow clone.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

FIX = "2aa90e4"          # the commit that added the lookbehind
BEFORE = "708a84d"       # its parent: the last commit with the bug live
INTRODUCED = "db34a29"   # the commit that added paper/audit_paper.py


def _show(rev_path):
    r = subprocess.run(["git", "show", rev_path],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="neurips_main.tex")
    args = ap.parse_args()

    src = _show(f"{BEFORE}:paper/audit_paper.py")
    tex = _show(f"{BEFORE}:paper/{args.variant}")
    if src is None or tex is None:
        print(f"UNCHECKED: this clone cannot read {BEFORE}. A shallow clone "
              f"or a worktree without the history will land here; fetch the "
              f"full history to run this.")
        return 3

    lines = [l for l in src.split("\n") if l.startswith("body = re.sub(")]
    if len(lines) != 4:
        print(f"UNCHECKED: expected 4 stripper lines at {BEFORE}, found "
              f"{len(lines)}. The reduction moved; re-derive by hand rather "
              f"than trusting this script.")
        return 3

    def run(ls):
        ns = {"re": re, "TEX": tex}
        for l in ls:
            exec(l, ns)                                  # noqa: S102
        return ns["body"]

    broken = run(lines)
    fixed = run(lines[:3] + [lines[3].replace('r"%.*"', 'r"(?<!\\\\)%.*"')])
    hidden = len(fixed) - len(broken)

    print(f"stripper source : {BEFORE}:paper/audit_paper.py (executed, not "
          f"reimplemented)")
    print(f"document        : {BEFORE}:paper/{args.variant}")
    print(f"exposure        : {INTRODUCED} (added the checker) .. {FIX} "
          f"(fixed it)")
    print()
    # Counted outside the f-string: nesting a raw pattern with
    # backslashes inside an f-string is how the first version of this
    # script reported 0 escaped percents and 139 comments.
    n_escaped = len(re.findall(r"\\%", tex))
    n_comments = len(re.findall(r"(?<!\\)%", tex))
    print(f"  escaped percent signs      : {n_escaped}")
    print(f"  real comment marks         : {n_comments}")
    print(f"  N, body that should be read: {len(fixed)}")
    print(f"  body that was read         : {len(broken)}")
    print(f"  hidden characters          : {hidden}")
    print(f"  hidden fraction            : {100.0 * hidden / len(fixed):.4f}%")
    print(f"  scanned fraction           : "
          f"{100.0 * len(broken) / len(fixed):.4f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
