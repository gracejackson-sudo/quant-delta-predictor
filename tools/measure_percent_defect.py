#!/usr/bin/env python3
"""Re-measure the escaped-percent defect from history, not from memory.

The appendix entry in paper/*.tex quotes four figures for the defect as it
stood at 708a84d, the last commit with it live. This reproduces them by
lifting the FOUR `body = re.sub(...)` lines verbatim out of
`708a84d:paper/audit_paper.py` and executing them against
`708a84d:paper/neurips_main.tex`. It does not reimplement the stripper,
because a reimplementation would answer a question about the reimplementation.

Usage:  python tools/measure_percent_defect.py            # report
        python tools/measure_percent_defect.py --write    # rewrite the JSON

Writes out/percent_defect.json, which is committed and is what
src/verify_claims.py reads, so the appendix's figures are macros like every
other figure in the paper rather than hand-typed values beside a tool that
could reproduce them. The JSON is the only path by which a measurement of
history reaches the paper; the test
test_percent_defect_json_matches_a_fresh_measurement re-measures and
compares, and skips loudly when the history is absent.

Exits 0 and prints a table. Exits 3 -- "could not check" -- when the history
it needs is not in this clone, which is the state of a shallow clone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys

FIX = "2aa90e4"          # the commit that added the lookbehind
BEFORE = "708a84d"       # its parent: the last commit with the bug live
INTRODUCED = "db34a29"   # the commit that added paper/audit_paper.py
VARIANTS = ("neurips_main.tex", "main.tex")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "out", "percent_defect.json")


def _show(rev_path):
    r = subprocess.run(["git", "show", rev_path],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def measure_one(src_lines, tex):
    """-> every quantity the appendix entry states, for one variant.

    The four `body = re.sub(...)` lines come from 708a84d's own
    paper/audit_paper.py and are EXECUTED, not reimplemented: a
    reimplementation would answer a question about the reimplementation.
    The fixed comparison re-executes the same four with only the lookbehind
    added to the last one, so the difference between the two is the defect
    and nothing else.
    """
    def run(ls):
        ns = {"re": re, "TEX": tex}
        for line in ls:
            exec(line, ns)                                   # noqa: S102
        return ns["body"]

    broken = run(src_lines)
    fixed = run(src_lines[:3]
                + [src_lines[3].replace('r"%.*"', 'r"(?<!\\\\)%.*"')])

    # The runs the broken rule removed, taken from the same pipeline state
    # the rule saw: the first three substitutions applied, the fourth not.
    pre_ns = {"re": re, "TEX": tex}
    for line in src_lines[:3]:
        exec(line, pre_ns)                                   # noqa: S102
    pre = pre_ns["body"]
    # The run is classified by the character BEFORE its percent sign, not by
    # the run's own first character: `%.*` matches AT the percent, so no run
    # ever begins with the backslash that distinguishes a printed percent
    # from a comment. The first version of this split tested
    # run.startswith("\\%") and reported every run as a real comment and
    # zero runs of prose -- the exact inversion of the defect it measures.
    runs = [(m.start(), m.group(0)) for m in re.finditer(r"%.*", pre)]
    real = [r for i, r in runs if i == 0 or pre[i - 1] != "\\"]
    prose = [r for i, r in runs if i > 0 and pre[i - 1] == "\\"]

    hidden = len(fixed) - len(broken)
    return {
        "escaped_percent_signs": len(re.findall(r"\\%", tex)),
        "real_comment_marks": len(re.findall(r"(?<!\\)%", tex)),
        "runs_removed": len(runs),
        "runs_that_were_comments": len(real),
        "runs_of_prose": len(prose),
        "chars_in_prose_runs": sum(len(r) for r in prose),
        "body_that_should_be_read": len(fixed),
        "body_that_was_read": len(broken),
        "hidden_characters": hidden,
        "space_left_per_run": sum(len(r) for r in prose) - hidden,
        # SIX places, not four, and the difference is a figure in the paper.
        # A full float repr served nothing and did harm -- "96.95495998690305"
        # contains "95495998690305", a hash-shaped token the supplement's
        # relabeller then resolves against git history -- so these are
        # rounded on the way in. At four places the hidden fraction becomes
        # 3.045, and rendering THAT to the two places the paper prints gives
        # 3.04 rather than 3.05: double rounding, which moved a published
        # figure by a hundredth. Six places keeps the second rounding
        # faithful and is still far too short to look like a hash.
        "hidden_pct": round(100.0 * hidden / len(fixed), 6),
        "scanned_pct": round(100.0 * len(broken) / len(fixed), 6),
    }


def measure_hidden_figures(src_lines, tex):
    """-> how many figures the broken stripper hid, and how many it left.

    The appendix says the unread text held ONE figure and that THREE more in
    the same two sentences were visible and passed anyway. Both counts are
    properties of 708a84d, so they are snapshotted here rather than
    recomputed by the registry: the registry must load with no git at all.

    Llama-3.3 is excluded from the visible count by the `-` in the
    lookbehind: it is a version inside a model name, not a figure, which is
    the judgement the shipped numeral check encodes too.
    """
    ns = {"re": re, "TEX": tex}
    for line in src_lines[:3]:
        exec(line, ns)                                       # noqa: S102
    pre = ns["body"]
    i = pre.find("but every LOFO-derived")
    j = pre.find("92.2\\%.", i) + len("92.2\\%.")
    if i < 0 or j <= i:
        return None
    seg = pre[i:j]
    cut = re.compile(r"%.*")
    num = re.compile(r"(?<![\w.\\-])\d+\.\d+")
    hidden = num.findall(" ".join(m.group(0) for m in cut.finditer(seg)))
    visible = num.findall(cut.sub(" ", seg))
    return {"hidden_figures": len(hidden), "visible_figures": len(visible),
            "hidden_values": hidden, "visible_values": visible,
            "sentences": " ".join(seg.split())}


def measure_lofo_at_before():
    """-> the Definition-B aggregate LOFO coverage on 77873d9's corpus.

    ONLY THE HISTORICAL HALF IS SNAPSHOTTED. The appendix describes a DRIFT
    between that revision's corpus and the present one, and the present half
    moves whenever the corpus does -- so freezing the difference would
    freeze a figure that is half live. The registry subtracts this from its
    own current aggregate, which keeps the drift honest and still needs no
    git.
    """
    rev = "77873d9"
    tex = _show(f"{rev}:data/dataset.csv")
    if tex is None:
        return None
    import io
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from predictor import SchemeMean, Conformal                # noqa: E402
    import strata                                             # noqa: E402
    d = pd.read_csv(io.StringIO(tex))
    d = d[d.acc_before >= 20.0].reset_index(drop=True)
    fams = sorted(d.family.unique())
    per = {}
    for t in fams:
        te = d[d.family == t]
        pool = d[d.family != t]
        n = c = 0
        for cf in sorted(pool.family.unique()):
            ca = pool[pool.family == cf]
            tr = pool[pool.family != cf]
            if len(ca) < 19 or len(tr) < 50:
                continue
            m = SchemeMean().fit(tr)
            k = Conformal(alpha=strata.ALPHA, mondrian_by="scheme").fit(m, ca)
            _, lo, hi, _ = k.predict_interval(te)
            y = te.delta.to_numpy(float)
            n += len(y)
            c += int(((y >= lo - 1e-9) & (y <= hi + 1e-9)).sum())
        if n:
            per[t] = 100.0 * c / n
    return {"revision": rev, "modelling_rows": int(len(d)),
            "aggb_pct": round(sum(per.values()) / len(per), 6),
            "per_fold_pct": {k: round(v, 6) for k, v in sorted(per.items())}}


def measure_exposure():
    """-> the span the defect was live, in days and commits, or None.

    None means a revision this needs is not in the clone. A --depth 50 clone
    of origin reaches 708a84d and 2aa90e4 but NOT db34a29, so the variant
    measurements succeed and only the exposure fails -- and the first
    version of this raised ValueError on the empty `git show` output instead
    of reporting that it could not check. A partially available history is
    the normal shallow-clone state, not an exceptional one.
    """
    def ct(h):
        r = subprocess.run(["git", "show", "-s", "--format=%ct", h],
                           capture_output=True, text=True, cwd=ROOT)
        try:
            return int(r.stdout.strip())
        except ValueError:
            return None
    a, b = ct(INTRODUCED), ct(FIX)
    if a is None or b is None:
        return None
    n = subprocess.run(["git", "rev-list", "--count",
                        f"{INTRODUCED}..{FIX}"],
                       capture_output=True, text=True, cwd=ROOT)
    if not n.stdout.strip().isdigit():
        return None
    return {
        # Floored, not rounded: the paper says "a span of 18 days" and the
        # true span is 18.26, so 18 is the number of whole days the defect
        # was live. Rounding would say 18 here and could say 19 next time
        # the same sentence was regenerated from a different pair.
        "exposure_days": math.floor((b - a) / 86400.0),
        "exposure_days_exact": round((b - a) / 86400.0, 6),
        "exposure_commits": int(n.stdout.strip()),
        "introduced": INTRODUCED,
        "fixed": FIX,
        "measured_at": BEFORE,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None,
                    help="report one variant instead of all of them")
    ap.add_argument("--write", action="store_true",
                    help="rewrite out/percent_defect.json")
    args = ap.parse_args()

    src = _show(f"{BEFORE}:paper/audit_paper.py")
    if src is None:
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

    want = (args.variant,) if args.variant else VARIANTS
    out = {"variants": {}}
    for v in want:
        tex = _show(f"{BEFORE}:paper/{v}")
        if tex is None:
            print(f"UNCHECKED: this clone cannot read {BEFORE}:paper/{v}.")
            return 3
        out["variants"][v] = measure_one(lines, tex)
    _exp = measure_exposure()
    if _exp is None:
        print(f"UNCHECKED: this clone cannot read {INTRODUCED} or {FIX}, so "
              f"the exposure span could not be measured. A shallow clone "
              f"reaches {BEFORE} and still lands here; fetch the full "
              f"history to run this.")
        return 3
    out["exposure"] = _exp
    _hf = measure_hidden_figures(lines, _show(f"{BEFORE}:paper/neurips_main.tex"))
    if _hf is None:
        print("UNCHECKED: the two sentences the appendix describes are not "
              "at the expected place in that revision; re-derive by hand.")
        return 3
    out["stripper"] = _hf
    _lo = measure_lofo_at_before()
    if _lo is None:
        print("UNCHECKED: cannot read 77873d9:data/dataset.csv, so the "
              "historical LOFO aggregate could not be snapshotted.")
        return 3
    out["lofo_at_before"] = _lo

    print(f"stripper source : {BEFORE}:paper/audit_paper.py (executed, not "
          f"reimplemented)")
    print(f"exposure        : {INTRODUCED} (added the checker) .. {FIX} "
          f"(fixed it) -- {out['exposure']['exposure_days']} days, "
          f"{out['exposure']['exposure_commits']} commits")
    print(f"  stripper hid {out['stripper']['hidden_figures']} figure(s) "
          f"{out['stripper']['hidden_values']}, left "
          f"{out['stripper']['visible_figures']} visible "
          f"{out['stripper']['visible_values']}")
    print(f"  LOFO aggregate (Def B) at {out['lofo_at_before']['revision']} "
          f"on {out['lofo_at_before']['modelling_rows']} rows: "
          f"{out['lofo_at_before']['aggb_pct']:.4f}%")
    for v, m in out["variants"].items():
        print(f"\n  {BEFORE}:paper/{v}")
        for k in ("escaped_percent_signs", "real_comment_marks",
                  "runs_removed", "runs_that_were_comments", "runs_of_prose",
                  "chars_in_prose_runs", "space_left_per_run",
                  "body_that_should_be_read", "body_that_was_read",
                  "hidden_characters"):
            print(f"    {k:<26}: {m[k]}")
        print(f"    {'hidden_pct':<26}: {m['hidden_pct']:.4f}")
        print(f"    {'scanned_pct':<26}: {m['scanned_pct']:.4f}")

    if args.write:
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
