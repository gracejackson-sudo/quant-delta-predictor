"""
Measure strict coverage per (scheme, size band) AND per scheme, and write
out/cell_coverage.json.

This is the evidence file the ranking consults before it is willing to show an
interval, at both levels. Protocol: the test family is unseen AND the
calibration family is a different unseen family, averaged over every ordered
family pair.

A note on counting, because it was wrong before. Under that protocol every test
row is scored once for each calibration family, so with 8 families each row
appears 7 times. Those are `scored_pairs`. They are NOT independent rows and
must never be shown or tested as if they were: `distinct_rows` is the real
number of rows and `distinct_checkpoints` is the real unit of evidence. The
rate itself (covered / scored_pairs) is a fine average over calibration
choices; only the count and any independence assumption were wrong.

Both levels are judged the same way, by the same code (rank.classify_cell):
  * the statistic is ONE-SIDED coverage -- does the true delta stay at or above
    the interval's lower bound? The risk being bounded is accuracy loss;
  * uncertainty comes from a bootstrap that resamples whole CHECKPOINTS;
  * a level with fewer than MIN_CELL_CHECKPOINTS checkpoints, or whose 90%
    bootstrap interval straddles the refusal line, is "insufficient evidence".
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load  # noqa: E402
from strata import ConservativeStratified, annotate  # noqa: E402
import cluster_boot  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out", "cell_coverage.json")
NOMINAL = 0.90
# Cells keep the original 20000 draws (their published bounds must not move).
# Schemes have more checkpoints, hence a Monte-Carlo (not exact) bootstrap, so
# they use more draws to make the bound stable to about 0.1pp across seeds.
CELL_DRAWS = 20_000
SCHEME_DRAWS = 200_000


def scored_pairs(d):
    """One row per (test row, calibration family) evaluation."""
    fams = sorted(d.family.unique())
    rows = []
    for tf in fams:
        for cf in fams:
            if cf == tf:
                continue
            te, ca = d[d.family == tf], d[d.family == cf]
            tr = d[~d.family.isin([tf, cf])]
            if len(te) == 0 or len(ca) < 19 or len(tr) < 50:
                continue
            m = ConservativeStratified().fit_calibrated(tr, ca)
            _, lo, hi, lv = m.predict_interval(te)
            t = annotate(te).copy()
            t["row_id"] = te.index.to_numpy()
            t["ok"] = (t.delta >= lo) & (t.delta <= hi)
            t["ok_one"] = t.delta >= lo
            t["lo_"], t["hi_"], t["lv"] = lo, hi, lv
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _boot(g, col, draws, seed=0):
    """Checkpoint-cluster bootstrap 90% interval (5th, 95th percentile, in %)."""
    grp = [x[col].to_numpy(float) for _, x in g.groupby("base_model")]
    lo, hi = cluster_boot.bounds([x.sum() for x in grp], [len(x) for x in grp],
                                 rng=np.random.default_rng(seed), draws=draws)
    return round(float(lo), 1), round(float(hi), 1)


def scheme_records(r):
    """Per-scheme coverage, judged exactly like a cell (see module docstring)."""
    out = {}
    for s, g in r.groupby("scheme"):
        byfam = g.groupby("family").ok_one.mean()
        lo1, hi1 = _boot(g, "ok_one", SCHEME_DRAWS)
        lo2, hi2 = _boot(g, "ok", SCHEME_DRAWS)
        out[s] = {
            "scheme": s,
            "scored_pairs": int(len(g)),
            "distinct_rows": int(g.row_id.nunique()),
            "distinct_checkpoints": int(g.base_model.nunique()),
            "distinct_families": int(g.family.nunique()),
            "coverage_one_sided": float(g.ok_one.mean()),
            "coverage": float(g.ok.mean()),                 # two-sided, for the 90% interval
            "boot90_lo": lo1, "boot90_hi": hi1,             # one-sided; what is judged
            "boot90_two_sided_lo": lo2, "boot90_two_sided_hi": hi2,
            "by_family_one_sided": {k: float(v) for k, v in byfam.items()},
            "worst_family_one_sided": float(byfam.min()),
            "worst_family_name": str(byfam.idxmin()),
            "spread_one_sided": float(byfam.max() - byfam.min()),
            "n_families_tested": int(byfam.size),
        }
    return out


def measure(d=None):
    if d is None:
        d = load(DATA)
    r = scored_pairs(d)

    out = {"nominal": NOMINAL, "pooled": {
        "coverage": float(r.ok.mean()),
        # Pooled one-sided coverage. The interval is mean +/- q on |delta-mu|,
        # a symmetric construction, so its nominal ONE-sided level is not 90%
        # but roughly 95%: the 10% that may miss is split across two tails.
        "coverage_one_sided": float(r.ok_one.mean()),
        "below_lo": float((r.delta < r.lo_).mean()),
        "above_hi": float((r.delta > r.hi_).mean()),
        "scored_pairs": int(len(r)),
        "distinct_rows": int(r.row_id.nunique()),
        "pairs_per_row": float(len(r) / r.row_id.nunique())}, "cells": {}}
    for (s, b), g in r.groupby(["scheme", "band"]):
        lo, hi = _boot(g, "ok_one", CELL_DRAWS)
        out["cells"][f"{s}|{b}"] = {
            "scheme": s, "band": b,
            "coverage": float(g.ok.mean()),
            "coverage_one_sided": float(g.ok_one.mean()),
            "scored_pairs": int(len(g)),
            "distinct_rows": int(g.row_id.nunique()),
            "distinct_checkpoints": int(g.base_model.nunique()),
            "distinct_families": int(g.family.nunique()),
            "pct_widened": float((g.lv == "stratum-widened").mean()),
            "boot90_lo": lo, "boot90_hi": hi,
        }
    out["schemes"] = scheme_records(r)
    for b, g in r.groupby("band"):
        out.setdefault("bands", {})[b] = {
            "coverage": float(g.ok.mean()), "scored_pairs": int(len(g)),
            "distinct_rows": int(g.row_id.nunique())}
    dm = annotate(d)
    # TRAINING support per cell -- distinct checkpoints is the honest measure
    # of how much independent evidence a cell rests on.
    out["support"] = {}
    for (s_, b_), g in dm.groupby(["scheme", "band"]):
        out["support"][f"{s_}|{b_}"] = {
            "train_rows": int(len(g)),
            "train_checkpoints": int(g.base_model.nunique()),
            "train_families": int(g.family.nunique()),
        }
    out["moe"] = {"rows": int(dm.moe.sum()),
                  "checkpoints": int(dm[dm.moe].base_model.nunique())}
    w = r[r.lv == "stratum-widened"]
    out["widening"] = {
        "pairs_widened": int(len(w)), "pairs_total": int(len(r)),
        "share": float(len(w) / len(r)),
        "coverage_where_applied": float(w.ok.mean()) if len(w) else None,
    }
    return out


def main():
    res = measure()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    p = res["pooled"]
    print(f"{p['distinct_rows']} rows, each scored under "
          f"{p['pairs_per_row']:.0f} calibration families "
          f"({p['scored_pairs']} evaluations); pooled coverage "
          f"{p['coverage']*100:.1f}% two-sided, "
          f"{p['coverage_one_sided']*100:.1f}% at or above the lower bound\n")
    print(f"{'level':<22}{'rows':>6}{'ckpts':>6}{'one-sided':>10}"
          f"{'boot 90% interval':>22}")
    for k, v in sorted(res["schemes"].items()):
        print(f"{k + ' (all sizes)':<22}{v['distinct_rows']:>6}"
              f"{v['distinct_checkpoints']:>6}{v['coverage_one_sided']*100:>9.1f}%"
              f"{'[' + str(v['boot90_lo']) + ', ' + str(v['boot90_hi']) + ']':>22}")
    for k, v in sorted(res["cells"].items()):
        print(f"{k:<22}{v['distinct_rows']:>6}{v['distinct_checkpoints']:>6}"
              f"{v['coverage_one_sided']*100:>9.1f}%"
              f"{'[' + str(v['boot90_lo']) + ', ' + str(v['boot90_hi']) + ']':>22}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
