"""Independent re-derivation of an externally reported finding.

A reader on Reddit reported that the two refused cells are not the same kind
of event: one fails on the downside, the other is substantially penalised for
the model BEATING its envelope. This recomputes every figure they gave from
raw data rather than trusting their summary, and writes the result for
verify_claims.py to check.

See ONE_SIDED_COVERAGE.md for the writeup.
"""
from __future__ import annotations
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import load                      # noqa: E402
from strata import ConservativeStratified, annotate  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out", "one_sided_audit.json")
CELLS = [("w8a16", ">10B"), ("w4a16", "<2B")]


def scored_rows():
    """Rebuild the leave-family-out scored rows, retaining lo/hi."""
    d = load(DATA)
    fams = sorted(d.family.unique())
    out = []
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
            t["lo"], t["hi"] = lo, hi
            t["ok"] = (t.delta >= lo) & (t.delta <= hi)
            out.append(t)
    return pd.concat(out, ignore_index=True)


def main():
    r = scored_rows()
    res = {"pooled_coverage_pct": round(100 * r.ok.mean(), 4),
           "pooled_scored_rows": int(len(r)), "cells": {}}
    rng = np.random.default_rng(0)
    for s, b in CELLS:
        g = r[(r.scheme == s) & (r.band == b)]
        below = g.delta < g.lo
        above = g.delta > g.hi
        g = g.copy()
        g["ok_one_sided"] = g.delta >= g.lo
        groups = [x for _, x in g.groupby("base_model")]
        n_groups = len(groups)
        # cluster (whole-checkpoint) bootstrap of the ONE-SIDED statistic,
        # since that is what the refusal decision is actually judged on
        boot = [pd.concat([groups[i] for i in
                           rng.integers(0, n_groups, n_groups)]).ok_one_sided.mean()
                for _ in range(20000)]
        boot = np.array(boot) * 100
        per_ckpt = {bm: round(100 * x.ok.mean(), 4)
                    for bm, x in g.groupby("base_model")}
        miss = g[~g.ok]
        res["cells"][f"{s}|{b}"] = {
            "mean_delta_of_misses": round(float(miss.delta.mean()), 4)
            if len(miss) else None,
            "two_sided_pct": round(100 * g.ok.mean(), 4),
            "one_sided_pct": round(100 * (g.delta >= g.lo).mean(), 4),
            "below_lo_pct": round(100 * below.mean(), 4),
            "above_hi_pct": round(100 * above.mean(), 4),
            "scored_rows": int(len(g)),
            "distinct_checkpoints": int(g.base_model.nunique()),
            "distinct_families": int(g.family.nunique()),
            "distinct_model_benchmark": int(g.groupby(
                ["model", "benchmark"]).ngroups),
            "per_checkpoint_coverage_pct": per_ckpt,
            "boot90_lo": round(float(np.percentile(boot, 5)), 1),
            "boot90_hi": round(float(np.percentile(boot, 95)), 1),
            "boot90_n_clusters": n_groups,
            "miss_mean_by_checkpoint": {
                bm: round(float(x[~x.ok].delta.mean()), 4)
                for bm, x in g.groupby("base_model") if (~x.ok).any()},
        }
    json.dump(res, open(OUT, "w"), indent=2)
    for k, v in res["cells"].items():
        print(f"  {k:<12} two-sided {v['two_sided_pct']:.1f}%  "
              f"one-sided {v['one_sided_pct']:.1f}%  "
              f"ckpts {v['distinct_checkpoints']}  "
              f"boot90 [{v['boot90_lo']:.0f}, {v['boot90_hi']:.0f}]")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
