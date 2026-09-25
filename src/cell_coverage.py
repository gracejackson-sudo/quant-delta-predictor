"""
Measure strict per-(scheme, size band) coverage and write out/cell_coverage.json.

This is the evidence file the ranking consults before it is willing to show an
interval for a given cell. Protocol: the test family is unseen AND the
calibration family is a different unseen family, averaged over every ordered
family pair.

Cells whose measured coverage is materially below nominal are recorded here and
the ranking REFUSES to print an interval for them (see rank.py REFUSE_BELOW).
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

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out", "cell_coverage.json")
NOMINAL = 0.90


def measure(d=None):
    if d is None:
        d = load(DATA)
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
            t["ok"] = (t.delta >= lo) & (t.delta <= hi)
            t["lo_"], t["hi_"] = lo, hi
            t["lv"] = lv
            rows.append(t)
    r = pd.concat(rows, ignore_index=True)

    out = {"nominal": NOMINAL, "pooled": {
        "coverage": float(r.ok.mean()),
        # Pooled one-sided coverage. The interval is mean +/- q on |delta-mu|,
        # a symmetric construction, so its nominal ONE-sided level is not 90%
        # but roughly 95%: the 10% that may miss is split across two tails.
        # Reporting this stops a one-sided measurement being compared against
        # a two-sided nominal, which would flatter every cell.
        "coverage_one_sided": float((r.delta >= r.lo_).mean())
        if "lo_" in r else None,
        "below_lo": float((r.delta < r.lo_).mean()) if "lo_" in r else None,
        "above_hi": float((r.delta > r.hi_).mean()) if "hi_" in r else None,
        "scored_rows": int(len(r))}, "cells": {}}
    rng = np.random.default_rng(0)
    for (s, b), g in r.groupby(["scheme", "band"]):
        n_ckpts = int(g.base_model.nunique())
        # Cluster (whole-checkpoint) bootstrap of the ONE-SIDED statistic --
        # the one the refusal decision is actually judged on. Resampling rows
        # would treat correlated per-checkpoint rows as independent evidence;
        # resampling checkpoints does not. With few checkpoints this interval
        # is coarse (few achievable resample compositions) rather than smooth,
        # which is exactly why a checkpoint-count floor is also needed below,
        # not a substitute for one. (External finding, see ONE_SIDED_COVERAGE.md)
        groups = [gg for _, gg in g.groupby("base_model")]
        one_sided_ok = (g.delta >= g.lo_) if "lo_" in g else None
        if one_sided_ok is not None:
            g = g.assign(_ok1=one_sided_ok)
            groups = [gg.assign(_ok1=(gg.delta >= gg.lo_)) for _, gg in g.groupby("base_model")]
            boot = [pd.concat([groups[i] for i in
                               rng.integers(0, n_ckpts, n_ckpts)])._ok1.mean()
                    for _ in range(20000)]
            boot90_lo = round(float(np.percentile(boot, 5)) * 100, 1)
            boot90_hi = round(float(np.percentile(boot, 95)) * 100, 1)
        else:
            boot90_lo = boot90_hi = None
        out["cells"][f"{s}|{b}"] = {
            "scheme": s, "band": b,
            "coverage": float(g.ok.mean()),
            # One-sided coverage. A downside envelope has not failed when the
            # model BEATS it, but two-sided containment counts that as a miss.
            # Reported alongside so the two kinds of miss stay distinguishable.
            # (External finding, see ONE_SIDED_COVERAGE.md)
            "coverage_one_sided": float((g.delta >= g.lo_).mean())
            if "lo_" in g else None,
            "scored_rows": int(len(g)),
            # The honest unit of evidence is the checkpoint, not the row: rows
            # from one checkpoint across many benchmarks are correlated views
            # of a single quantization run. This is already enforced on train
            # support; reporting it here makes it visible for coverage too.
            "distinct_checkpoints": n_ckpts,
            "distinct_families": int(g.family.nunique()),
            "pct_widened": float((g.lv == "stratum-widened").mean()),
            "boot90_lo": boot90_lo,
            "boot90_hi": boot90_hi,
        }
    for b, g in r.groupby("band"):
        out.setdefault("bands", {})[b] = {
            "coverage": float(g.ok.mean()), "scored_rows": int(len(g))}
    dm = annotate(d if d is not None else load(DATA))
    # TRAINING support per cell -- distinct checkpoints is the honest measure
    # of how much independent evidence a cell rests on. Scored rows can be
    # large while the underlying evidence is one checkpoint repeated across
    # benchmarks, which is not independent evidence at all.
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
        "rows_widened": int(len(w)), "rows_total": int(len(r)),
        "share": float(len(w) / len(r)),
        "coverage_where_applied": float(w.ok.mean()) if len(w) else None,
    }
    return out


def main():
    res = measure()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"pooled {res['pooled']['coverage']*100:.1f}% over "
          f"{res['pooled']['scored_rows']} scored rows\n")
    print(f"{'cell':<24}{'scored':>8}{'coverage':>10}{'widened':>9}")
    for k, v in sorted(res["cells"].items(), key=lambda kv: kv[1]["coverage"]):
        one = v.get("coverage_one_sided")
        mark = ("  <-- one-sided below 85%"
                if (one if one is not None else v["coverage"]) < 0.85 else "")
        print(f"{k:<24}{v['scored_rows']:>8}{v['coverage']*100:>9.1f}%"
              f"{v['pct_widened']*100:>8.0f}%{mark}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
