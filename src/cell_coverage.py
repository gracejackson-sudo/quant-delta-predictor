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
            t["lv"] = lv
            rows.append(t)
    r = pd.concat(rows, ignore_index=True)

    out = {"nominal": NOMINAL, "pooled": {
        "coverage": float(r.ok.mean()), "scored_rows": int(len(r))}, "cells": {}}
    for (s, b), g in r.groupby(["scheme", "band"]):
        out["cells"][f"{s}|{b}"] = {
            "scheme": s, "band": b,
            "coverage": float(g.ok.mean()),
            "scored_rows": int(len(g)),
            "pct_widened": float((g.lv == "stratum-widened").mean()),
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
        mark = "  <-- below nominal" if v["coverage"] < 0.85 else ""
        print(f"{k:<24}{v['scored_rows']:>8}{v['coverage']*100:>9.1f}%"
              f"{v['pct_widened']*100:>8.0f}%{mark}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
