"""
Does stratifying by size band actually improve coverage, or just look busier?

Honest test: leave-one-family-out. Fit on 7 families, score the 8th, so the
size cells are never fitted on the checkpoints they are scored against. Then
the same comparison on the prospective set, which is where the failure was
originally observed.

Reports overall coverage, coverage WITHIN each size band (the thing item #4 is
supposed to fix), and interval width, for stratified vs scheme-only.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load  # noqa: E402
from strata import SchemeOnlyBaseline, StratifiedBaseline, annotate  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
PROSPECTIVE = os.path.join(HERE, "..", "out", "real_use_case.csv")


def lofo(d, cls):
    out = []
    for fam in sorted(d.family.unique()):
        tr, te = d[d.family != fam], d[d.family == fam]
        if len(tr) < 50 or len(te) == 0:
            continue
        m = cls().fit(tr)
        yhat, lo, hi, lvl = m.predict_interval(te)
        t = annotate(te).copy()
        t["yhat"], t["lo"], t["hi"], t["level"] = yhat, lo, hi, lvl
        t["covered"] = (t.delta >= t.lo) & (t.delta <= t.hi)
        t["width"] = t.hi - t.lo
        t["ae"] = np.abs(t.delta - t.yhat)
        t["fold"] = fam
        out.append(t)
    return pd.concat(out, ignore_index=True)


def summarize(name, r):
    print(f"\n  {name}")
    print(f"    overall      n={len(r):<5} coverage={r.covered.mean()*100:5.1f}%"
          f"  mean width={r.width.mean():5.2f}pp  MAE={r.ae.mean():.3f}")
    for b in ["<2B", "2-10B", ">10B"]:
        g = r[r.band == b]
        if len(g) == 0:
            continue
        print(f"      {b:<7} n={len(g):<5} coverage={g.covered.mean()*100:5.1f}%"
              f"  mean width={g.width.mean():5.2f}pp")
    agg = r[r.scheme.isin(["w4a16", "nvfp4"])]
    small_agg = agg[agg.band == "<2B"]
    if len(small_agg):
        print(f"      4-bit on <2B models: n={len(small_agg)} "
              f"coverage={small_agg.covered.mean()*100:.1f}%  "
              f"width={small_agg.width.mean():.2f}pp   <-- the failure case")


def main():
    d = load(DATA)
    print("=" * 72)
    print("DOES SIZE STRATIFICATION HELP?  (leave-one-family-out)")
    print("=" * 72)

    rs = lofo(d, StratifiedBaseline)
    ro = lofo(d, SchemeOnlyBaseline)
    summarize("scheme only (current shipped behaviour)", ro)
    summarize("stratified by size band", rs)

    print("\n  per-band change in |coverage - 90%| (lower is better):")
    for b in ["<2B", "2-10B", ">10B"]:
        a = ro[ro.band == b]
        s = rs[rs.band == b]
        if len(a) == 0 or len(s) == 0:
            continue
        ea, es = abs(a.covered.mean() - .9), abs(s.covered.mean() - .9)
        verdict = "better" if es < ea - 1e-9 else ("worse" if es > ea else "same")
        print(f"    {b:<7} {ea*100:5.1f}pp -> {es*100:5.1f}pp   {verdict}")

    print("\n  how often did a stratified cell actually get used?")
    print("   ", rs.level.value_counts().to_dict())

    # ---------------------------------------------------------------- prospective
    if os.path.exists(PROSPECTIVE):
        print("\n" + "=" * 72)
        print("SAME COMPARISON ON THE PROSPECTIVE SET (never used to fit)")
        print("=" * 72)
        p = pd.read_csv(PROSPECTIVE)
        keep = [c for c in ("model", "base_model", "family", "params_b",
                            "scheme", "benchmark", "acc_before", "acc_after",
                            "delta", "group") if c in p.columns]
        p = p[keep]
        for name, cls in (("scheme only", SchemeOnlyBaseline),
                          ("stratified", StratifiedBaseline)):
            m = cls().fit(d)
            yhat, lo, hi, lvl = m.predict_interval(p)
            t = annotate(p).copy()
            t["lo"], t["hi"], t["level"] = lo, hi, lvl
            t["covered"] = (t.delta >= t.lo) & (t.delta <= t.hi)
            t["width"] = t.hi - t.lo
            print(f"\n  {name}: {int(t.covered.sum())}/{len(t)} = "
                  f"{t.covered.mean()*100:.1f}%  width={t.width.mean():.2f}pp")
            for b in ["<2B", "2-10B", ">10B"]:
                g = t[t.band == b]
                if len(g):
                    print(f"      {b:<7} n={len(g):<4} "
                          f"coverage={g.covered.mean()*100:5.1f}%  "
                          f"width={g.width.mean():.2f}pp")
            sm = t[(t.band == "<2B")]
            if len(sm):
                print(f"      gemma-3-1b style (<2B): "
                      f"{int(sm.covered.sum())}/{len(sm)}")

    # ---------------------------------------------------------------- MoE
    print("\n" + "=" * 72)
    print("MoE: can we stratify on it?")
    print("=" * 72)
    m = StratifiedBaseline().fit(d)
    if m.moe_stats:
        s = m.moe_stats
        print(f"  MoE rows in training: {s['n']} from "
              f"{m.n_moe_checkpoints} checkpoints "
              f"(all Mixtral), mean {s['mean']:+.3f}pp, worst {s['worst']:+.2f}")
    print("  -> too few distinct checkpoints to form its own envelope. "
          "Carried as a\n     WARNING flag instead of a separate interval.")

    print("\n  cells rejected as too thin (fall back to scheme level):")
    for (s_, b), c in sorted(m.rejected.items()):
        print(f"    {s_:<12} {b:<7} n={c['n']:<3} "
              f"checkpoints={c['n_checkpoints']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
