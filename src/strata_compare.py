#!/usr/bin/env python3
"""Measure the three size-stratification variants against the shipped band.

Writes out/strata_compare.json. Exists because RANKING.md and S5 quoted this
comparison for weeks from a script (src/validate_strata.py) that printed to
stdout and registered nothing, so the figures sat unwatched and two of them
(90.3% falling to 84.4%) were measured with the width fitted in-sample.

Three variants, all built on the SHIPPED band via strata.calibrated_fit:
  scheme only   -- no size rule at all
  widen-only    -- what ships: a size cell may raise the scheme half-width,
                   never lower it
  full          -- per-cell centre and width, free to narrow (discarded in
                   September; kept here as the measured contrast)

Reported on both populations, because the paper's headline is the strict
prospective subset and the comparison has more power on the full set:
coverage, mean half-width, and the GAINED/LOST decomposition against the
scheme-only inside-set. The decomposition is the point: widen-only must be a
superset of scheme-only (a rule that can only widen cannot drop a covered
row), so "gained 2, lost 0" is the empirical half of that argument and is
registered rather than asserted.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.join(HERE, "..")

from model import load                                        # noqa: E402
from strata import (ConservativeStratified, SchemeOnlyBaseline,  # noqa: E402
                    StratifiedBaseline, annotate, calibrated_fit)

OUT = os.path.join(ROOT, "out", "strata_compare.json")
TRAIN_FAMILIES = {"llama-3", "qwen2.5", "granite", "mistral", "qwen3",
                  "gemma-2"}
_EPS = 1e-9
VARIANTS = (("scheme_only", SchemeOnlyBaseline),
            ("widen_only", ConservativeStratified),
            ("full", StratifiedBaseline))


def _inside(d, cls, rows):
    m = calibrated_fit(d, cls=cls)
    _, lo, hi, _ = m.predict_interval(rows)
    y = rows.delta.to_numpy(float)
    ok = (y >= lo - _EPS) & (y <= hi + _EPS)
    return set(rows.index[ok]), float(np.mean((hi - lo) / 2))


def compare(d, rows):
    sets, widths = {}, {}
    for name, cls in VARIANTS:
        sets[name], widths[name] = _inside(d, cls, rows)
    base = sets["scheme_only"]
    out = {"n_rows": int(len(rows))}
    for name, _ in VARIANTS:
        out[name] = {
            "inside": int(len(sets[name])),
            "coverage_pct": 100 * len(sets[name]) / len(rows),
            "mean_half_width_pp": widths[name],
            "gained_vs_scheme_only": int(len(sets[name] - base)),
            "lost_vs_scheme_only": int(len(base - sets[name])),
        }
    out["widen_only_is_superset"] = base.issubset(sets["widen_only"])
    return out


def main():
    d = annotate(load(os.path.join(ROOT, "data", "dataset.csv")))
    p = annotate(pd.read_csv(os.path.join(ROOT, "out", "real_use_case.csv")))
    strict = p[(~p.family.isin(TRAIN_FAMILIES)) & (p.group != "llama-4")]
    res = {"all_prospective": compare(d, p), "strict": compare(d, strict)}

    # Which cells actually raise their scheme's width, under the retracted
    # in-sample band and under the shipped one. S5 says "six became four" and
    # that only two are the same cells; both halves are counted here so the
    # claim is gated. A scheme whose widening band is REPLACED (one size out,
    # another in) is distinguished from one that gains or drops a band --
    # W8A16 is the only replacement, and that uniqueness is the registered
    # figure rather than a sentence.
    def _widening(m):
        return {k for k, c in m.by_stratum.items()
                if c["half_width"] > (m.by_scheme.get(k[0])
                                      or m.global_)["half_width"]}
    _ins = ConservativeStratified().fit(d)
    _cal = calibrated_fit(d)
    wi, wc = _widening(_ins), _widening(_cal)
    _repl = 0
    for _s in {k[0] for k in wi | wc}:
        _a = {k[1] for k in wi if k[0] == _s}
        _b = {k[1] for k in wc if k[0] == _s}
        if _a and _b and _a != _b and len(_a) == len(_b):
            _repl += 1
    res["cells"] = {
        "widening_in_sample": sorted("|".join(k) for k in wi),
        "widening_shipped": sorted("|".join(k) for k in wc),
        "widening_both": sorted("|".join(k) for k in (wi & wc)),
        "n_widening_in_sample": len(wi),
        "n_widening_shipped": len(wc),
        "n_widening_both": len(wi & wc),
        "n_schemes_band_replaced": _repl,
    }
    json.dump(res, open(OUT, "w"), indent=2)
    for pop in ("strict", "all_prospective"):
        r = res[pop]
        print(f"{pop} ({r['n_rows']} rows)")
        for name, _ in VARIANTS:
            v = r[name]
            print(f"  {name:12s} {v['inside']:3d}/{r['n_rows']:3d} = "
                  f"{v['coverage_pct']:6.2f}%  hw {v['mean_half_width_pp']:.3f}pp"
                  f"  +{v['gained_vs_scheme_only']}/-{v['lost_vs_scheme_only']}")
        print(f"  widen-only superset of scheme-only: "
              f"{r['widen_only_is_superset']}")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
