"""
Is the symmetric conformal band or the asymmetric empirical band the better
thing to SHOW a user?

The shipped tool prints mean +/- conformal half-width. That is symmetric about
the mean by construction. The delta distribution is left-skewed, so a symmetric
band spends width on an upside that rarely happens and under-covers the
downside that does. gemma-3-1b W4A16 sits exactly in that gap: the symmetric
bound catches 3 of 6 benchmarks, the empirical 5th percentile catches 6.

Measured here under the same strict protocol used everywhere else: test family
unseen AND calibration family a different unseen family.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")
ALPHA = 0.10


def conformal_band(cal_deltas, centre, alpha=ALPHA):
    r = np.sort(np.abs(np.asarray(cal_deltas, float) - centre))
    n = len(r)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q = r[k - 1] if k <= n else np.inf
    return centre - q, centre + q


def empirical_band(cal_deltas, alpha=ALPHA):
    """Two-sided empirical band with the finite-sample index correction."""
    v = np.sort(np.asarray(cal_deltas, float))
    n = len(v)
    klo = int(np.floor((n + 1) * (alpha / 2))) - 1
    khi = int(np.ceil((n + 1) * (1 - alpha / 2))) - 1
    if klo < 0 or khi > n - 1:
        return -np.inf, np.inf
    return float(v[klo]), float(v[khi])


def hybrid_band(cal_deltas, centre, alpha=ALPHA):
    """
    Asymmetric empirical band where the calibration set supports it, falling
    back to the symmetric conformal band when it does not.

    Empirical covers better (it follows the real left skew) but needs
    n >= 2/alpha - 1 for a two-sided index to exist; below that it returns
    +/-inf, which is useless to a user. Conformal always returns something
    finite. Taking the empirical band when defined and conformal otherwise
    keeps the better shape without ever emitting an infinite interval.
    """
    elo, ehi = empirical_band(cal_deltas, alpha)
    if np.isfinite(elo) and np.isfinite(ehi):
        return elo, ehi, "empirical"
    clo, chi = conformal_band(cal_deltas, centre, alpha)
    return clo, chi, "conformal_fallback"


def main():
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
            for s, g in te.groupby("scheme"):
                cg = ca[ca.scheme == s]
                tg = tr[tr.scheme == s]
                if len(cg) < 9 or len(tg) < 5:
                    continue
                centre = float(tg.delta.mean())
                clo, chi = conformal_band(cg.delta.to_numpy(), centre)
                elo, ehi = empirical_band(cg.delta.to_numpy())
                y = g.delta.to_numpy(float)
                hlo, hhi, hsrc = hybrid_band(cg.delta.to_numpy(), centre)
                rows.append(pd.DataFrame({
                    "scheme": s, "test_family": tf, "y": y,
                    "row_id": g.index.to_numpy(),
                    "conf_ok": (y >= clo) & (y <= chi),
                    "emp_ok": (y >= elo) & (y <= ehi),
                    "hyb_ok": (y >= hlo) & (y <= hhi),
                    "conf_w": chi - clo, "emp_w": ehi - elo,
                    "hyb_w": hhi - hlo, "hyb_src": hsrc,
                    "conf_lo": clo, "emp_lo": elo, "hyb_lo": hlo,
                }))
    r = pd.concat(rows, ignore_index=True)

    print("=" * 74)
    print("SYMMETRIC CONFORMAL vs ASYMMETRIC EMPIRICAL BAND")
    print("=" * 74)
    print(f"\nscored evaluations: {len(r)} = {r.row_id.nunique()} rows, each scored "
          f"under several calibration families (strict held-out calibration)")
    print(f"   {'band':<28}{'coverage':>10}{'mean width':>13}"
          f"{'lower-side misses':>20}")
    for nm, ok, w, lo in (("symmetric conformal (shipped)", r.conf_ok,
                           r.conf_w, r.conf_lo),
                          ("asymmetric empirical", r.emp_ok, r.emp_w,
                           r.emp_lo),
                          ("HYBRID (empirical + fallback)", r.hyb_ok,
                           r.hyb_w, r.hyb_lo)):
        below = float((r.y < lo).mean())
        print(f"   {nm:<28}{100*ok.mean():>9.1f}%{w.mean():>12.2f}pp"
              f"{100*below:>19.1f}%")

    print(f"\n   infinite-width rows: empirical "
          f"{int((~np.isfinite(r.emp_w)).sum())}, hybrid "
          f"{int((~np.isfinite(r.hyb_w)).sum())}")
    print(f"   hybrid fell back to conformal on "
          f"{100*(r.hyb_src=='conformal_fallback').mean():.1f}% of rows")
    print("\n   per-scheme coverage:")
    g = r.groupby("scheme").agg(conf=("conf_ok", "mean"),
                                emp=("emp_ok", "mean"),
                                hyb=("hyb_ok", "mean"),
                                n=("y", "size"))
    for s, row in g.iterrows():
        better = "empirical" if row.emp > row.conf else (
            "conformal" if row.conf > row.emp else "tie")
        print(f"     {s:<13} conformal {100*row.conf:5.1f}%   "
              f"empirical {100*row.emp:5.1f}%   hybrid {100*row.hyb:5.1f}%   "
              f"n={int(row.n)}")

    # the concrete case
    print("\n   gemma-3-1b-it W4A16 real deltas vs each band (fit on all data):")
    w4 = d[d.scheme == "w4a16"].delta.to_numpy()
    c = conformal_band(w4, float(w4.mean()))
    e = empirical_band(w4)
    gem = [-3.03, -2.99, -2.90, -2.41, -1.34, 1.40]
    print(f"     symmetric conformal [{c[0]:+.2f}, {c[1]:+.2f}] -> "
          f"{sum(1 for x in gem if c[0] <= x <= c[1])}/6 inside")
    print(f"     asymmetric empirical[{e[0]:+.2f}, {e[1]:+.2f}] -> "
          f"{sum(1 for x in gem if e[0] <= x <= e[1])}/6 inside")

    res = {
        "n_scored_pairs": int(len(r)),
        "n_distinct_rows": int(r.row_id.nunique()),
        "conformal": {"coverage": float(r.conf_ok.mean()),
                      "mean_width": float(r.conf_w.mean()),
                      "lower_miss_rate": float((r.y < r.conf_lo).mean())},
        "empirical": {"coverage": float(r.emp_ok.mean()),
                      "mean_width": float(r.emp_w.mean()),
                      "lower_miss_rate": float((r.y < r.emp_lo).mean())},
        "hybrid": {"coverage": float(r.hyb_ok.mean()),
                   "mean_width": float(r.hyb_w[np.isfinite(r.hyb_w)].mean()),
                   "lower_miss_rate": float((r.y < r.hyb_lo).mean()),
                   "fallback_share": float(
                       (r.hyb_src == "conformal_fallback").mean())},
        "per_scheme": {s: {"conformal": float(v.conf), "empirical":
                           float(v.emp), "hybrid": float(v.hyb),
                           "n": int(v.n)}
                       for s, v in g.iterrows()},
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "interval_shape.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {OUT}/interval_shape.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
