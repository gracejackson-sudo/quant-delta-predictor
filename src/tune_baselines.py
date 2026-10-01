"""Nested cross-validated tuning of the ridge and gradient-boosting baselines.

Implements TUNING_PREREGISTRATION.md exactly. Read that file first; it was
committed before this script was run.

Outer loop: leave-one-family-out, the same 6 folds the shipped `pred_mae::*`
figures use. Inner loop: leave-one-family-out again, over the 5 outer-training
families only. Hyperparameters are selected once per outer fold, from inner
folds that never contain a row of that fold's held-out family.

The untuned path in src/diagnose.py is untouched and still runs, so the tuned
and untuned figures are both reproducible.

Writes out/tuned_baselines.json.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(__file__))
from model import Predictor, featurize, load  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out", "tuned_baselines.json")

# --- grids, fixed in TUNING_PREREGISTRATION.md before this was run ---------
RIDGE_ALPHAS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
GB_GRID = [
    dict(learning_rate=lr, max_depth=md, min_samples_leaf=leaf,
         l2_regularization=l2, max_iter=300, random_state=0,
         early_stopping=False)
    for lr in (0.01, 0.05, 0.1)
    for md in (2, 3, None)
    for leaf in (5, 15, 30)
    for l2 in (0.0, 1.0, 10.0)
]
RIDGE_EDGES = {min(RIDGE_ALPHAS), max(RIDGE_ALPHAS)}


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def fit_ridge(tr, te, alpha):
    return Predictor(alpha=alpha).fit(tr).predict(te)


def fit_gb(tr, te, params):
    Xtr, _ = featurize(tr)
    Xte, _ = featurize(te)
    m = HistGradientBoostingRegressor(**params).fit(
        Xtr, tr.delta.to_numpy(float))
    return m.predict(Xte)


def select(tr, settings, fit):
    """Inner leave-one-family-out over `tr` only. Returns (best, table).

    Ties go to the first setting in grid order, which is fixed in advance.
    """
    fams = sorted(tr.family.unique())
    scored = []
    for s in settings:
        num = den = 0.0
        for g in fams:
            itr, iva = tr[tr.family != g], tr[tr.family == g]
            if itr.empty or iva.empty:
                continue
            num += len(iva) * mae(iva.delta.to_numpy(float), fit(itr, iva, s))
            den += len(iva)
        scored.append((num / den, s))
    best = min(range(len(scored)), key=lambda i: scored[i][0])
    return scored[best][1], scored[best][0]


def main():
    d = load(DATA)
    fams = sorted(d.family.unique())
    print(f"rows {len(d)} | outer folds {len(fams)}: {fams}")
    print(f"ridge grid {len(RIDGE_ALPHAS)} | gb grid {len(GB_GRID)}\n")

    res = {"outer_folds": [], "grid_sizes": {"ridge": len(RIDGE_ALPHAS),
                                             "grad_boost": len(GB_GRID)}}
    t0 = time.time()
    for f in fams:
        te, tr = d[d.family == f], d[d.family != f]
        y = te.delta.to_numpy(float)

        a_best, a_inner = select(tr, RIDGE_ALPHAS, fit_ridge)
        r_mae = mae(y, fit_ridge(tr, te, a_best))

        g_best, g_inner = select(tr, GB_GRID, fit_gb)
        g_mae = mae(y, fit_gb(tr, te, g_best))

        # the untuned settings, same folds, for the paired comparison
        r_fixed = mae(y, fit_ridge(tr, te, 3.0))
        g_fixed = mae(y, fit_gb(tr, te, dict(
            max_depth=3, max_iter=300, learning_rate=0.05,
            min_samples_leaf=15, l2_regularization=1.0, random_state=0)))

        res["outer_folds"].append({
            "held_out_family": f, "n_test": int(len(te)),
            "ridge_alpha_selected": a_best,
            "ridge_alpha_on_grid_edge": a_best in RIDGE_EDGES,
            "ridge_inner_mae": a_inner,
            "ridge_tuned_mae": r_mae, "ridge_fixed_mae": r_fixed,
            "gb_params_selected": {k: v for k, v in g_best.items()
                                   if k in ("learning_rate", "max_depth",
                                            "min_samples_leaf",
                                            "l2_regularization")},
            "gb_inner_mae": g_inner,
            "gb_tuned_mae": g_mae, "gb_fixed_mae": g_fixed,
        })
        print(f"  {f:<10} n={len(te):>3}  ridge alpha={a_best:<7} "
              f"tuned={r_mae:.4f} fixed={r_fixed:.4f}   "
              f"gb {g_best['learning_rate']}/{g_best['max_depth']}/"
              f"{g_best['min_samples_leaf']}/{g_best['l2_regularization']} "
              f"tuned={g_mae:.4f} fixed={g_fixed:.4f}   "
              f"[{time.time()-t0:.0f}s]")

    w = np.array([o["n_test"] for o in res["outer_folds"]], float)
    for key, fld in (("ridge_tuned", "ridge_tuned_mae"),
                     ("ridge_fixed", "ridge_fixed_mae"),
                     ("grad_boost_tuned", "gb_tuned_mae"),
                     ("grad_boost_fixed", "gb_fixed_mae")):
        v = np.array([o[fld] for o in res["outer_folds"]], float)
        res[f"weighted_mae::{key}"] = float((v * w).sum() / w.sum())

    res["ridge_any_grid_edge"] = any(o["ridge_alpha_on_grid_edge"]
                                     for o in res["outer_folds"])
    res["ridge_alphas"] = [o["ridge_alpha_selected"] for o in res["outer_folds"]]
    res["seconds"] = time.time() - t0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2, default=str)
    print("\nrow-weighted LOFO MAE:")
    for k in ("ridge_fixed", "ridge_tuned", "grad_boost_fixed",
              "grad_boost_tuned"):
        print(f"  {k:<20} {res['weighted_mae::' + k]:.6f}")
    print(f"\nwrote {OUT}  ({res['seconds']:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
