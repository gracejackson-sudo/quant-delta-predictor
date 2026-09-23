"""
The actual feasibility experiment.

Three split regimes, deliberately reported side by side:

  A  row-level random split            -- LEAKY. Rows from the same model+config
                                         appear in train and calibration. This is
                                         what a naive implementation reports.
  B  leave-one-base-model-out          -- test model unseen; calibration drawn
                                         from other models (group-disjoint).
  C  leave-one-family-out, calibrated  -- test family unseen AND calibration
     on a different held-out family      family unseen. The stranger scenario.

See RESEARCH.md section 1 thread E for why the gap between A and C is the
headline number rather than a footnote.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from model import (  # noqa: E402
    Predictor, calibrate, evaluate, interval, load, r2, noise_scale,
)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")
ALPHA = 0.10  # 90% intervals
ALPHA_GRID = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def select_alpha(dtr):
    """Grouped CV inside the training split only -- no test/cal leakage."""
    groups = dtr.base_model.to_numpy()
    k = min(5, len(np.unique(groups)))
    if k < 2:
        return 3.0
    best, best_mae = 3.0, np.inf
    for a in ALPHA_GRID:
        errs = []
        for tr, va in GroupKFold(n_splits=k).split(dtr, groups=groups):
            m = Predictor(alpha=a).fit(dtr.iloc[tr])
            errs.append(np.abs(dtr.iloc[va].delta.to_numpy(float)
                               - m.predict(dtr.iloc[va])))
        mae = float(np.mean(np.concatenate(errs)))
        if mae < best_mae:
            best, best_mae = a, mae
    return best


def run_fold(dtr, dcal, dte, normalized):
    """Fit, calibrate, predict. Returns (per-row records, fold metrics)."""
    a = select_alpha(dtr)
    m = Predictor(alpha=a).fit(dtr)
    qhat = calibrate(m, dcal, ALPHA, normalized)
    yhat, lo, hi = interval(m, dte, qhat, normalized)
    met = evaluate(dte, yhat, lo, hi)
    met["alpha_ridge"] = a
    met["qhat"] = qhat
    met["mae_global_mean"] = float(np.mean(np.abs(
        dte.delta.to_numpy(float) - m.predict_global_mean(dte))))
    met["mae_bench_mean"] = float(np.mean(np.abs(
        dte.delta.to_numpy(float) - m.predict_bench_mean(dte))))
    recs = pd.DataFrame({
        "model": dte.model.to_numpy(), "family": dte.family.to_numpy(),
        "base_model": dte.base_model.to_numpy(),
        "scheme": dte.scheme.to_numpy(), "benchmark": dte.benchmark.to_numpy(),
        "acc_before": dte.acc_before.to_numpy(),
        "y": dte.delta.to_numpy(float), "yhat": yhat, "lo": lo, "hi": hi,
        "yhat_global_mean": m.predict_global_mean(dte),
        "yhat_bench_mean": m.predict_bench_mean(dte),
    })
    recs["covered"] = (recs.y >= recs.lo) & (recs.y <= recs.hi)
    return recs, met


def pooled(recs):
    """Pooled row-level metrics across all folds of a regime."""
    y, yhat = recs.y.to_numpy(), recs.yhat.to_numpy()
    return {
        "n_rows": int(len(recs)),
        "coverage": float(recs.covered.mean()),
        "mean_half_width": float(np.mean((recs.hi - recs.lo) / 2)),
        "median_half_width": float(np.median((recs.hi - recs.lo) / 2)),
        "mae": float(np.mean(np.abs(y - yhat))),
        "mae_global_mean": float(np.mean(np.abs(y - recs.yhat_global_mean))),
        "mae_bench_mean": float(np.mean(np.abs(y - recs.yhat_bench_mean))),
        "r2": r2(y, yhat),
        "r2_bench_mean": r2(y, recs.yhat_bench_mean.to_numpy()),
    }


# ------------------------------------------------------------------ regimes
def regime_A(d, normalized, seeds=200):
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(d))
        n1, n2 = int(0.5 * len(d)), int(0.8 * len(d))
        tr, ca, te = d.iloc[idx[:n1]], d.iloc[idx[n1:n2]], d.iloc[idx[n2:]]
        recs, _ = run_fold(tr, ca, te, normalized)
        recs["fold"] = s
        out.append(recs)
    return pd.concat(out, ignore_index=True)


def regime_B(d, normalized, seeds=5):
    out = []
    for bm in sorted(d.base_model.unique()):
        te = d[d.base_model == bm]
        rest = d[d.base_model != bm]
        others = sorted(rest.base_model.unique())
        if len(te) == 0 or len(others) < 4:
            continue
        for s in range(seeds):
            rng = np.random.default_rng(hash(bm) % 10_000 + s)
            perm = list(rng.permutation(others))
            n_cal = max(1, int(round(0.3 * len(perm))))
            cal_models = set(perm[:n_cal])
            ca = rest[rest.base_model.isin(cal_models)]
            tr = rest[~rest.base_model.isin(cal_models)]
            if len(ca) < 19 or len(tr) < 30:
                continue
            recs, _ = run_fold(tr, ca, te, normalized)
            recs["fold"] = f"{bm}|s{s}"
            out.append(recs)
    return pd.concat(out, ignore_index=True)


def regime_C(d, normalized):
    out = []
    fams = sorted(d.family.unique())
    for test_f in fams:
        for cal_f in fams:
            if cal_f == test_f:
                continue
            te = d[d.family == test_f]
            ca = d[d.family == cal_f]
            tr = d[~d.family.isin([test_f, cal_f])]
            if len(ca) < 19 or len(tr) < 30:
                continue
            recs, _ = run_fold(tr, ca, te, normalized)
            recs["fold"] = f"test={test_f}|cal={cal_f}"
            out.append(recs)
    return pd.concat(out, ignore_index=True)


def fmt(tag, p):
    return (f"{tag:<34} n={p['n_rows']:>5}  cov={p['coverage']*100:5.1f}%  "
            f"half-width={p['mean_half_width']:5.2f}pp  MAE={p['mae']:.3f}  "
            f"(mean-baseline {p['mae_global_mean']:.3f}, "
            f"bench-mean {p['mae_bench_mean']:.3f})  R2={p['r2']:+.3f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load(DATA)
    print(f"dataset: {len(d)} rows after dropping acc_before<20 | "
          f"{d.base_model.nunique()} base models | {d.family.nunique()} families")
    print(f"target: delta = acc_after - acc_before, "
          f"mean {d.delta.mean():+.3f}pp, sd {d.delta.std():.3f}pp\n")

    results, store = {}, {}
    for norm in (False, True):
        tag = "normalized" if norm else "absolute"
        print(f"===== nonconformity score: {tag} residuals "
              f"(alpha={ALPHA}, target coverage {100*(1-ALPHA):.0f}%) =====")
        for name, fn in (("A_row_random_LEAKY", regime_A),
                         ("B_leave_model_out", regime_B),
                         ("C_leave_family_out", regime_C)):
            recs = fn(d, norm)
            p = pooled(recs)
            per_fold = recs.groupby("fold").covered.mean()
            p["mean_per_fold_coverage"] = float(per_fold.mean())
            p["min_per_fold_coverage"] = float(per_fold.min())
            p["n_folds"] = int(per_fold.size)
            results[f"{tag}|{name}"] = p
            store[f"{tag}|{name}"] = recs
            print(fmt(name, p))
            print(f"{'':34} per-fold coverage: mean "
                  f"{p['mean_per_fold_coverage']*100:.1f}%, "
                  f"worst fold {p['min_per_fold_coverage']*100:.1f}% "
                  f"over {p['n_folds']} folds")
        print()

    # coefficients from a model fit on everything, for interpretation only
    m = Predictor(alpha=select_alpha(d)).fit(d)
    print("===== ridge coefficients (fit on all data, standardized) =====")
    for n, c in m.coefs()[:14]:
        print(f"  {n:<28} {c:+.4f}")
    results["_coefs_top"] = [[n, float(c)] for n, c in m.coefs()]
    results["_ridge_alpha_full"] = m.alpha

    # how much of the observed spread is just eval noise?
    ns = noise_scale(d)
    results["_noise"] = {
        "mean_analytic_delta_se_pp": float(ns.mean()),
        "sd_of_delta_pp": float(d.delta.std()),
        "implied_noise_share_of_variance": float(
            min(1.0, (ns ** 2).mean() / d.delta.var())),
    }
    print("\n===== noise floor =====")
    print(f"  mean analytic SE on a single delta : "
          f"{results['_noise']['mean_analytic_delta_se_pp']:.3f}pp")
    print(f"  observed sd of delta               : "
          f"{results['_noise']['sd_of_delta_pp']:.3f}pp")
    print(f"  implied share of variance that is measurement noise: "
          f"{results['_noise']['implied_noise_share_of_variance']*100:.0f}%")

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for k, v in store.items():
        v.to_csv(os.path.join(OUT, f"preds_{k.replace('|','_')}.csv"),
                 index=False)
    print(f"\nwrote {OUT}/results.json and per-row predictions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
