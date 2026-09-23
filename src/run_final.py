"""
Final evaluation of the shipped predictor (scheme-mean + conformal).

Regimes:
  A  row-level random split                      LEAKY reference
  B  leave-one-base-model-out (group-disjoint cal)
  C  leave-one-family-out, calibrated on TWO other held-out families
     -> the stranger scenario: unseen model AND unseen calibration family

Also computes the irreducible MAE floor from evaluation noise, so that "no
skill" can be distinguished from "already at the physical limit".
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load, noise_scale, r2  # noqa: E402
from predictor import Conformal, GlobalMean, SchemeMean  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")
ALPHA = 0.10

LOSSLESS = ["w8a16", "fp8_dynamic"]  # treated as ~zero true delta (thread A)


def mae_floor(d):
    """
    Irreducible MAE: for each benchmark, estimate E|eval noise| directly as the
    mean absolute deviation of delta around its mean among near-lossless
    schemes (true delta ~ 0 there, so the spread is measurement noise). Pool
    across benchmarks weighted by how often each benchmark appears overall.
    No Gaussian assumption.
    """
    rows = []
    for b, g in d.groupby("benchmark"):
        gl = g[g.scheme.isin(LOSSLESS)]
        if len(gl) < 4:
            continue
        v = gl.delta.to_numpy(float)
        rows.append({"benchmark": b, "w": len(g), "n_lossless": len(gl),
                     "E_abs_noise": float(np.mean(np.abs(v - v.mean()))),
                     "sd_lossless": float(v.std(ddof=1)),
                     "sd_all": float(g.delta.std(ddof=1)),
                     "noise_var_share": float(min(1.0, v.var(ddof=1)
                                                  / max(g.delta.var(ddof=1),
                                                        1e-9)))})
    t = pd.DataFrame(rows)
    floor = float((t.E_abs_noise * t.w).sum() / t.w.sum())
    return floor, t


def one(model_cls, conf_kwargs, dtr, dcal, dte):
    m = model_cls().fit(dtr)
    c = Conformal(alpha=ALPHA, **conf_kwargs).fit(m, dcal)
    yhat, lo, hi, fb = c.predict_interval(dte)
    return pd.DataFrame({
        "model": dte.model.to_numpy(), "family": dte.family.to_numpy(),
        "base_model": dte.base_model.to_numpy(),
        "scheme": dte.scheme.to_numpy(), "benchmark": dte.benchmark.to_numpy(),
        "acc_before": dte.acc_before.to_numpy(),
        "acc_after": dte.acc_after.to_numpy(),
        "y": dte.delta.to_numpy(float), "yhat": yhat, "lo": lo, "hi": hi,
        "backoff": fb,
        "covered": (dte.delta.to_numpy(float) >= lo)
        & (dte.delta.to_numpy(float) <= hi),
    })


CONFIGS = {
    "marginal": dict(mondrian_by=None, normalized=False),
    "mondrian_scheme": dict(mondrian_by="scheme", normalized=False),
    "mondrian_scheme_normalized": dict(mondrian_by="scheme", normalized=True),
}


def regime_A(d, cfg, seeds=200):
    out = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        i = rng.permutation(len(d))
        n1, n2 = int(.5 * len(d)), int(.8 * len(d))
        r = one(SchemeMean, cfg, d.iloc[i[:n1]], d.iloc[i[n1:n2]],
                d.iloc[i[n2:]])
        r["fold"] = s
        out.append(r)
    return pd.concat(out, ignore_index=True)


def regime_B(d, cfg, seeds=5):
    out = []
    for bm in sorted(d.base_model.unique()):
        te, rest = d[d.base_model == bm], d[d.base_model != bm]
        others = sorted(rest.base_model.unique())
        for s in range(seeds):
            rng = np.random.default_rng(abs(hash(bm)) % 9973 + s)
            perm = list(rng.permutation(others))
            cal = set(perm[:max(1, int(round(.3 * len(perm))))])
            ca, tr = rest[rest.base_model.isin(cal)], \
                rest[~rest.base_model.isin(cal)]
            if len(ca) < 19 or len(tr) < 30:
                continue
            r = one(SchemeMean, cfg, tr, ca, te)
            r["fold"] = f"{bm}|s{s}"
            out.append(r)
    return pd.concat(out, ignore_index=True)


def regime_C(d, cfg):
    """Unseen test family; calibration on two OTHER unseen families."""
    out = []
    fams = sorted(d.family.unique())
    for test_f in fams:
        rest_f = [f for f in fams if f != test_f]
        for cal_pair in itertools.combinations(rest_f, 2):
            te = d[d.family == test_f]
            ca = d[d.family.isin(cal_pair)]
            tr = d[~d.family.isin((test_f,) + cal_pair)]
            if len(ca) < 19 or len(tr) < 30:
                continue
            r = one(SchemeMean, cfg, tr, ca, te)
            r["fold"] = f"test={test_f}|cal={'+'.join(cal_pair)}"
            out.append(r)
    return pd.concat(out, ignore_index=True)


def summarize(r, floor):
    y = r.y.to_numpy()
    fin = np.isfinite(r.hi - r.lo)
    per_fold = r.groupby("fold").covered.mean()
    return {
        "n_rows": int(len(r)),
        "coverage": float(r.covered.mean()),
        "mean_half_width_pp": float(np.mean(((r.hi - r.lo) / 2)[fin])),
        "median_half_width_pp": float(np.median(((r.hi - r.lo) / 2)[fin])),
        "pct_infinite_intervals": float(100 * (~fin).mean()),
        "pct_backoff": float(100 * r.backoff.mean()),
        "mae": float(np.mean(np.abs(y - r.yhat))),
        "mae_floor_from_eval_noise": floor,
        "r2": r2(y, r.yhat.to_numpy()),
        "n_folds": int(per_fold.size),
        "mean_per_fold_coverage": float(per_fold.mean()),
        "worst_fold_coverage": float(per_fold.min()),
        "worst_fold": str(per_fold.idxmin()),
        "pct_intervals_excluding_zero": float(
            100 * ((r.lo > 0) | (r.hi < 0)).mean()),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load(DATA)
    floor, ftab = mae_floor(d)

    print(f"dataset: {len(d)} rows | {d.base_model.nunique()} base models | "
          f"{d.family.nunique()} families | {d.scheme.nunique()} schemes")
    print(f"target delta: mean {d.delta.mean():+.3f}pp  sd {d.delta.std():.3f}pp")
    print(f"\nirreducible MAE floor from evaluation noise: {floor:.3f}pp")
    print("  (per-benchmark noise estimated from near-lossless schemes)")
    print(ftab.round(3).to_string(index=False))

    # baseline MAE for reference, same honest regime
    base_maes = []
    for test_f in sorted(d.family.unique()):
        tr, te = d[d.family != test_f], d[d.family == test_f]
        gm = GlobalMean().fit(tr)
        base_maes.append((len(te), float(np.mean(np.abs(
            te.delta.to_numpy(float) - gm.predict(te))))))
    mae_glob = sum(n * v for n, v in base_maes) / sum(n for n, _ in base_maes)
    print(f"\nglobal-mean baseline MAE, leave-one-family-out: {mae_glob:.4f}pp")

    results, store = {"_mae_floor_pp": floor,
                      "_mae_global_baseline_lofo": mae_glob,
                      "_noise_table": ftab.to_dict("records")}, {}

    for cname, cfg in CONFIGS.items():
        print(f"\n===== calibration: {cname} "
              f"(alpha={ALPHA}, nominal {100*(1-ALPHA):.0f}%) =====")
        for rname, fn in (("A_row_random_LEAKY", regime_A),
                          ("B_leave_model_out", regime_B),
                          ("C_leave_family_out", regime_C)):
            r = fn(d, cfg)
            s = summarize(r, floor)
            results[f"{cname}|{rname}"] = s
            store[f"{cname}|{rname}"] = r
            print(f"  {rname:<22} cov={s['coverage']*100:5.1f}%  "
                  f"half-width={s['mean_half_width_pp']:5.2f}pp  "
                  f"MAE={s['mae']:.3f} (floor {floor:.3f}, "
                  f"baseline {mae_glob:.3f})  "
                  f"worst fold {s['worst_fold_coverage']*100:4.1f}%")
            if cname != "marginal":
                by = r.groupby("scheme").covered.mean().sort_values()
                print("      per-scheme coverage: " + ", ".join(
                    f"{k}={v*100:.0f}%" for k, v in by.items()))

    with open(os.path.join(OUT, "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for k, v in store.items():
        v.to_csv(os.path.join(OUT, f"final_{k.replace('|','_')}.csv"),
                 index=False)
    print(f"\nwrote {OUT}/final_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
