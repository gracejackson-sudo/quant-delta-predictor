"""
Diagnostics that decide WHICH failure we are looking at.

Q1  Is the lack of skill a ridge artifact? -> try a nonlinear model and a
    dead-simple scheme lookup under the same honest split.
Q2  Is scheme confounded with family, so that leave-family-out structurally
    destroys the config signal?
Q3  How big is the measurement-noise floor EMPIRICALLY (not just analytically)?
    Near-lossless schemes have true delta ~ 0, so their spread is ~ pure noise.
Q4  Which folds collapse, and why?
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(__file__))
from model import Predictor, featurize, load, noise_scale, r2  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")


def lofo_folds(d):
    for f in sorted(d.family.unique()):
        yield f, d[d.family != f], d[d.family == f]


def q1_model_class(d):
    print("== Q1: is 'no skill' a property of the data or of ridge? ==")
    print("   leave-one-family-out, MAE (lower is better)\n")
    rows = []
    for fam, tr, te in lofo_folds(d):
        y = te.delta.to_numpy(float)
        # global mean
        p_glob = np.full(len(te), tr.delta.mean())
        # per-benchmark mean
        bm = tr.groupby("benchmark").delta.mean()
        p_bench = te.benchmark.map(bm).fillna(tr.delta.mean()).to_numpy(float)
        # per-scheme mean  <- the dumbest thing that uses the config at all
        sm = tr.groupby("scheme").delta.mean()
        p_sch = te.scheme.map(sm).fillna(tr.delta.mean()).to_numpy(float)
        # per (scheme, benchmark) mean, backing off to scheme then global
        sbm = tr.groupby(["scheme", "benchmark"]).delta.mean()
        p_sb = np.array([
            sbm.get((s, b), sm.get(s, tr.delta.mean()))
            for s, b in zip(te.scheme, te.benchmark)
        ])
        # ridge
        m = Predictor(alpha=3.0).fit(tr)
        p_ridge = m.predict(te)
        # gradient boosting
        Xtr, _ = featurize(tr)
        Xte, _ = featurize(te)
        gb = HistGradientBoostingRegressor(
            max_depth=3, max_iter=300, learning_rate=0.05,
            min_samples_leaf=15, l2_regularization=1.0, random_state=0,
        ).fit(Xtr, tr.delta.to_numpy(float))
        p_gb = gb.predict(Xte)

        rows.append({
            "family": fam, "n": len(te),
            "global_mean": np.mean(np.abs(y - p_glob)),
            "bench_mean": np.mean(np.abs(y - p_bench)),
            "scheme_mean": np.mean(np.abs(y - p_sch)),
            "scheme_x_bench": np.mean(np.abs(y - p_sb)),
            "ridge": np.mean(np.abs(y - p_ridge)),
            "grad_boost": np.mean(np.abs(y - p_gb)),
        })
    t = pd.DataFrame(rows).set_index("family")
    print(t.round(3).to_string())
    n = t.pop("n")
    w = (t.mul(n, axis=0).sum() / n.sum()).sort_values()
    print("\n   row-weighted mean MAE across folds:")
    for k, v in w.items():
        print(f"     {k:<16} {v:.4f}")
    print("\n   -> any predictor that does not beat 'global_mean' has no "
          "usable signal out-of-family.")
    import json as _j
    _j.dump({"weighted": {k: float(v) for k, v in w.items()},
             "per_fold": t.assign(n=n).reset_index().to_dict("records")},
            open(os.path.join(OUT, "predictor_comparison.json"), "w"),
            indent=2, default=float)
    return {"per_fold": t.assign(n=n).reset_index().to_dict("records"),
            "weighted": w.to_dict()}


def q2_confounding(d):
    print("\n== Q2: is scheme confounded with family? ==\n")
    ct = pd.crosstab(d.family, d.scheme)
    print(ct.to_string())
    per_scheme_fams = (d.groupby("scheme").family.nunique()
                       .sort_values().rename("n_families"))
    print("\n   families per scheme:")
    print(per_scheme_fams.to_string())
    print("\n   -> a scheme present in only 1-2 families cannot be learned "
          "under leave-one-family-out.")
    return {"crosstab": ct.to_dict(), "families_per_scheme":
            per_scheme_fams.to_dict()}


def q3_empirical_noise(d):
    print("\n== Q3: empirical noise floor from near-lossless schemes ==\n")
    # W8A16 and FP8-dynamic are reported as effectively lossless, so their
    # observed delta spread is dominated by evaluation noise, not real damage.
    lossless = d[d.scheme.isin(["w8a16", "fp8_dynamic"])]
    print(f"   near-lossless rows: {len(lossless)} "
          f"(schemes w8a16, fp8_dynamic)")
    print(f"   their mean delta  : {lossless.delta.mean():+.3f}pp "
          f"(should be ~0 if truly lossless)")
    print(f"   their sd of delta : {lossless.delta.std():.3f}pp  "
          f"<- empirical noise estimate")
    print(f"   all-rows sd       : {d.delta.std():.3f}pp")
    emp = lossless.delta.std()
    share = min(1.0, (emp ** 2) / d.delta.var())
    print(f"   => measurement noise explains ~{share*100:.0f}% of the total "
          f"variance in delta")
    print(f"   analytic (independent-binomial) SE, mean: "
          f"{noise_scale(d).mean():.3f}pp  <- upper bound, exceeds observed "
          f"sd, confirming paired evals are correlated")

    print("\n   per-benchmark: empirical (near-lossless rows) vs analytic")
    rows = []
    for b, g in d.groupby("benchmark"):
        gl = lossless[lossless.benchmark == b]
        rows.append({
            "benchmark": b, "n_all": len(g), "sd_all": g.delta.std(),
            "n_lossless": len(gl),
            "sd_lossless": gl.delta.std() if len(gl) > 2 else np.nan,
            "analytic_se": noise_scale(g).mean(),
        })
    t = pd.DataFrame(rows).set_index("benchmark").sort_values("n_all",
                                                             ascending=False)
    print(t.round(3).to_string())
    return {"empirical_noise_sd_pp": float(emp),
            "noise_variance_share": float(share),
            "per_benchmark": t.reset_index().to_dict("records")}


def q4_worst_folds():
    print("\n== Q4: which folds collapse? ==\n")
    out = {}
    for tag in ("absolute", "normalized"):
        p = os.path.join(OUT, f"preds_{tag}_C_leave_family_out.csv")
        if not os.path.exists(p):
            continue
        r = pd.read_csv(p)
        g = (r.groupby("fold")
             .agg(n=("covered", "size"), coverage=("covered", "mean"))
             .sort_values("coverage"))
        print(f"   [{tag}] worst 6 of {len(g)} (test family | cal family):")
        print(g.head(6).round(3).to_string())
        byfam = (r.groupby("family")
                 .agg(n=("covered", "size"), coverage=("covered", "mean"))
                 .sort_values("coverage"))
        print(f"\n   [{tag}] coverage by TEST family:")
        print(byfam.round(3).to_string())
        byscheme = (r.groupby("scheme")
                    .agg(n=("covered", "size"), coverage=("covered", "mean"))
                    .sort_values("coverage"))
        print(f"\n   [{tag}] coverage by scheme:")
        print(byscheme.round(3).to_string())
        out[tag] = {"by_family": byfam.reset_index().to_dict("records"),
                    "by_scheme": byscheme.reset_index().to_dict("records"),
                    "worst_folds": g.head(6).reset_index().to_dict("records")}
        print()
    return out


def main():
    d = load(DATA)
    res = {}
    res["q1_model_class"] = q1_model_class(d)
    res["q2_confounding"] = q2_confounding(d)
    res["q3_noise"] = q3_empirical_noise(d)
    res["q4_worst_folds"] = q4_worst_folds()
    with open(os.path.join(OUT, "diagnostics.json"), "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"wrote {OUT}/diagnostics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
