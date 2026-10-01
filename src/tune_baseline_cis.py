"""Checkpoint-bootstrap CIs on the tuned-vs-untuned MAE differences.

Reads the hyperparameter selections already recorded in
out/tuned_baselines.json by src/tune_baselines.py and reuses them verbatim --
nothing is re-selected here, so this adds uncertainty quantification without
touching the pre-registered protocol. Merges the result back into the same
artifact.

The unit resampled is the checkpoint, matching every other interval in the
paper: rows from one quantization run are not independent evidence.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, os.path.dirname(__file__))
from model import Predictor, featurize, load  # noqa: E402

HERE = os.path.dirname(__file__)
ART = os.path.join(HERE, "..", "out", "tuned_baselines.json")
DRAWS, SEED = 20_000, 0


def main():
    d = load(os.path.join(HERE, "..", "data", "dataset.csv"))
    art = json.load(open(ART))
    sel = {o["held_out_family"]: (o["ridge_alpha_selected"],
                                 o["gb_params_selected"])
           for o in art["outer_folds"]}

    parts = []
    for f in sorted(d.family.unique()):
        te, tr = d[d.family == f], d[d.family != f]
        y = te.delta.to_numpy(float)
        alpha, gp = sel[f]
        Xtr, _ = featurize(tr)
        Xte, _ = featurize(te)
        gb = HistGradientBoostingRegressor(max_iter=300, random_state=0,
                                           early_stopping=False, **gp)
        pred = {
            "global_mean": np.full(len(te), tr.delta.mean()),
            "scheme_mean": te.scheme.map(
                tr.groupby("scheme").delta.mean()).fillna(
                    tr.delta.mean()).to_numpy(float),
            "ridge_tuned": Predictor(alpha=alpha).fit(tr).predict(te),
            "grad_boost_tuned": gb.fit(
                Xtr, tr.delta.to_numpy(float)).predict(Xte),
        }
        t = pd.DataFrame({k: np.abs(y - v) for k, v in pred.items()})
        t["base_model"] = te.base_model.to_numpy()
        parts.append(t)
    e = pd.concat(parts, ignore_index=True)

    clusters = [g for _, g in e.groupby("base_model")]
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(clusters), len(clusters)) for _ in range(DRAWS)]
    pooled = [pd.concat([clusters[j] for j in i]) for i in idx]

    out = {}
    for a, b, key in (("scheme_mean", "ridge_tuned", "scheme_vs_ridge_tuned"),
                      ("scheme_mean", "grad_boost_tuned", "scheme_vs_gb_tuned"),
                      ("global_mean", "ridge_tuned", "global_vs_ridge_tuned"),
                      ("global_mean", "scheme_mean", "global_vs_scheme")):
        dif = np.array([s[a].mean() - s[b].mean() for s in pooled])
        lo, hi = np.percentile(dif, 2.5), np.percentile(dif, 97.5)
        out[key] = {"estimate": float(e[a].mean() - e[b].mean()),
                    "lo": float(lo), "hi": float(hi),
                    "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"  {key:<24} {out[key]['estimate']:+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  "
              f"{'excludes 0' if out[key]['excludes_zero'] else 'includes 0'}")

    art["differences"] = out
    art["difference_draws"] = DRAWS
    art["difference_seed"] = SEED
    art["difference_clusters"] = len(clusters)
    json.dump(art, open(ART, "w"), indent=2, default=str)
    print(f"\nmerged into {ART}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
