"""
The literal deliverable: train on 2-3 model families, then predict real
held-out (model, quant_config) pairs and check whether the true measured delta
falls inside the 90% interval.

Nothing about the held-out model is in training OR calibration.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load  # noqa: E402
from predictor import Conformal, SchemeMean  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")
ALPHA = 0.10

TRAIN_FAMILIES = ["llama-3.1", "qwen2.5"]   # the 2 biggest families
CAL_FAMILY = "granite"                       # held out for calibration only

# real held-out (model, config) pairs from families the predictor never saw
HOLDOUTS = [
    "RedHatAI/Mistral-Small-24B-Instruct-2501-FP8-dynamic",
    "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",
    "RedHatAI/gemma-2-9b-it-quantized.w4a16",
    "RedHatAI/Qwen3-8B-quantized.w4a16",
    "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load(DATA)

    tr = d[d.family.isin(TRAIN_FAMILIES)]
    ca = d[d.family == CAL_FAMILY]
    print(f"TRAIN       : families {TRAIN_FAMILIES} -> {len(tr)} rows, "
          f"{tr.base_model.nunique()} base models")
    print(f"CALIBRATION : family '{CAL_FAMILY}' -> {len(ca)} rows "
          f"(disjoint from train and from every test model)")

    m = SchemeMean().fit(tr)
    print("\nlearned scheme-mean table (percentage points):")
    for s, v in sorted(m.by_scheme.items(), key=lambda t: t[1]):
        print(f"   {s:<14} {v:+.3f}pp   (n={m.n_by_scheme[s]})")
    print(f"   {'<unseen>':<14} {m.global_mean:+.3f}pp   (global back-off)")

    conf = Conformal(alpha=ALPHA, mondrian_by="scheme").fit(m, ca)
    print("\nconformal 90% half-widths from calibration family:")
    for s in sorted(conf.q_group):
        q = conf.q_group[s]
        print(f"   {s:<14} +/-{q:5.2f}pp   (n_cal={conf.n_group[s]})")
    print(f"   marginal       +/-{conf.q_marginal:5.2f}pp")

    all_rows, summary = [], []
    for mid in HOLDOUTS:
        te = d[d.model == mid]
        if te.empty:
            print(f"\n!! {mid}: no rows, skipping")
            continue
        assert te.family.iloc[0] not in TRAIN_FAMILIES + [CAL_FAMILY], \
            "hold-out family leaked into train/cal"
        yhat, lo, hi, fb = conf.predict_interval(te)
        t = pd.DataFrame({
            "benchmark": te.benchmark.to_numpy(),
            "before": te.acc_before.to_numpy(),
            "after": te.acc_after.to_numpy(),
            "true_delta": te.delta.to_numpy(float),
            "pred": yhat, "lo": lo, "hi": hi,
        })
        t["inside_90"] = (t.true_delta >= t.lo) & (t.true_delta <= t.hi)
        t.insert(0, "model", mid)
        all_rows.append(t)

        cov = t.inside_90.mean()
        print(f"\n=== {mid}")
        print(f"    family '{te.family.iloc[0]}' (UNSEEN), scheme "
              f"'{te.scheme.iloc[0]}', {te.params_b.iloc[0]:g}B params")
        print(t.drop(columns=["model"]).to_string(
            index=False,
            formatters={"before": "{:.2f}".format, "after": "{:.2f}".format,
                        "true_delta": "{:+.2f}".format, "pred": "{:+.2f}".format,
                        "lo": "{:+.2f}".format, "hi": "{:+.2f}".format}))
        print(f"    -> {int(t.inside_90.sum())}/{len(t)} inside the 90% "
              f"interval ({cov*100:.0f}%)")
        summary.append({"model": mid, "family": te.family.iloc[0],
                        "scheme": te.scheme.iloc[0], "n": int(len(t)),
                        "n_inside": int(t.inside_90.sum()),
                        "coverage": float(cov)})

    a = pd.concat(all_rows, ignore_index=True)
    n_in, n = int(a.inside_90.sum()), len(a)
    print("\n" + "=" * 72)
    print(f"POOLED over {len(summary)} held-out models from "
          f"{a.model.nunique()} unseen (model, config) pairs:")
    print(f"  {n_in}/{n} true deltas inside the 90% interval = "
          f"{100*n_in/n:.1f}%  (nominal 90%)")
    lo_b, hi_b = binom_ci(n_in, n)
    print(f"  95% binomial CI on that coverage: "
          f"[{lo_b*100:.1f}%, {hi_b*100:.1f}%]")
    print(f"  mean interval half-width: "
          f"{np.mean((a.hi - a.lo) / 2):.2f}pp")
    print(f"  mean |true - pred|      : "
          f"{np.mean(np.abs(a.true_delta - a.pred)):.3f}pp")
    print(f"  intervals excluding zero: "
          f"{100*((a.lo > 0) | (a.hi < 0)).mean():.0f}%")

    a.to_csv(os.path.join(OUT, "holdout_demo.csv"), index=False)
    with open(os.path.join(OUT, "holdout_demo.json"), "w") as f:
        json.dump({"train_families": TRAIN_FAMILIES,
                   "cal_family": CAL_FAMILY,
                   "per_model": summary,
                   "pooled_n": n, "pooled_inside": n_in,
                   "pooled_coverage": n_in / n}, f, indent=2)
    print(f"\nwrote {OUT}/holdout_demo.csv")
    return 0


def binom_ci(k, n, conf=0.95):
    """Clopper-Pearson exact interval."""
    from scipy.stats import beta
    a = 1 - conf
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return float(lo), float(hi)


if __name__ == "__main__":
    sys.exit(main())
