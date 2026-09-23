"""
Track 3, empirical: bias-corrected intervals using REAL adversarial anchors.

The earlier GPD extrapolation failed because it tried to infer the shape of a
population it had never seen (closed only 13-19% of the bias). We now have
that population: 27 measured w4a16 rows from deliberately-bad recipes.

Model. A user's recipe is either competent (probability 1-pi) or botched
(probability pi). Competent recipes behave like the published corpus.
Botched ones behave like the published corpus PLUS the excess damage we
measured, where

    excess = delta(bad recipe) - delta(correct recipe, same model, same
             benchmark, same harness)

Excess-over-control is used rather than raw delta because our control arm
measured -1.89pp where the published w4a16 mean is -0.73pp: our simplified
GPTQ is worse than production llm-compressor, and the harness differs
(0-shot vs 5-shot MMLU, flagged by the validator on 10 rows). Differencing
against our own control cancels both.

The corrected lower bound is then the alpha/2 quantile of the mixture. This
is an ESTIMATE conditional on pi, not a point identification -- pi is still
unknown. What has changed is that the tail is now measured instead of assumed.
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
ADV = os.path.join(HERE, "..", "data", "adversarial", "adversarial_runs.csv")
OUT = os.path.join(HERE, "..", "out")
ALPHA = 0.10
PIS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)


def load_excess():
    a = pd.read_csv(ADV)
    a["delta"] = a.acc_after - a.acc_before
    ctrl = a[a.is_control == 1].set_index(["base_model", "benchmark"]).delta
    a["ctrl"] = [ctrl.get((b, bm), np.nan)
                 for b, bm in zip(a.base_model, a.benchmark)]
    a["excess"] = a.delta - a.ctrl
    bad = a[(a.is_control == 0) & a.excess.notna()]
    return a, bad


def corrected_quantile(published, excess, pi, alpha=ALPHA, n=200000, seed=0):
    """alpha/2 quantile of the competent/botched mixture."""
    rng = np.random.default_rng(seed)
    pub = np.asarray(published, float)
    exc = np.asarray(excess, float)
    draw = rng.choice(pub, n, replace=True)
    botched = rng.random(n) < pi
    if botched.any() and len(exc):
        draw[botched] += rng.choice(exc, int(botched.sum()), replace=True)
    return float(np.quantile(draw, alpha / 2))


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load(DATA)
    a, bad = load_excess()
    res = {"alpha": ALPHA}

    print("=" * 74)
    print("EMPIRICAL BIAS CORRECTION (real adversarial anchors)")
    print("=" * 74)

    ctrl = a[a.is_control == 1].delta
    print(f"\ncontrol arm (correct w4a16, our harness): n={len(ctrl)} "
          f"mean {ctrl.mean():+.2f}pp")
    print(f"published w4a16 mean: {d[d.scheme=='w4a16'].delta.mean():+.2f}pp")
    print(f"-> our 'good' baseline is {abs(ctrl.mean()-d[d.scheme=='w4a16'].delta.mean()):.2f}pp "
          f"worse; excess-over-control cancels this")

    print("\nexcess damage by recipe (w4a16 arms only):")
    w4 = bad[bad.scheme == "w4a16"]
    print(w4.groupby("recipe").excess.agg(["count", "mean", "min"]).round(2)
          .to_string())
    res["excess_by_recipe"] = w4.groupby("recipe").excess.agg(
        ["count", "mean", "min"]).round(4).to_dict("index")

    exc = w4.excess.to_numpy()
    res["excess_pool"] = {"n": int(len(exc)), "mean": float(exc.mean()),
                          "p05": float(np.quantile(exc, 0.05)),
                          "min": float(exc.min())}
    print(f"\npooled w4a16 excess: n={len(exc)} mean {exc.mean():+.2f}pp "
          f"5th pct {np.quantile(exc,0.05):+.2f}pp worst {exc.min():+.2f}pp")

    print("\n" + "-" * 74)
    print("CORRECTED LOWER BOUND for published w4a16, by assumed botch rate")
    print("-" * 74)
    pub = d[d.scheme == "w4a16"].delta.to_numpy()
    print(f"   {'pi':>6}{'corrected lower':>18}{'vs uncorrected':>17}")
    res["w4a16_by_pi"] = {}
    base_lo = None
    for pi in PIS:
        lo = corrected_quantile(pub, exc, pi)
        if base_lo is None:
            base_lo = lo
        print(f"   {pi:>6.2f}{lo:>17.2f}pp{lo-base_lo:>+16.2f}pp")
        res["w4a16_by_pi"][str(pi)] = float(lo)

    # ---- does it now catch the known miss?
    print("\n" + "-" * 74)
    print("DOES IT CATCH THE KNOWN FAILURE? gemma-3-1b-it W4A16")
    print("-" * 74)
    gem = [-3.03, -2.99, -2.90, -2.41, -1.34, 1.40]
    print(f"   real deltas: {gem}")
    res["gemma_catch"] = {}
    for pi in PIS:
        lo = res["w4a16_by_pi"][str(pi)]
        caught = sum(1 for x in gem if x >= lo)
        res["gemma_catch"][str(pi)] = {"lower": lo, "inside": caught,
                                       "n": len(gem)}
        print(f"   pi={pi:.2f}  lower={lo:+.2f}pp  "
              f"inside: {caught}/{len(gem)}")

    with open(os.path.join(OUT, "bias_correction_empirical.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nwrote {OUT}/bias_correction_empirical.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
