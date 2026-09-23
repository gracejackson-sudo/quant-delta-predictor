"""
TRACK 3 -- selection-bias-corrected asymmetric conformal intervals.

The problem: RedHatAI publishes quantization recipes that worked. Configs that
damaged a model were never released, so the observed delta distribution is
censored from below. Every interval we (and BenchPress) compute is therefore an
interval over *survivors*, and understates downside by an unknown amount.

THE IDENTIFICATION PROBLEM, stated up front because it governs everything:
if a fraction pi of attempted configs were never published, and censoring is on
the outcome, then the observed distribution G relates to the true F by

    F(x) = pi + (1 - pi) * G(x)

so the true q-quantile is  F^-1(q) = G^-1((q - pi)/(1 - pi))  for q > pi.
When q <= pi the true q-quantile lies below everything ever observed and is
NOT IDENTIFIED from published data alone: the Manski worst-case lower bound is
-infinity. For a 90% interval we need q = 0.05, so any censoring rate above 5%
puts the lower bound outside what published data can determine.

Two consequences:
  1. The correction cannot be estimated from the published corpus alone. It
     needs either an assumption about the tail shape, or external left-tail
     data (deliberately-bad configs we measure ourselves).
  2. The honest deliverable is therefore a SENSITIVITY ANALYSIS: report how
     far the lower bound moves as a function of assumed censoring rate pi,
     plus a tail-extrapolation estimate under a stated assumption.

This module implements both, and validates the estimator on synthetic data
where the true (uncensored) distribution is known.
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


# ----------------------------------------------------------------- estimator
def gpd_fit_lower_tail(x, threshold_q=0.25):
    """
    Peaks-over-threshold fit to the LEFT tail via method of moments.

    Work with exceedances y = u - x for x below threshold u, so the left tail
    becomes a right tail and standard GPD applies. Method of moments is used
    rather than MLE because these tails have 20-50 points and MLE is unstable
    there; the estimator is reported with its own caveat.
    """
    x = np.asarray(x, float)
    u = np.quantile(x, threshold_q)
    y = u - x[x < u]
    y = y[y > 0]
    if len(y) < 8:
        return None
    m, v = y.mean(), y.var(ddof=1)
    if v <= 0 or m <= 0:
        return None
    # MoM for GPD(shape xi, scale sigma)
    xi = 0.5 * (1 - m * m / v)
    sigma = 0.5 * m * (m * m / v + 1)
    xi = float(np.clip(xi, -0.5, 0.45))     # keep finite mean/variance
    return {"u": float(u), "xi": xi, "sigma": float(max(sigma, 1e-6)),
            "n_exceed": int(len(y)), "n_total": int(len(x)),
            "zeta": float(len(y) / len(x))}


def gpd_quantile(fit, p_tail, F_u):
    """
    x such that F(x) = p_tail, using a peaks-over-threshold tail for F:

        F(x) = F(u) * [1 + xi*(u - x)/sigma]^(-1/xi)     for x < u

    F_u is the mass of the TRUE distribution below the threshold u. Passing
    the observed exceedance fraction here instead of F_u was the bug caught by
    the synthetic check: it made a higher assumed censoring rate move the
    bound UP rather than down.
    """
    if fit is None or p_tail <= 0 or F_u <= 0 or p_tail >= F_u:
        return None
    xi, sig, u = fit["xi"], fit["sigma"], fit["u"]
    r = p_tail / F_u
    if abs(xi) < 1e-8:
        y = -sig * np.log(r)
    else:
        y = (sig / xi) * (r ** (-xi) - 1.0)
    return float(u - y)


def corrected_lower(observed, pi, alpha=ALPHA, threshold_q=0.25):
    """
    Lower bound of a (1-alpha) interval on the TRUE (uncensored) distribution,
    given an assumed censoring rate pi.

    q_true = alpha/2. Observed data covers true quantiles above pi, so:
      * if alpha/2 > pi: the bound is identified, map through G.
      * else: extrapolate with the fitted GPD (an assumption, flagged as such).
    """
    x = np.sort(np.asarray(observed, float))
    q = alpha / 2.0
    if pi <= 0:
        return float(np.quantile(x, q)), "empirical"
    if q > pi:
        # identified: the target quantile is still inside the observed support
        g = (q - pi) / (1.0 - pi)
        return float(np.quantile(x, g)), "reweighted"
    # not identified -- extrapolate F's left tail from the observed shape
    fit = gpd_fit_lower_tail(x, threshold_q)
    if fit is None:
        return float(x.min()), "unidentified_floor"
    # mass of the TRUE distribution below the fitting threshold u
    F_u = pi + (1.0 - pi) * threshold_q
    val = gpd_quantile(fit, q, F_u)
    if val is None:
        return float(x.min()), "unidentified_floor"
    return val, "gpd_extrapolated"


# ------------------------------------------------------------ synthetic check
def draw_truth(rng, n, scenario):
    """
    Two worlds, because the correction's validity depends entirely on which
    one we are in:

      'smooth'  -- censored configs are the continuation of the same tail.
                   The GPD continuation assumption HOLDS.
      'mixture' -- censored configs are a distinct failure mode (a second
                   population), not a continuation. The assumption FAILS.
    """
    if scenario == "smooth":
        # single skewed population: student-t scaled, left-skewed
        x = -np.abs(rng.standard_t(4, n)) * 0.9 - 0.3
        return x + rng.normal(0, 0.2, n)
    good = rng.normal(-0.3, 0.8, int(n * 0.85))
    bad = rng.normal(-3.5, 2.0, n - len(good))
    return np.concatenate([good, bad])


def synthetic_validation(n=4000, pi_true=0.15, seed=0, reps=200,
                         scenario="mixture"):
    """
    Does the estimator recover the true left tail when we KNOW the censoring?

    Ground truth: deltas from a left-skewed mixture (most configs fine, a
    minority genuinely damaging). Censor the worst pi_true fraction, hand the
    estimator only the survivors, and ask whether its corrected lower bound
    achieves nominal coverage against the FULL population.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(reps):
        truth = draw_truth(rng, n, scenario)
        cut = np.quantile(truth, pi_true)
        obs = truth[truth > cut]                 # survivorship
        naive_lo = float(np.quantile(obs, ALPHA / 2))
        for pi_assumed in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
            lo, how = corrected_lower(obs, pi_assumed)
            rows.append({
                "rep": rep, "pi_assumed": pi_assumed, "method": how,
                "lo": lo, "naive_lo": naive_lo,
                "true_q05": float(np.quantile(truth, ALPHA / 2)),
                # coverage of the LOWER side against the full population
                "below_lo_true": float((truth < lo).mean()),
                "below_naive_true": float((truth < naive_lo).mean()),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- reality
def apply_to_schemes(d, pis=(0.0, 0.05, 0.10, 0.15, 0.20, 0.30)):
    out = {}
    for s, g in d.groupby("scheme"):
        x = g.delta.to_numpy(float)
        fit = gpd_fit_lower_tail(x)
        rec = {"n": int(len(x)), "observed_min": float(x.min()),
               "observed_q05": float(np.quantile(x, 0.05)),
               "gpd": fit, "by_pi": {}}
        for pi in pis:
            lo, how = corrected_lower(x, pi)
            rec["by_pi"][str(pi)] = {"lower": lo, "method": how}
        out[s] = rec
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load(DATA)
    res = {"alpha": ALPHA}

    print("=" * 74)
    print("TRACK 3 -- SELECTION-BIAS-CORRECTED LOWER BOUNDS")
    print("=" * 74)
    print("\nIdentification: for a 90% interval the lower bound needs the 5th")
    print("percentile of the TRUE distribution. If more than 5% of attempted")
    print("configs went unpublished, that quantile lies below everything ever")
    print("observed and is not identified from published data alone.")
    print("What follows is therefore a sensitivity analysis plus a tail")
    print("extrapolation under a stated assumption -- not a point estimate.")

    print("\n" + "-" * 74)
    print("A. SYNTHETIC VALIDATION (censoring known, truth known)")
    print("-" * 74)
    res["synthetic"] = {}
    for scenario in ("smooth", "mixture"):
        sv_ = synthetic_validation(scenario=scenario)
        at_t = sv_[sv_.pi_assumed == 0.15]
        nv = sv_.below_naive_true.mean()
        res.setdefault("scenarios", {})[scenario] = {
            "naive_mass_below": float(nv),
            "corrected_mass_below_at_true_pi": float(
                at_t.below_lo_true.mean()),
            "target": ALPHA / 2,
            "fraction_of_gap_closed": float(
                (nv - at_t.below_lo_true.mean()) / (nv - ALPHA / 2))
            if nv > ALPHA / 2 else float("nan"),
        }
    print("\n   scenario comparison (true pi = 0.15, target mass 0.050):")
    print(f"   {'scenario':<12}{'naive':>10}{'corrected':>12}{'gap closed':>13}")
    for k, vv in res["scenarios"].items():
        print(f"   {k:<12}{vv['naive_mass_below']:>10.4f}"
              f"{vv['corrected_mass_below_at_true_pi']:>12.4f}"
              f"{100*vv['fraction_of_gap_closed']:>12.0f}%")
    sv = synthetic_validation(scenario="mixture")
    print(f"   true censoring rate: 15%   nominal lower-tail mass: "
          f"{ALPHA/2:.3f}")
    print(f"   {'pi assumed':>11}{'method':>20}{'mean lower':>12}"
          f"{'true mass below':>17}")
    for pi, g in sv.groupby("pi_assumed"):
        meth = g.method.mode().iloc[0]
        print(f"   {pi:>11.2f}{meth:>20}{g.lo.mean():>12.3f}"
              f"{g.below_lo_true.mean():>17.4f}")
        res["synthetic"][str(pi)] = {
            "method": meth, "mean_lower": float(g.lo.mean()),
            "true_mass_below": float(g.below_lo_true.mean())}
    naive = sv.below_naive_true.mean()
    print(f"\n   uncorrected (naive) bound leaves "
          f"{naive:.4f} of the TRUE population below it, "
          f"against a target of {ALPHA/2:.3f}")
    at_truth = res["synthetic"]["0.15"]["true_mass_below"]
    print(f"   corrected at the true pi=0.15 leaves {at_truth:.4f}")
    res["synthetic_naive_mass_below"] = float(naive)
    ok = abs(at_truth - ALPHA / 2) < abs(naive - ALPHA / 2)
    print(f"   -> correction moves the bound in the right direction: {ok}")
    res["synthetic_correction_helps"] = bool(ok)

    print("\n" + "-" * 74)
    print("B. APPLIED TO THE REAL PER-SCHEME DISTRIBUTIONS")
    print("-" * 74)
    sch = apply_to_schemes(d)
    res["schemes"] = sch
    print(f"   {'scheme':<13}{'n':>5}{'obs min':>9}{'pi=0':>8}{'pi=.05':>8}"
          f"{'pi=.10':>8}{'pi=.20':>8}{'pi=.30':>8}")
    for s in sorted(sch):
        r = sch[s]
        row = f"   {s:<13}{r['n']:>5}{r['observed_min']:>9.2f}"
        for pi in ("0.0", "0.05", "0.1", "0.2", "0.3"):
            key = pi if pi in r["by_pi"] else pi.rstrip("0")
            v_ = r["by_pi"].get(key, r["by_pi"].get(str(float(pi))))
            row += f"{v_['lower']:>8.2f}" if v_ else f"{'--':>8}"
        print(row)

    print("\n   GPD left-tail fits (shape xi > 0 means a heavy tail):")
    for s in sorted(sch):
        f_ = sch[s]["gpd"]
        if f_:
            print(f"     {s:<13} xi={f_['xi']:+.3f}  sigma={f_['sigma']:.3f}  "
                  f"exceedances={f_['n_exceed']}/{f_['n_total']}")
        else:
            print(f"     {s:<13} too few left-tail points to fit")

    print("\n" + "-" * 74)
    print("C. EXACT, ASSUMPTION-FREE RESULT (this is the part that holds)")
    print("-" * 74)
    print("   If a fraction pi of attempted configs were never published and")
    print("   censoring is on the outcome, the naive lower bound (the alpha/2")
    print("   quantile of survivors) leaves exactly")
    print("       pi + (1 - pi) * alpha/2")
    print("   of the TRUE population below it. No distributional assumption.")
    print(f"\n   {'assumed pi':>11}{'true lower-miss rate':>24}"
          f"{'effective interval':>21}")
    res["exact_degradation"] = {}
    for pi in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        miss = pi + (1 - pi) * ALPHA / 2
        eff = 1 - (miss + ALPHA / 2)
        print(f"   {pi:>11.2f}{miss:>24.4f}{eff*100:>20.1f}%")
        res["exact_degradation"][str(pi)] = {
            "lower_miss_rate": float(miss),
            "effective_two_sided": float(eff)}
    emp = res.get("synthetic_naive_mass_below")
    pred = 0.15 + 0.85 * ALPHA / 2
    print(f"\n   empirical check at pi=0.15: predicted {pred:.4f}, "
          f"measured {emp:.4f}  -> match: {abs(emp-pred) < 0.005}")
    res["exact_formula_validated"] = bool(abs(emp - pred) < 0.005)

    print("\n   Consequence: for a 90% interval the lower bound needs the 5th")
    print("   percentile. Any pi above 5% puts it outside the observed")
    print("   support, so a genuine 90% interval CANNOT be built from")
    print("   published data alone at that censoring rate.")

    with open(os.path.join(OUT, "bias_correction.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"\nwrote {OUT}/bias_correction.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
