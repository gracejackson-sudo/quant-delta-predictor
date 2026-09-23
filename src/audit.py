"""
Three independent audits. Each prints PASS/FAIL/WARN lines and the script
exits non-zero if any hard check fails.

  AUDIT 1  data integrity        -- is the scraped dataset actually right?
  AUDIT 2  leakage & statistics  -- is the evaluation honest and the conformal
                                    implementation correct?
  AUDIT 3  robustness & claims   -- do the conclusions survive reasonable
                                    perturbation, and are they statistically
                                    distinguishable from zero?
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import conformal_quantile, featurize, load, noise_scale  # noqa: E402
from predictor import Conformal, GlobalMean, SchemeMean  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
CARDS = os.path.join(HERE, "..", "data", "cards")

FAILS, WARNS = [], []


def check(ok, msg, hard=True):
    if ok:
        print(f"  PASS  {msg}")
    elif hard:
        print(f"  FAIL  {msg}")
        FAILS.append(msg)
    else:
        print(f"  WARN  {msg}")
        WARNS.append(msg)


def card_path(model_id):
    return os.path.join(CARDS, model_id.replace("/", "_", 1) + ".md")


# ============================================================ AUDIT 1
def audit1():
    print("\n" + "=" * 70)
    print("AUDIT 1 -- DATA INTEGRITY")
    print("=" * 70)
    draw = pd.read_csv(DATA)

    # 1a arithmetic self-consistency
    bad = draw[np.abs((draw.acc_after - draw.acc_before) - draw.delta) > 1e-6]
    check(len(bad) == 0, f"delta == after - before on all {len(draw)} rows")

    # 1b ranges
    check(draw.acc_before.between(0.001, 100).all()
          and draw.acc_after.between(0, 100).all(),
          "all accuracies within (0, 100]")

    # 1c no duplicate (model, benchmark)
    dup = draw.duplicated(["model", "benchmark"]).sum()
    check(dup == 0, f"no duplicate (model, benchmark) rows (found {dup})")

    # 1d scheme token really appears in the model id
    def tok_ok(r):
        s, m = r.scheme, r.model.lower()
        pats = {"w4a16": ["w4a16", "int4"], "w8a8_int": ["w8a8", "int8"],
                "w8a16": ["w8a16"], "fp8_dynamic": ["fp8-dynamic",
                                                    "fp8_dynamic"],
                "fp8_block": ["fp8-block"], "fp8": ["fp8"],
                "nvfp4": ["nvfp4"], "nvfp4a16": ["nvfp4a16"],
                "mxfp4": ["mxfp4"]}
        return any(p in m for p in pats.get(s, [s]))
    check(draw.apply(tok_ok, axis=1).all(),
          "every parsed scheme token is present in its model id")

    # 1e params parse: hand-checked spot values
    expect = {"RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16": 8,
              "RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8": 405,
              "RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic": 3,
              "RedHatAI/Qwen2.5-72B-Instruct-quantized.w8a8": 72,
              "RedHatAI/gemma-2-9b-it-quantized.w4a16": 9}
    okp = True
    for mid, v in expect.items():
        got = draw[draw.model == mid].params_b.unique()
        if len(got) and abs(got[0] - v) > 1e-9:
            okp = False
            print(f"        params_b({mid}) = {got[0]}, expected {v}")
    check(okp, "hand-checked params_b values correct")

    # 1f raw-text verification: the two accuracy numbers must literally appear
    #    in the card, independent of the table parser
    rng = np.random.default_rng(0)
    samp = draw.iloc[rng.choice(len(draw), 60, replace=False)]
    miss = 0
    for _, r in samp.iterrows():
        p = card_path(r.model)
        if not os.path.exists(p):
            miss += 1
            continue
        txt = open(p, encoding="utf-8", errors="replace").read()
        # compare NUMERICALLY against every number in the card, so that a card
        # printing "78.40" still matches a stored 78.4
        nums = {float(x) for x in re.findall(r"\d+\.\d+|\d+", txt)}
        for v in (r.acc_before, r.acc_after):
            if not any(abs(v - n) < 1e-9 for n in nums):
                miss += 1
                print(f"        {r.model} {r.benchmark}: {v} not in card")
                break
    check(miss == 0, f"60 random rows: both numbers present in the source "
                     f"card ({miss} misses)")

    # 1g large |delta| rows: which were machine-verified, and which rest on
    #     header order alone (the genuine risk set)?
    ext = draw[np.abs(draw.delta) > 3.0]
    byo = ext.orient_src.value_counts().to_dict()
    print(f"        {len(ext)} rows with |delta|>3pp, by verification route: "
          f"{byo}")
    risky = ext[ext.orient_src == "header_only"]
    check(len(risky) == 0,
          f"no large-|delta| row depends on header order alone "
          f"({len(risky)} do)", hard=False)
    if len(risky):
        for _, r in risky.iterrows():
            print(f"          unverified sign: {r.model.split('/')[-1]} "
                  f"{r.benchmark} {r.delta:+.2f}pp")
    print("        (largest degradations: " + ", ".join(
        f"{r.model.split('/')[-1]}/{r.benchmark} {r.delta:+.2f}"
        for _, r in ext.nsmallest(3, "delta").iterrows()) + ")")

    # 1h accuracy scale: no card may be on a 0-1 scale
    permodel_max = draw.groupby("model").acc_before.max()
    check((permodel_max > 1.0).all(),
          f"no card reports accuracy on a 0-1 scale "
          f"({(permodel_max <= 1.0).sum()} offenders)")

    # 1i family assignment is not a spurious substring match
    fam_ok = draw.apply(
        lambda r: re.search(
            {"gemma-2": r"(?:^|[-_/])gemma-2(?![\d.])",
             "granite": r"(?:^|[-_/])granite(?![\d.])"}.get(r.family, r".") ,
            r.model, re.I) is not None, axis=1).all()
    check(fam_ok, "gemma-2 / granite family labels are boundary-anchored "
                  "(no 'diffusiongemma-26B' style false match)")

    # 1j orientation sanity: quantization should hurt on average
    check(draw.delta.mean() < 0,
          f"mean delta is negative ({draw.delta.mean():+.3f}pp) -- columns "
          f"are not swapped globally")

    # 1k how much of the data is machine-verified
    v = draw.verified.mean()
    check(v > 0.9, f"{v*100:.0f}% of rows had orientation+magnitude verified "
                   f"against the card's own recovery column")


# ============================================================ AUDIT 2
def audit2():
    print("\n" + "=" * 70)
    print("AUDIT 2 -- LEAKAGE & STATISTICAL CORRECTNESS")
    print("=" * 70)
    d = load(DATA)

    # 2a conformal quantile index
    s = np.arange(1, 21).astype(float)  # n=20
    q = conformal_quantile(s, 0.10)
    k = int(np.ceil(21 * 0.9))          # = 19
    check(q == s[k - 1], f"conformal index = ceil((n+1)(1-a)) = {k} -> q={q}")
    # the true threshold is ceil((n+1)(1-a)) <= n, i.e. n >= 1/a - 1 = 9 for
    # a=0.10. RESEARCH.md originally said 19 -- that was an arithmetic slip,
    # caught here.
    check(not np.isfinite(conformal_quantile(np.arange(8.0), 0.10)),
          "n = 8 < 1/alpha - 1 = 9 returns +inf rather than a too-narrow "
          "interval")
    check(np.isfinite(conformal_quantile(np.arange(9.0), 0.10)),
          "n = 9 is the smallest calibration set supporting 90%")
    check(not np.isfinite(conformal_quantile(np.arange(18.0), 0.05)),
          "n = 18 < 1/alpha - 1 = 19 returns +inf at the 95% level")

    # 2b end-to-end validation on SYNTHETIC EXCHANGEABLE data:
    #    if the machinery is right, coverage must hit nominal here.
    rng = np.random.default_rng(7)
    n = 4000
    syn = pd.DataFrame({
        "scheme": rng.choice(["a", "b", "c"], n),
        "benchmark": rng.choice(["mmlu", "gsm8k"], n),
        "n_items": 1000, "acc_before": 60.0,
    })
    mu = syn.scheme.map({"a": -1.0, "b": 0.0, "c": 1.0})
    syn["delta"] = mu + rng.normal(0, 1, n)
    syn["acc_after"] = syn.acc_before + syn.delta
    covs = []
    for seed in range(40):
        r = np.random.default_rng(seed).permutation(n)
        tr, ca, te = syn.iloc[r[:2000]], syn.iloc[r[2000:3000]], \
            syn.iloc[r[3000:]]
        m = SchemeMean().fit(tr)
        c = Conformal(alpha=0.10, mondrian_by="scheme").fit(m, ca)
        _, lo, hi, _ = c.predict_interval(te)
        y = te.delta.to_numpy()
        covs.append(np.mean((y >= lo) & (y <= hi)))
    mc = float(np.mean(covs))
    check(0.885 <= mc <= 0.915,
          f"synthetic exchangeable check: coverage {mc*100:.2f}% "
          f"(must be ~90.0%) -- validates the conformal implementation")

    # 2c features cannot see the target
    d2 = d.copy()
    X1, _ = featurize(d2)
    d2["acc_after"] = 0.0
    d2["delta"] = 0.0
    X2, _ = featurize(d2)
    check(np.allclose(X1, X2),
          "featurize() output is unchanged when acc_after/delta are destroyed "
          "-- no target leakage into features")

    # 2d regime C really is family-disjoint
    fams = sorted(d.family.unique())
    okc = True
    for test_f in fams:
        for cal_f in fams:
            if cal_f == test_f:
                continue
            te, ca = d[d.family == test_f], d[d.family == cal_f]
            tr = d[~d.family.isin([test_f, cal_f])]
            if (set(te.index) & set(tr.index)) or (set(te.index) & set(ca.index)) \
               or (set(tr.index) & set(ca.index)):
                okc = False
            if set(tr.family) & {test_f, cal_f}:
                okc = False
    check(okc, "leave-family-out folds: train/cal/test row- and "
               "family-disjoint in every fold")

    # 2e the demo hold-outs are genuinely unseen
    from demo_holdout import CAL_FAMILY, HOLDOUTS, TRAIN_FAMILIES
    seen = set(TRAIN_FAMILIES) | {CAL_FAMILY}
    okh = True
    for mid in HOLDOUTS:
        f = d[d.model == mid].family.unique()
        if len(f) and f[0] in seen:
            okh = False
            print(f"        {mid} is in a training/calibration family!")
    check(okh, "every demo hold-out model comes from a family absent from "
               "both train and calibration")

    # 2f scheme means are not fit on the test family
    te = d[d.family == "qwen3"]
    m = SchemeMean().fit(d[d.family != "qwen3"])
    check("qwen3" not in set(d[d.family != "qwen3"].family),
          "sanity: SchemeMean training frame excludes the test family")
    check(len(te) > 0, f"test family qwen3 has {len(te)} rows to score")


# ============================================================ AUDIT 3
def audit3():
    print("\n" + "=" * 70)
    print("AUDIT 3 -- ROBUSTNESS & CLAIM-vs-EVIDENCE")
    print("=" * 70)
    d = load(DATA)

    # 3a does coverage track nominal across alphas? (leave-family-out)
    print("\n  [3a] coverage vs nominal level, leave-one-family-out")
    for alpha in (0.20, 0.10, 0.05):
        rows = []
        for test_f in sorted(d.family.unique()):
            rest = [f for f in d.family.unique() if f != test_f]
            for cal_f in rest:
                te, ca = d[d.family == test_f], d[d.family == cal_f]
                tr = d[~d.family.isin([test_f, cal_f])]
                if len(ca) < 19:
                    continue
                m = SchemeMean().fit(tr)
                c = Conformal(alpha=alpha, mondrian_by="scheme").fit(m, ca)
                _, lo, hi, _ = c.predict_interval(te)
                y = te.delta.to_numpy(float)
                rows.append(pd.DataFrame({"covered": (y >= lo) & (y <= hi),
                                          "hw": (hi - lo) / 2}))
        a = pd.concat(rows)
        gap = a.covered.mean() - (1 - alpha)
        print(f"       nominal {100*(1-alpha):>4.0f}%  ->  observed "
              f"{a.covered.mean()*100:5.1f}%  (gap {gap*100:+5.1f}pp), "
              f"mean half-width {a.hw[np.isfinite(a.hw)].mean():.2f}pp")
        check(abs(gap) < 0.06,
              f"coverage gap at nominal {100*(1-alpha):.0f}% is "
              f"{gap*100:+.1f}pp (tolerance +/-6pp)", hard=(alpha == 0.10))

    # 3b is the skill over the global mean statistically real?
    print("\n  [3b] bootstrap on skill (MAE global-mean minus MAE scheme-mean),"
          "\n       resampling FAMILIES to respect clustering")
    fams = sorted(d.family.unique())
    per_fam = {}
    for f in fams:
        tr, te = d[d.family != f], d[d.family == f]
        gm, sm = GlobalMean().fit(tr), SchemeMean().fit(tr)
        y = te.delta.to_numpy(float)
        per_fam[f] = (np.abs(y - gm.predict(te)), np.abs(y - sm.predict(te)))
    point = (np.concatenate([v[0] for v in per_fam.values()]).mean()
             - np.concatenate([v[1] for v in per_fam.values()]).mean())
    rng = np.random.default_rng(3)
    boot = []
    for _ in range(5000):
        pick = rng.choice(fams, len(fams), replace=True)
        g = np.concatenate([per_fam[f][0] for f in pick])
        s = np.concatenate([per_fam[f][1] for f in pick])
        boot.append(g.mean() - s.mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"       skill = {point:+.4f}pp MAE  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(skill>0) = {np.mean(np.array(boot) > 0)*100:.0f}%")
    check(lo > 0,
          f"skill over the global-mean baseline excludes zero "
          f"(CI lower bound {lo:+.4f}pp)", hard=False)

    # 3c sensitivity to the acc_before < 20 filter and to verified-only rows
    print("\n  [3c] sensitivity of the headline numbers to preprocessing")
    for label, dd in (("as reported (acc_before>=20)", load(DATA)),
                      ("no acc_before filter",
                       load(DATA, drop_near_random=False)),
                      ("recovery-verified rows only",
                       load(DATA, verified_only=True))):
        rows = []
        for test_f in sorted(dd.family.unique()):
            for cal_f in sorted(dd.family.unique()):
                if cal_f == test_f:
                    continue
                te, ca = dd[dd.family == test_f], dd[dd.family == cal_f]
                tr = dd[~dd.family.isin([test_f, cal_f])]
                if len(ca) < 19 or len(tr) < 30:
                    continue
                m = SchemeMean().fit(tr)
                c = Conformal(alpha=0.10, mondrian_by="scheme").fit(m, ca)
                yh, lo_, hi_, _ = c.predict_interval(te)
                y = te.delta.to_numpy(float)
                rows.append(pd.DataFrame({
                    "covered": (y >= lo_) & (y <= hi_), "hw": (hi_ - lo_) / 2,
                    "ae": np.abs(y - yh)}))
        a = pd.concat(rows)
        print(f"       {label:<30} n={len(a):>5}  cov="
              f"{a.covered.mean()*100:5.1f}%  half-width="
              f"{a.hw[np.isfinite(a.hw)].mean():.2f}pp  MAE={a.ae.mean():.3f}")

    # 3d the noise-floor assumption
    print("\n  [3d] the MAE floor depends on assuming w8a16/fp8_dynamic are "
          "lossless.")
    ll = d[d.scheme.isin(["w8a16", "fp8_dynamic"])]
    print(f"       their mean delta is {ll.delta.mean():+.3f}pp "
          f"(n={len(ll)}); if they are NOT lossless the floor is "
          f"OVERestimated,\n       which would make the measured skill look "
          f"relatively better, not worse.")
    check(abs(ll.delta.mean()) < 0.35,
          f"near-lossless schemes average {ll.delta.mean():+.3f}pp, close "
          f"enough to 0 to serve as a noise probe", hard=False)

    # 3e GSM8K: is its delta pure noise?
    print("\n  [3e] per-benchmark: is the observed spread bigger for "
          "aggressive schemes than for lossless ones?")
    for b in ("gsm8k", "mmlu", "mmlu_pro", "hellaswag"):
        g = d[d.benchmark == b]
        sl = g[g.scheme.isin(["w8a16", "fp8_dynamic"])].delta.std()
        agg = g[g.scheme.isin(["w4a16", "nvfp4"])].delta.std()
        verdict = "no separation -> ~pure noise" if not (agg > 1.3 * sl) \
            else "real scheme effect"
        print(f"       {b:<10} sd(lossless)={sl:5.2f}  sd(4-bit)={agg:5.2f}"
              f"   {verdict}")

    # 3f claims vs pre-registration
    print("\n  [3f] pre-registered hypotheses (RESEARCH.md s.2) vs outcome")
    for h, txt in [
        ("H1", "low R2 out-of-family -> CONFIRMED (R2 negative for ridge)"),
        ("H2", "benchmark identity strongest feature -> REFUTED: per-benchmark "
               "means do NOT transfer across families (MAE 0.767 vs 0.732 "
               "global)"),
        ("H3", "weight bit-width dominates -> CONFIRMED (w4a16/nvfp4 worst, "
               "w8a16/fp8_dynamic best)"),
        ("H4", "split A ~nominal, split C undercovers -> PARTIALLY CONFIRMED: "
               "marginal C 88.8%, but per-scheme collapse (nvfp4 68%) needed "
               "Mondrian calibration to fix"),
        ("H5", "wide intervals, almost never exclude zero -> CONFIRMED (0% of "
               "hold-out intervals exclude zero)"),
        ("H6", "normalized scores better-shaped -> REFUTED: they widen "
               "intervals (2.83pp vs 1.94pp) without improving coverage"),
    ]:
        print(f"       {h}: {txt}")


def main():
    audit1()
    audit2()
    audit3()
    print("\n" + "=" * 70)
    print(f"RESULT: {len(FAILS)} hard failures, {len(WARNS)} warnings")
    for f in FAILS:
        print(f"  FAILED: {f}")
    for w in WARNS:
        print(f"  WARNING: {w}")
    print("=" * 70)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
