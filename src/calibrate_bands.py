"""Round item 1: replace the in-sample band with a calibrated one.

Runs the protocol fixed in CALIBRATION_PREREGISTRATION.md. Does not modify
the shipped path: `ConservativeStratified.fit()` stays callable and the
in-sample figures stay reproducible, because the paper has to be able to
quote what the old path gave.

Two constructions, primary designated in advance:

  PRIMARY   split conformal, checkpoint-blocked. One global partition of the
            38 checkpoints into fit (2/3) and calibration (1/3), drawn once
            from default_rng(0). Centres from the fit side, half-widths from
            the conformal quantile of residuals on the calibration side,
            residuals taken about the centre predict_interval() will use.
            Mechanism is strata.ConservativeStratified.fit_calibrated.

  SECONDARY jackknife+, leave-one-checkpoint-out. For each non-test
            checkpoint i, a centre refit without i; the interval from the
            quantiles of {centre_-i -+ |residual_i|}. Uses every non-test
            checkpoint, at a 1-2*alpha guarantee (80% at alpha=0.10).

Evaluation is leave-one-checkpoint-out over all 38 checkpoints, so every row
is scored by a band that never saw its own checkpoint. The pre-registration
specified the fit/calibration partition and the evaluation unit but did not
spell out how they compose; a single global three-way split leaves no test
set at this n, so the nested form is the only one that yields a coverage
measurement. Resolved this way before any result was computed, and it is the
conservative resolution: evaluation units are seen by neither the centre nor
the width.

One reading recorded for the same reason. "Checkpoint-blocked" means the
partition respects checkpoint boundaries and evaluation is per checkpoint;
the conformal quantile index counts calibration ROWS, as fit_calibrated
already does. The nine-unit arithmetic in section 2 of the pre-registration
is about how many independent units a width rests on, which is why every
calibration set below is reported with both its row count and its checkpoint
count.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from model import load                                      # noqa: E402
from strata import (ALPHA, CAL_FRACTION, CAL_SEED,  # noqa: E402
                    ConservativeStratified, annotate, checkpoint_partition,
                    conformal_q)

ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "out", "calibrated_bands.json")
SEED = CAL_SEED
MIN_CAL_UNITS = 9          # ceil((n+1)(1-0.10)) <= n  =>  n >= 9


def _cov(y, lo, hi):
    e = 1e-9
    return float(((y >= lo - e) & (y <= hi + e)).mean())


def split_conformal_loco(d, cal_ckpts):
    """Primary. Returns per-row records."""
    rows = []
    for test in sorted(d.base_model.unique()):
        te = d[d.base_model == test]
        rest = d[d.base_model != test]
        dcal = rest[rest.base_model.isin(cal_ckpts)]
        dfit = rest[~rest.base_model.isin(cal_ckpts)]
        if dfit.empty or dcal.empty:
            continue
        m = ConservativeStratified().fit_calibrated(dfit, dcal)
        yhat, lo, hi, lv = m.predict_interval(te)
        for i, (_, r) in enumerate(te.iterrows()):
            rows.append({"model": r.model, "ckpt": test, "scheme": r.scheme,
                         "band": r.band, "delta": float(r.delta),
                         "lo": float(lo[i]), "hi": float(hi[i]),
                         "level": str(lv[i]),
                         "inside": bool(lo[i] - 1e-9 <= r.delta <= hi[i] + 1e-9)})
    return pd.DataFrame(rows)


def jackknife_plus_loco(d, alpha=ALPHA):
    """Secondary. Mondrian by scheme where the scheme has >= MIN_CAL_UNITS
    contributing checkpoints, else pooled."""
    rows = []
    all_ckpts = sorted(d.base_model.unique())
    for test in all_ckpts:
        te = d[d.base_model == test]
        rest = d[d.base_model != test]
        # leave-one-checkpoint-out residuals, each about a centre that never
        # saw that checkpoint
        recs = []
        for i in sorted(rest.base_model.unique()):
            inner_fit = rest[rest.base_model != i]
            held = rest[rest.base_model == i]
            # DELIBERATELY IN-SAMPLE on the inner fold: jackknife+ needs a
            # centre refit without checkpoint i, and its residual is taken
            # about that centre. Do not convert.
            m = ConservativeStratified().fit(inner_fit)
            yhat, _, _, _ = m.predict_interval(held)
            for j, (_, r) in enumerate(held.iterrows()):
                recs.append({"ckpt": i, "scheme": r.scheme,
                             "centre": float(yhat[j]),
                             "resid": abs(float(r.delta) - float(yhat[j]))})
        R = pd.DataFrame(recs)
        # DELIBERATELY IN-SAMPLE: the centre jackknife+ intervals are
        # reported about. Do not convert.
        m_full = ConservativeStratified().fit(rest)
        yhat_te, _, _, lv_te = m_full.predict_interval(te)
        for j, (_, r) in enumerate(te.iterrows()):
            sub = R[R.scheme == r.scheme]
            unit = "scheme"
            if sub.ckpt.nunique() < MIN_CAL_UNITS:
                sub, unit = R, "pooled"
            lo_terms = np.sort(sub.centre.to_numpy() - sub.resid.to_numpy())
            hi_terms = np.sort(sub.centre.to_numpy() + sub.resid.to_numpy())
            n = len(lo_terms)
            k_lo = int(np.floor(alpha * (n + 1)))
            k_hi = int(np.ceil((1 - alpha) * (n + 1)))
            lo = float(lo_terms[max(k_lo - 1, 0)])
            hi = float(hi_terms[min(k_hi - 1, n - 1)])
            rows.append({"model": r.model, "ckpt": test, "scheme": r.scheme,
                         "band": r.band, "delta": float(r.delta),
                         "lo": lo, "hi": hi, "level": unit,
                         "inside": bool(lo - 1e-9 <= r.delta <= hi + 1e-9)})
    return pd.DataFrame(rows)


def summarise(t, label, d):
    y = t.delta.to_numpy(float)
    lo = t.lo.to_numpy(float); hi = t.hi.to_numpy(float)
    out = {"label": label, "rows": int(len(t)),
           "checkpoints": int(t.ckpt.nunique()),
           "coverage_pct": 100 * _cov(y, lo, hi),
           "mean_half_width_pp": float(np.mean((hi - lo) / 2)),
           "per_scheme": {}, "vacuous_cells": []}
    for s, g in t.groupby("scheme"):
        rng_s = float(d[d.scheme == s].delta.max() - d[d.scheme == s].delta.min())
        hw = float(np.mean((g.hi - g.lo) / 2))
        out["per_scheme"][s] = {
            "rows": int(len(g)), "checkpoints": int(g.ckpt.nunique()),
            "coverage_pct": 100 * _cov(g.delta.to_numpy(float),
                                       g.lo.to_numpy(float),
                                       g.hi.to_numpy(float)),
            "mean_half_width_pp": hw,
            "observed_delta_range_pp": rng_s,
            "levels": sorted(g.level.unique().tolist()),
        }
        if hw > rng_s:
            out["vacuous_cells"].append(s)
    return out


def main():
    d = annotate(load(os.path.join(ROOT, "data", "dataset.csv")))
    cal, fit = checkpoint_partition(d)
    res = {"alpha": ALPHA, "seed": SEED, "cal_fraction": CAL_FRACTION,
           "n_rows": int(len(d)), "n_checkpoints": int(d.base_model.nunique()),
           "partition": {"calibration_checkpoints": sorted(cal),
                         "fit_checkpoints": sorted(fit)}}

    # what the shipped in-sample path gives, for the A/B
    # DELIBERATELY IN-SAMPLE: the retracted band, recorded here as the
    # baseline the split-conformal comparison is measured against. This
    # script exists to contrast the two constructions. Do not convert.
    m_ins = ConservativeStratified().fit(d)
    res["in_sample_half_width_pp"] = {
        s: float(c["half_width"]) for s, c in m_ins.by_scheme.items()}

    # per-scheme calibration support, reported with both counts
    c_rows = d[d.base_model.isin(cal)]
    res["calibration_support"] = {
        s: {"rows": int(len(g)), "checkpoints": int(g.base_model.nunique())}
        for s, g in c_rows.groupby("scheme")}

    print("partition: %d calibration checkpoints, %d fit checkpoints"
          % (len(cal), len(fit)), flush=True)
    print("calibration support per scheme (rows / checkpoints):", flush=True)
    for s, v in sorted(res["calibration_support"].items()):
        print("  %-12s %3d rows  %2d ckpts%s"
              % (s, v["rows"], v["checkpoints"],
                 "   <-- below the 9-unit floor" if v["checkpoints"] < MIN_CAL_UNITS else ""),
              flush=True)

    print("\nprimary: split conformal, checkpoint-blocked ...", flush=True)
    t1 = split_conformal_loco(d, cal)
    res["split_conformal"] = summarise(t1, "split conformal (checkpoint-blocked)", d)
    print("  coverage %.2f%% on %d rows, mean half-width %.3fpp"
          % (res["split_conformal"]["coverage_pct"],
             res["split_conformal"]["rows"],
             res["split_conformal"]["mean_half_width_pp"]), flush=True)

    print("\nsecondary: jackknife+, leave-one-checkpoint-out ...", flush=True)
    t2 = jackknife_plus_loco(d)
    res["jackknife_plus"] = summarise(t2, "jackknife+ (leave-one-checkpoint-out)", d)
    print("  coverage %.2f%% on %d rows, mean half-width %.3fpp"
          % (res["jackknife_plus"]["coverage_pct"],
             res["jackknife_plus"]["rows"],
             res["jackknife_plus"]["mean_half_width_pp"]), flush=True)

    # The twenty-seed sensitivity is part of the pre-registered protocol and
    # four registry keys read it, but partition_sensitivity() was defined and
    # never called from main() in any revision -- so running this script
    # dropped the block and paper/gen_numbers.py then exited on four missing
    # keys. Called by hand it reproduced the committed figures exactly, which
    # means the paper's numbers were right and the pipeline that claimed to
    # produce them was not. It is called here.
    print("\n20-partition sensitivity (pre-registered; one run, cannot move "
          "the headline) ...", flush=True)
    res["partition_sensitivity"] = partition_sensitivity(d)

    json.dump(res, open(OUT, "w"), indent=2, default=str)
    t1.to_csv(os.path.join(ROOT, "out", "calibrated_split_conformal.csv"), index=False)
    t2.to_csv(os.path.join(ROOT, "out", "calibrated_jackknife_plus.csv"), index=False)
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def partition_sensitivity(d, seeds=range(20)):
    """Pre-registered: run ONCE, after the primary result is recorded. It is a
    spread, and it cannot change which partition's numbers are the headline."""
    ckpts = sorted(d.base_model.unique())
    n_cal = int(round(len(ckpts) * CAL_FRACTION))
    out = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        cal = set(rng.choice(ckpts, size=n_cal, replace=False).tolist())
        t = split_conformal_loco(d, cal)
        y = t.delta.to_numpy(float)
        out.append({"seed": int(sd),
                    "coverage_pct": 100 * _cov(y, t.lo.to_numpy(float),
                                               t.hi.to_numpy(float)),
                    "mean_half_width_pp": float(np.mean((t.hi - t.lo) / 2))})
        print("  seed %2d  coverage %.2f%%  half-width %.3fpp"
              % (sd, out[-1]["coverage_pct"], out[-1]["mean_half_width_pp"]),
              flush=True)
    return out
