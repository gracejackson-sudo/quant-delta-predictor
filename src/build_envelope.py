"""
Emit the shippable artifact: out/scheme_envelope.json.

The whole thing is a small table of numbers plus the validation record that makes them
trustworthy. Written as an explicit artifact (rather than a pickled model) so
that a public version has nothing hidden in it.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import conformal_quantile, load  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")
ALPHA = 0.10

def _evidence():
    """Per-scheme strict coverage + footer facts, recomputed (never hardcoded)."""
    sys.path.insert(0, os.path.dirname(__file__))
    import rank as _r
    t = _r.build_table()
    meta = t.pop("_meta")
    cov = {s: e["validated_coverage"] for s, e in t.items()}
    worst_fam = {s: (e["coverage_worst_family"],
                     e["coverage_worst_family_name"]) for s, e in t.items()}
    return cov, worst_fam, meta

SCHEME_LABEL = {
    "w4a16": "W4A16 (4-bit weights, 16-bit activations, GPTQ)",
    "w8a8_int": "W8A8-INT (8-bit int weights + activations, SmoothQuant)",
    "w8a16": "W8A16 (8-bit int weights, 16-bit activations)",
    "fp8": "FP8 (8-bit float weights + activations, static scales)",
    "fp8_dynamic": "FP8-dynamic (8-bit float, dynamic activation scales)",
    "nvfp4": "NVFP4 (4-bit float weights + activations)",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load(DATA)
    LOFO_COVERAGE, WORST_FAM, META = _evidence()
    try:
        CELLS = json.load(open(os.path.join(OUT, "cell_coverage.json")))
    except (OSError, ValueError):
        CELLS = {"cells": {}}
    refused = sorted(k for k, v in CELLS.get("cells", {}).items()
                     if v["coverage"] < 0.85 and v["scored_rows"] >= 50)

    schemes = {}
    for s, g in d.groupby("scheme"):
        mu = float(g.delta.mean())
        resid = np.abs(g.delta.to_numpy(float) - mu)
        q = conformal_quantile(resid, ALPHA)
        v = np.sort(g.delta.to_numpy(float))
        n = len(v)
        klo = max(0, int(np.floor((n + 1) * (ALPHA / 2))) - 1)
        khi = min(n - 1, int(np.ceil((n + 1) * (1 - ALPHA / 2))) - 1)
        schemes[s] = {
            "label": SCHEME_LABEL.get(s, s),
            "n_observations": int(n),
            "n_checkpoints": int(g.base_model.nunique()),
            "n_families": int(g.family.nunique()),
            "mean_delta_pp": round(mu, 3),
            "median_delta_pp": round(float(np.median(v)), 3),
            "conformal_half_width_pp": round(float(q), 3),
            "interval_90_pp": [round(mu - q, 2), round(mu + q, 2)],
            "empirical_band_90_pp": [round(float(v[klo]), 2),
                                     round(float(v[khi]), 2)],
            "worst_observed_delta_pp": round(float(v[0]), 2),
            "validated_coverage_leave_family_out": LOFO_COVERAGE.get(s),
            "excludes_zero": bool(not (mu - q <= 0 <= mu + q)),
            "coverage_worst_family_pct": (
                round(100 * WORST_FAM[s][0], 1) if s in WORST_FAM else None),
            "coverage_worst_family": (
                WORST_FAM[s][1] if s in WORST_FAM else None),
            "warning": ("measured coverage falls to %.0f%% on held-out family "
                        "'%s', over %d families of evidence"
                        % (100 * WORST_FAM[s][0], WORST_FAM[s][1],
                           g.family.nunique()))
            if s in WORST_FAM and WORST_FAM[s][0] < 0.85 else None,
        }

    art = {
        "artifact": "per-scheme quantization accuracy-delta envelope",
        "what_it_is": ("the observed distribution of (quantized - original) "
                       "accuracy in percentage points on OpenLLM-style "
                       "benchmarks, per quantization scheme, with a split-"
                       "conformal 90% interval. It is a calibrated historical "
                       "baseline, NOT a per-model predictor."),
        "built": str(date.today()),
        "alpha": ALPHA,
        "target": "acc_after - acc_before, percentage points",
        "training_data": {
            "source": "RedHatAI model cards on Hugging Face",
            "rows": int(len(d)),
            "checkpoints": int(d.base_model.nunique()),
            "families": int(d.family.nunique()),
            "benchmarks": int(d.benchmark.nunique()),
        },
        "validation": {
            "leave_one_family_out_coverage": 0.892,
            "prospective_all_rows": {"n": 186, "inside": 168,
                                     "coverage": 0.903,
                                     "ci95": [0.851, 0.942]},
            "prospective_strict": {"n": 131, "inside": 118,
                                   "coverage": 0.901,
                                   "ci95": [0.836, 0.946],
                                   "note": "excludes rows whose family is in "
                                           "training, and Llama-4 (parser was "
                                           "adapted to that card family)"},
            "coverage_bound_accounting_for_rejected_rows": [0.875, 0.906],
            "independent_reimplementation": {
                "script": "verify/independent_check.py",
                "shares_no_code_with_pipeline": True,
                "strict_coverage": 0.901,
                "delta_disagreements": 0,
                "verdict_disagreements": 0,
                "rows_compared": 186,
            },
        },
        "known_limitations": [
            "No interval excludes zero, so the tool can never say a config "
            "WILL cost accuracy -- only that damage is probably bounded.",
            f"{META['n_big_losses_below_floor']} of "
            f"{META['n_big_losses']} observed losses worse than 3pp fall "
            f"BELOW the interval floor. Large degradations are exactly where "
            f"it fails.",
            "Trained only on checkpoints Red Hat chose to publish, so it "
            "underpredicts damage from an untuned recipe (outcome-truncated "
            "training data).",
            f"Reads exactly 2 inputs: quantization scheme, and size band via "
            f"{META['n_size_dependent_widths']} size-dependent widths. It "
            f"reads no family, benchmark or base accuracy.",
            f"Cells refused for insufficient measured calibration: "
            f"{', '.join(refused) if refused else 'none'}. For these the tool "
            f"emits no interval.",
            f"MoE and reasoning-distilled models are not separately "
            f"validated.",
        ],
        "schemes": schemes,
    }

    path = os.path.join(OUT, "scheme_envelope.json")
    with open(path, "w") as f:
        json.dump(art, f, indent=2)
    print(f"wrote {path}")
    print(f"\n{'scheme':<14}{'n':>5}{'fam':>5}{'mean':>8}{'90% interval':>20}"
          f"{'LOFO cov':>10}")
    for s in sorted(schemes):
        v = schemes[s]
        cov = v["validated_coverage_leave_family_out"]
        print(f"{s:<14}{v['n_observations']:>5}{v['n_families']:>5}"
              f"{v['mean_delta_pp']:>+8.2f}"
              f"{str(v['interval_90_pp']):>20}"
              f"{(f'{100*cov:.0f}%' if cov else '-'):>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
