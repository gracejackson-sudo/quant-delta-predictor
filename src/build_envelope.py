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
    """Per-scheme coverage records and the footer facts, recomputed by rank.py
    (never hardcoded here). The coverage VERDICTS, including which cells are
    refused, come from rank.classify_cell: this file used to apply its own,
    older rule, and kept naming two refused cells after the tool had stopped."""
    sys.path.insert(0, os.path.dirname(__file__))
    import rank as _r
    t = _r.build_table()
    meta = t.pop("_meta")
    return t, meta, _r


SCHEME_LABEL = {
    "w4a16": "W4A16 (4-bit weights, 16-bit activations, GPTQ)",
    "w8a8_int": "W8A8-INT (8-bit int weights + activations, SmoothQuant)",
    "w8a16": "W8A16 (8-bit int weights, 16-bit activations)",
    "fp8": "FP8 (8-bit float weights + activations, static scales)",
    "fp8_dynamic": "FP8-dynamic (8-bit float, dynamic activation scales)",
    "nvfp4": "NVFP4 (4-bit float weights + activations)",
}


def _prospective():
    """Strict and all-rows prospective coverage, recomputed from
    out/independent_check.csv (the same filter and interval as verify_claims)."""
    import re
    import pandas as pd
    from scipy.stats import beta
    p = os.path.join(OUT, "independent_check.csv")
    if not os.path.exists(p):
        return None
    ic = pd.read_csv(p)

    def cp(k, n):
        return [round(float(beta.ppf(0.025, k, n - k + 1)), 3),
                round(float(beta.ppf(0.975, k + 1, n - k)), 3)]
    strict = ic[~ic.model.map(lambda m: bool(
        re.search(r"Llama-3\.1", m.split("/")[-1], re.I)
        or re.search(r"(^|[-_])Qwen3(?![.\d])", m.split("/")[-1], re.I)
        or re.search(r"Llama-4", m.split("/")[-1], re.I)))]
    out = {}
    for name, g in (("prospective_all_rows", ic), ("prospective_strict", strict)):
        k, n = int(g.inside.sum()), int(len(g))
        out[name] = {"n": n, "inside": k, "coverage": round(k / n, 3), "ci95": cp(k, n)}
    return out


def build_artifact():
    d = load(DATA)
    TABLE, META, R = _evidence()
    try:
        CELLS = json.load(open(os.path.join(OUT, "cell_coverage.json")))
    except (OSError, ValueError):
        CELLS = {"cells": {}}
    refused = list(META["refused_cells"])
    insufficient = list(META["insufficient_evidence_cells"])
    sch_states = {k: e["coverage_state"] for k, e in TABLE.items()}
    sch_insufficient = sorted(k for k, v in sch_states.items() if v == "insufficient_evidence")
    sch_refused = sorted(k for k, v in sch_states.items() if v == "refused")

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
            # One-sided (does the true delta stay at or above the lower bound),
            # judged by the same classifier as every size cell.
            "coverage": {
                "state": TABLE[s]["coverage_state"],
                "one_sided": round(TABLE[s]["coverage_one_sided"], 4),
                "two_sided": round(TABLE[s]["coverage_two_sided"], 4),
                "bootstrap_90_pct": TABLE[s]["coverage_boot90"],
                "rows": TABLE[s]["coverage_rows"],
                "checkpoints": TABLE[s]["coverage_ckpts"],
                "scored_pairs": TABLE[s]["coverage_scored_pairs"],
                "worst_family": TABLE[s]["coverage_worst_family_name"],
                "worst_family_one_sided": round(TABLE[s]["coverage_worst_family"], 4),
                "by_size_band": TABLE[s]["band_verdicts"] if "band_verdicts" in TABLE[s] else R.band_verdicts(s, CELLS),
            },
            "excludes_zero": bool(not (mu - q <= 0 <= mu + q)),
            "warning": ("measured coverage falls to %.0f%% on held-out family "
                        "'%s', over %d families of evidence"
                        % (100 * TABLE[s]["coverage_worst_family"],
                           TABLE[s]["coverage_worst_family_name"],
                           g.family.nunique()))
            if TABLE[s]["coverage_worst_family"] < R.REFUSE_BELOW else None,
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
            "leave_one_family_out": {
                "rows": CELLS.get("pooled", {}).get("distinct_rows"),
                "calibration_families_per_row": CELLS.get("pooled", {}).get("pairs_per_row"),
                "evaluations": CELLS.get("pooled", {}).get("scored_pairs"),
                "coverage_two_sided": round(CELLS.get("pooled", {}).get("coverage", float("nan")), 4),
                "coverage_one_sided": round(CELLS.get("pooled", {}).get("coverage_one_sided", float("nan")), 4),
                "note": "each row is scored once per calibration family, so "
                        "evaluations are not independent rows",
            },
            **(_prospective() or {}),
            "prospective_strict_note": "excludes rows whose family is in "
                                       "training, and Llama-4 (parser was "
                                       "adapted to that card family)",
            "typed_not_recomputed": {
                "provenance": "these come from FINDINGS.md and the last run "
                              "of verify/independent_check.py, which needs "
                              "the raw model cards; they are not recomputed here",
                "coverage_bound_accounting_for_rejected_rows": [0.875, 0.906],
                "independent_reimplementation": {
                    "script": "verify/independent_check.py",
                    "shares_no_code_with_pipeline": True,
                    "delta_disagreements": 0,
                    "verdict_disagreements": 0,
                    "rows_compared": 186,
                },
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
            f"Coverage verdicts (rank.classify_cell): "
            f"{len(refused)} size cells refused ({', '.join(refused) or 'none'}); "
            f"{len(insufficient)} size cells and {len(sch_insufficient)} schemes "
            f"({', '.join(sch_insufficient) or 'none'}) have insufficient evidence "
            f"(fewer than {R.MIN_CELL_CHECKPOINTS} checkpoints or a "
            f"checkpoint-bootstrap interval straddling {R.REFUSE_BELOW*100:.0f}%): "
            f"{', '.join(insufficient) or 'none'}. An interval is still shown for "
            f"an insufficient-evidence cell but flagged; only a refused cell has "
            f"its interval withheld.",
            f"MoE and reasoning-distilled models are not separately "
            f"validated.",
        ],
        "schemes": schemes,
        "cells_refused": refused,
        "cells_insufficient_evidence": insufficient,
        "schemes_refused": sch_refused,
        "schemes_insufficient_evidence": sch_insufficient,
    }
    return art


def main():
    os.makedirs(OUT, exist_ok=True)
    art = build_artifact()
    schemes = art["schemes"]
    path = os.path.join(OUT, "scheme_envelope.json")
    with open(path, "w") as f:
        json.dump(art, f, indent=2)
    print(f"wrote {path}")
    print(f"\n{'scheme':<14}{'n':>5}{'fam':>5}{'mean':>8}{'90% interval':>20}"
          f"{'loss-side':>10}")
    for s in sorted(schemes):
        v = schemes[s]
        cov = v["coverage"]["one_sided"]
        print(f"{s:<14}{v['n_observations']:>5}{v['n_families']:>5}"
              f"{v['mean_delta_pp']:>+8.2f}"
              f"{str(v['interval_90_pp']):>20}"
              f"{f'{100*cov:.0f}%':>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
