"""
Ingest contract for empirical adversarial quantization runs (Track 3 anchor).

The bias correction needs LEFT-TAIL data: deltas from configs deliberately
chosen to damage a model. This module defines exactly what the GPU run must
hand back, and fails loudly on a mismatch rather than silently producing a
wrong correction.

Every check here exists because something like it already went wrong once:
  * 0-1 vs 0-100 accuracy scale  -> day-1 bug, 2 models silently corrupted
  * delta not equal to after-before -> would invalidate every downstream number
  * unknown benchmark name       -> MATH-500 was read as Math-Lvl-5
  * base accuracy disagreeing with the published corpus for the same
    checkpoint -> means the eval harness or prompt template differs, which
    makes the delta non-comparable with the rest of the data

Usage:
    python src/adversarial_schema.py path/to/adversarial_runs.csv
    python src/adversarial_schema.py --template      # write a blank template
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from harvest import BENCH_N  # noqa: E402
from model import load  # noqa: E402
from strata import size_band  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
TEMPLATE = os.path.join(HERE, "..", "data", "adversarial_runs.template.csv")

# --- the contract -----------------------------------------------------------
REQUIRED = [
    "model",        # identifier you give the quantized artifact
    "base_model",   # the checkpoint it was made from
    "scheme",       # w4a16 / w8a8_int / nvfp4 / ... or a NEW name (declare it)
    "recipe",       # free text: what was deliberately wrong with this config
    "params_b",     # float, billions of parameters
    "benchmark",    # must be one of the known benchmark keys
    "acc_before",   # base model accuracy, 0-100 scale
    "acc_after",    # quantized accuracy, 0-100 scale
]
OPTIONAL = ["family", "n_items", "weight_bits", "act_bits", "num_type",
            "method", "is_instruct", "notes"]

KNOWN_SCHEMES = {"w4a16", "w8a8_int", "w8a16", "fp8", "fp8_dynamic", "nvfp4"}


def template():
    rows = [{
        "model": "local/SmolLM-135M-Instruct-w2a16-nocalib",
        "base_model": "SmolLM-135M-Instruct",
        "scheme": "w2a16",
        "recipe": "2-bit weights, GPTQ with 8 random-token calibration samples",
        "params_b": 0.135,
        "benchmark": "arc_challenge",
        "acc_before": 37.20,
        "acc_after": 25.10,
        "family": "smollm",
        "n_items": 1172,
        "weight_bits": 2, "act_bits": 16, "num_type": "int",
        "method": "gptq", "is_instruct": 1,
        "notes": "deliberately mismatched calibration set",
    }]
    pd.DataFrame(rows).to_csv(TEMPLATE, index=False)
    return TEMPLATE


def validate(path, corpus=None):
    """-> (ok: bool, problems: list[str], warnings: list[str], df|None)"""
    probs, warns = [], []
    if not os.path.exists(path):
        return False, [f"file not found: {path}"], [], None
    try:
        df = pd.read_csv(path)
    except Exception as e:                                   # noqa: BLE001
        return False, [f"cannot parse as CSV: {e}"], [], None

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        probs.append(f"missing required columns: {missing}")
        return False, probs, warns, None
    if len(df) == 0:
        return False, ["file has zero rows"], [], None

    # --- types / nulls
    for c in ("acc_before", "acc_after", "params_b"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            probs.append(f"{c}: {int(df[c].isna().sum())} non-numeric/missing")
    for c in ("model", "base_model", "scheme", "benchmark", "recipe"):
        if df[c].isna().any() or (df[c].astype(str).str.strip() == "").any():
            probs.append(f"{c}: blank values present")

    # --- THE SCALE CHECK (day-1 bug)
    acc = pd.concat([df.acc_before, df.acc_after]).dropna()
    if len(acc) and acc.max() <= 1.0:
        probs.append("accuracies look like a 0-1 scale (max <= 1.0); this "
                     "pipeline expects 0-100. Multiply by 100 and re-submit")
    if (acc < 0).any() or (acc > 100).any():
        probs.append("accuracies outside [0, 100]")

    # --- delta consistency if a delta column was supplied
    if "delta" in df.columns:
        d_ = pd.to_numeric(df.delta, errors="coerce")
        bad = (np.abs((df.acc_after - df.acc_before) - d_) > 1e-6).sum()
        if bad:
            probs.append(f"delta != acc_after - acc_before on {int(bad)} rows")

    # --- benchmark vocabulary
    unknown = sorted(set(df.benchmark) - set(BENCH_N))
    if unknown:
        probs.append(f"unknown benchmark keys {unknown}; allowed: "
                     f"{sorted(BENCH_N)}")

    # --- scheme vocabulary
    new_schemes = sorted(set(df.scheme) - KNOWN_SCHEMES)
    if new_schemes:
        warns.append(f"new scheme names {new_schemes} -- intended for "
                     f"adversarial configs, but they will NOT join an "
                     f"existing per-scheme envelope; they anchor the tail only")

    # --- duplicates
    dup = df.duplicated(["model", "benchmark"]).sum()
    if dup:
        probs.append(f"{int(dup)} duplicate (model, benchmark) rows")

    # --- cross-check against the published corpus
    corpus = load(DATA) if corpus is None else corpus
    ref = (corpus.groupby(["base_model", "benchmark"]).acc_before
           .mean().to_dict())
    drift = []
    for _, r in df.iterrows():
        key = (r.base_model, r.benchmark)
        if key in ref and pd.notna(r.acc_before):
            gap = abs(ref[key] - r.acc_before)
            if gap > 2.0:
                drift.append((r.base_model, r.benchmark, ref[key],
                              r.acc_before, gap))
    if drift:
        warns.append(
            f"{len(drift)} rows whose base accuracy differs from the published "
            f"corpus by >2pp -- likely a different harness or prompt template, "
            f"which makes the delta non-comparable: " +
            "; ".join(f"{b}/{bm} corpus={c:.1f} yours={y:.1f}"
                      for b, bm, c, y, _ in drift[:4]))

    # --- is the data actually adversarial?
    if not probs:
        dd = df.acc_after - df.acc_before
        n_bad = int((dd <= -3.0).sum())
        if n_bad == 0:
            warns.append(
                f"no row lost more than 3pp (worst {dd.min():+.2f}pp). These "
                f"configs are not damaging enough to anchor a left tail; the "
                f"correction needs genuinely bad outcomes")
        else:
            warns.append(
                f"{n_bad}/{len(df)} rows lost more than 3pp "
                f"(worst {dd.min():+.2f}pp) -- usable as left-tail anchors")

    return (len(probs) == 0), probs, warns, (None if probs else df)


def to_corpus_rows(df):
    """Normalise a validated frame into dataset.csv's column layout."""
    out = df.copy()
    out["delta"] = (out.acc_after - out.acc_before).round(4)
    if "family" not in out:
        out["family"] = "adversarial"
    if "n_items" not in out:
        out["n_items"] = out.benchmark.map(BENCH_N)
    for c, default in (("weight_bits", np.nan), ("act_bits", np.nan),
                       ("num_type", "int"), ("method", "adversarial"),
                       ("is_instruct", 0)):
        if c not in out:
            out[c] = default
    out["band"] = out.params_b.map(size_band)
    out["is_adversarial"] = 1
    out["verified"] = 0
    out["orient_src"] = "measured_locally"
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "--template":
        print(f"wrote {template()}")
        return 0
    if not argv:
        print(__doc__)
        print("REQUIRED columns:", REQUIRED)
        print("OPTIONAL columns:", OPTIONAL)
        print("allowed benchmarks:", sorted(BENCH_N))
        return 0
    ok, probs, warns, df = validate(argv[0])
    print(f"validating {argv[0]}")
    if df is not None:
        print(f"  rows: {len(df)}  models: {df.model.nunique()}  "
              f"schemes: {sorted(set(df.scheme))}")
    for p_ in probs:
        print(f"  FAIL  {p_}")
    for w in warns:
        print(f"  WARN  {w}")
    print(f"\n{'READY TO INGEST' if ok else 'REJECTED - fix the above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
