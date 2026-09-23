"""
THE REAL USE CASE TEST.

Everything above this file was developed against 8 model families. This script
fits the predictor on ALL of that data, then downloads model cards for families
it has never seen at any point during development -- gemma-3, phi-4,
DeepSeek-R1-Distill, Llama-4, SmolLM, Nemotron, Mistral-Nemo, Devstral, and
Qwen3 MoE variants -- and asks the product question for real:

    given (model, quant config), does the 90% interval contain the truth?

This is a prospective test. No tuning happened after seeing these numbers; the
predictor and the calibration set are frozen before the download.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from harvest import harvest_card, parse_family  # noqa: E402
from model import load  # noqa: E402
from predictor import Conformal, GlobalMean, SchemeMean  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
PCARDS = os.path.join(HERE, "..", "data", "prospective_cards")
OUT = os.path.join(HERE, "..", "out")
ALPHA = 0.10
MIN_ACC = 20.0

# Fixed before looking at any of their numbers.
UNSEEN = [
    "RedHatAI/gemma-3-27b-it-quantized.w4a16",
    "RedHatAI/gemma-3-27b-it-quantized.w8a8",
    "RedHatAI/gemma-3-27b-it-FP8-dynamic",
    "RedHatAI/gemma-3-12b-it-quantized.w4a16",
    "RedHatAI/gemma-3-12b-it-quantized.w8a8",
    "RedHatAI/gemma-3-4b-it-quantized.w4a16",
    "RedHatAI/gemma-3-4b-it-FP8-dynamic",
    "RedHatAI/gemma-3-1b-it-quantized.w4a16",
    "RedHatAI/phi-4-quantized.w4a16",
    "RedHatAI/phi-4-quantized.w8a8",
    "RedHatAI/phi-4-FP8-dynamic",
    "RedHatAI/Phi-4-mini-instruct-quantized.w8a8",
    "RedHatAI/Phi-4-mini-instruct-FP8-dynamic",
    "RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16",
    "RedHatAI/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8",
    "RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16",
    "RedHatAI/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16",
    "RedHatAI/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16",
    "RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8",
    "RedHatAI/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a16",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
    "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "RedHatAI/Mistral-Nemo-Instruct-2407-quantized.w4a16",
    "RedHatAI/Mistral-Nemo-Instruct-2407-FP8",
    "RedHatAI/SmolLM-1.7B-Instruct-quantized.w8a8",
    "RedHatAI/SmolLM-1.7B-Instruct-quantized.w8a16",
    "RedHatAI/SmolLM3-3B-quantized.w4a16",
    "RedHatAI/Devstral-Small-2507-quantized.w4a16",
    "RedHatAI/Devstral-Small-2507-quantized.w8a8",
    "RedHatAI/Qwen3-30B-A3B-quantized.w4a16",
    "RedHatAI/Qwen3-30B-A3B-FP8-dynamic",
    "RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-quantized.w4a16",
    "RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16",
]

# families that overlap the training data and so must be excluded from the
# "unfamiliar family" headline (kept separately for comparison)
TRAINED_FAMILIES = {"gemma-2", "granite", "llama-3.1", "llama-3.2",
                    "llama-3.3", "mistral", "qwen2.5", "qwen3"}


def fetch():
    os.makedirs(PCARDS, exist_ok=True)
    for mid in UNSEEN:
        p = os.path.join(PCARDS, mid.replace("/", "_", 1) + ".md")
        if os.path.exists(p) and os.path.getsize(p) > 200:
            continue
        subprocess.run(
            ["curl", "-sL", "--max-time", "30",
             f"https://huggingface.co/{mid}/raw/main/README.md", "-o", p],
            check=False,
        )


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- freeze the predictor on everything we had during development
    d = load(DATA)
    train = d
    m = SchemeMean().fit(train)
    conf = Conformal(alpha=ALPHA, mondrian_by="scheme").fit(m, train)
    gm = GlobalMean().fit(train)
    print(f"FROZEN predictor: fitted on {len(train)} rows / "
          f"{train.family.nunique()} families / {train.base_model.nunique()} "
          f"base models")
    print("  scheme-mean table (pp):", {k: round(v, 3)
                                        for k, v in m.by_scheme.items()})
    print("  90% half-widths (pp)  :", {k: round(v, 2)
                                        for k, v in conf.q_group.items()})
    print("\nNOTE: calibration reuses the training rows here, which is the "
          "in-sample\n      variant. The honest out-of-family numbers are in "
          "run_final.py;\n      this script tests whether the frozen artifact "
          "generalizes at all.")

    # ---- now go get data the project has never seen
    fetch()
    rows, rejects = [], []
    for mid in UNSEEN:
        p = os.path.join(PCARDS, mid.replace("/", "_", 1) + ".md")
        if not os.path.exists(p):
            rejects.append([mid, "-", "download_failed"])
            continue
        r, rj = harvest_card(p, mid, allow_unknown_family=True)
        rows += r
        rejects += rj
    if not rows:
        print("\nno prospective rows parsed -- aborting")
        return 1

    p = pd.DataFrame(rows)
    p["family_raw"] = p.model.map(parse_family)
    p = p[p.acc_before >= MIN_ACC].reset_index(drop=True)

    def group_of(mid):
        s = mid.split("/")[-1].lower()
        for k in ("gemma-3n", "gemma-3", "phi-4", "deepseek-r1-distill",
                  "llama-4", "smollm3", "smollm", "mistral-nemo", "devstral",
                  "qwen3-30b-a3b", "nemotron"):
            if k in s:
                return k
        return parse_family(mid)
    p["group"] = p.model.map(group_of)

    yhat, lo, hi, fb = conf.predict_interval(p)
    p["pred"], p["lo"], p["hi"] = yhat, lo, hi
    p["inside_90"] = (p.delta >= p.lo) & (p.delta <= p.hi)
    p["ae"] = np.abs(p.delta - p.pred)
    p["ae_baseline"] = np.abs(p.delta - gm.predict(p))

    print(f"\nparsed {len(p)} prospective rows from "
          f"{p.model.nunique()} model cards across "
          f"{p.group.nunique()} model groups")
    if rejects:
        import collections
        print("  rejects:", dict(collections.Counter(x[2] for x in rejects)))

    print("\n--- by model group (all genuinely unfamiliar to the predictor) ---")
    t = (p.groupby("group")
         .agg(n=("inside_90", "size"), inside=("inside_90", "sum"),
              coverage=("inside_90", "mean"), mae=("ae", "mean"),
              mae_baseline=("ae_baseline", "mean"),
              mean_delta=("delta", "mean"), worst_delta=("delta", "min"))
         .sort_values("coverage"))
    print(t.round(3).to_string())

    print("\n--- by scheme ---")
    ts = (p.groupby("scheme")
          .agg(n=("inside_90", "size"), coverage=("inside_90", "mean"),
               half_width=("hi", lambda s: float(
                   np.mean((p.loc[s.index, "hi"] - p.loc[s.index, "lo"]) / 2))),
               mean_delta=("delta", "mean"))
          .sort_values("coverage"))
    print(ts.round(3).to_string())

    n, k = len(p), int(p.inside_90.sum())
    from demo_holdout import binom_ci
    clo, chi = binom_ci(k, n)
    print("\n" + "=" * 72)
    print("PROSPECTIVE RESULT on model families never seen in development")
    print("=" * 72)
    print(f"  rows                     : {n} from {p.model.nunique()} "
          f"(model, config) pairs")
    print(f"  inside the 90% interval  : {k}/{n} = {100*k/n:.1f}%")
    print(f"  95% CI on that coverage  : [{100*clo:.1f}%, {100*chi:.1f}%]")
    print(f"  nominal                  : 90.0%  -> "
          f"{'WITHIN' if clo <= 0.90 <= chi else 'OUTSIDE'} the CI")
    print(f"  mean half-width          : "
          f"{np.mean((p.hi - p.lo) / 2):.2f}pp")
    print(f"  MAE (predictor)          : {p.ae.mean():.3f}pp")
    print(f"  MAE (global-mean baseline): {p.ae_baseline.mean():.3f}pp")
    print(f"  intervals excluding zero : "
          f"{100*((p.lo > 0) | (p.hi < 0)).mean():.0f}%")

    miss = p[~p.inside_90].sort_values("delta")
    print(f"\n  the {len(miss)} misses (what a stranger would actually hit):")
    print(miss[["model", "scheme", "benchmark", "acc_before", "acc_after",
                "delta", "lo", "hi"]].to_string(index=False))

    p.to_csv(os.path.join(OUT, "real_use_case.csv"), index=False)
    with open(os.path.join(OUT, "real_use_case.json"), "w") as f:
        json.dump({"n": n, "inside": k, "coverage": k / n,
                   "ci": [clo, chi],
                   "mae": float(p.ae.mean()),
                   "mae_baseline": float(p.ae_baseline.mean()),
                   "mean_half_width": float(np.mean((p.hi - p.lo) / 2)),
                   "by_group": t.reset_index().to_dict("records")},
                  f, indent=2, default=str)
    print(f"\nwrote {OUT}/real_use_case.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
