"""Test whether per-channel weight kurtosis predicts LoRA-merge forgetting.

Independent re-derivation for KURTOSIS_LORA_FINDINGS.md: reads the two raw
GPU-run artifacts directly (data/adversarial/lora_forgetting.csv and
weight_kurtosis.json) and computes every number the writeup cites. No number
in that doc should be hand-typed; re-run this script to check any of them.
"""
from __future__ import annotations
import json
import os
import sys

import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
LORA_CSV = os.path.join(ROOT, "data", "adversarial", "lora_forgetting.csv")
KURT_JSON = os.path.join(ROOT, "data", "adversarial", "weight_kurtosis.json")


def main():
    lora = pd.read_csv(LORA_CSV)
    kurt = {m["model"]: m for m in json.load(open(KURT_JSON))}

    lora_models = set(lora.base_model.unique())
    kurt_models = set(kurt.keys())
    overlap = lora_models & kurt_models

    print(f"LoRA base models ({len(lora_models)}): {sorted(lora_models)}")
    print(f"Kurtosis models  ({len(kurt_models)}): {sorted(kurt_models)}")
    print(f"Overlap ({len(overlap)}): {sorted(overlap)}")
    print(f"LoRA-only, no kurtosis measured: {sorted(lora_models - kurt_models)}")
    print(f"Kurtosis-only, no LoRA measured: {sorted(kurt_models - lora_models)}")
    print()

    print("=== Per-model summary ===")
    for bm, g in lora.groupby("base_model"):
        k = kurt.get(bm)
        kurt_str = (f"pct_kurt_gt3={k['pct_channels_kurt_gt_3']:.3f}%"
                    if k else "NOT MEASURED")
        print(f"{bm}: n_pairs={len(g)} mean_forgetting={g.forgetting.mean():.4f} "
              f"std={g.forgetting.std():.4f} range=[{g.forgetting.min():.2f},"
              f"{g.forgetting.max():.2f}]  kurtosis: {kurt_str}")
    print()

    sub = lora.copy()
    sub["pct_kurt_gt3"] = sub.base_model.map(
        lambda m: kurt.get(m, {}).get("pct_channels_kurt_gt_3"))
    sub = sub.dropna(subset=["pct_kurt_gt3"])
    print("=== Pooled kurtosis-vs-forgetting correlation (CAVEAT: pseudo-"
          f"replicated -- {sub.base_model.nunique()} distinct model-level "
          f"kurtosis values repeated across {len(sub)} rows) ===")
    r_p, p_p = stats.pearsonr(sub.pct_kurt_gt3, sub.forgetting)
    r_s, p_s = stats.spearmanr(sub.pct_kurt_gt3, sub.forgetting)
    print(f"Pearson r={r_p:.4f} (p={p_p:.4f})  Spearman r={r_s:.4f} (p={p_s:.4f})")
    print()

    print("=== forgetting vs LoRA rank, all 8 pairs (potential confound) ===")
    r_rank, p_rank = stats.spearmanr(lora.lora_rank, lora.forgetting)
    print(f"Spearman r={r_rank:.4f} (p={p_rank:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
