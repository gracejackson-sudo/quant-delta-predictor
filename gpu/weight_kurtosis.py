"""Per-channel weight kurtosis for checkpoints in the corpus.

An external suggestion on Reddit: "W4A16 losses usually come from weight
outlier channels, conditioning the CI on weight kurtosis instead of
scheme + size might tighten things up."

This computes the feature. It does NOT test the hypothesis -- four
checkpoints cannot support the leave-one-family-out protocol the rest of the
project uses, so the honest output here is a measured feature plus a
directional look at how it lines up with observed damage.

Kurtosis is taken per OUTPUT CHANNEL (row of the weight matrix), because that
is the granularity at which per-channel scales are assigned during
quantization. A channel whose weights are heavy-tailed cannot be represented
well by a single scale, which is the mechanism the suggestion points at.

Excess kurtosis: 0 for a Gaussian, positive for heavy tails.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
try:
    import diskguard
except Exception:
    diskguard = None

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "data", "adversarial", "weight_kurtosis.json")

# The four ungated sub-10B checkpoints approved for this run.
MODELS = [
    ("Qwen/Qwen2.5-0.5B-Instruct", 0.9),
    ("Qwen/Qwen2.5-1.5B-Instruct", 2.9),
    ("ibm-granite/granite-3.1-2b-instruct", 4.7),
    ("Qwen/Qwen3-8B", 15.3),
]
# Only the projections quantization actually touches.
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
           "gate_proj", "up_proj", "down_proj")


def log(*a):
    print(*a, flush=True)


def channel_kurtosis(w: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis per output channel (row), float64 for stability."""
    x = w.to(torch.float64)
    mu = x.mean(dim=1, keepdim=True)
    d = x - mu
    var = (d ** 2).mean(dim=1)
    m4 = (d ** 4).mean(dim=1)
    return m4 / (var ** 2 + 1e-30) - 3.0


def summarise(mid):
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        mid, dtype=torch.float32, low_cpu_mem_usage=True)
    per_layer, all_k = [], []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if not any(t in name for t in TARGETS):
            continue
        k = channel_kurtosis(mod.weight.detach())
        all_k.append(k)
        per_layer.append({"layer": name,
                          "mean_kurtosis": float(k.mean()),
                          "p99_kurtosis": float(k.quantile(0.99)),
                          "channels": int(k.numel())})
    if not all_k:
        raise SystemExit(f"no target Linear layers found in {mid}")
    k = torch.cat(all_k)
    res = {
        "model": mid,
        "n_channels": int(k.numel()),
        "n_layers": len(per_layer),
        "mean_kurtosis": float(k.mean()),
        "median_kurtosis": float(k.median()),
        "p99_kurtosis": float(k.quantile(0.99)),
        "max_kurtosis": float(k.max()),
        # share of channels far outside Gaussian -- the "outlier channel" count
        "pct_channels_kurt_gt_3": float((k > 3).float().mean() * 100),
        "pct_channels_kurt_gt_10": float((k > 10).float().mean() * 100),
        "seconds": round(time.time() - t0, 1),
        "per_layer": per_layer,
    }
    del model
    gc.collect()
    return res


def main():
    need = sum(g for _, g in MODELS) + 2.0
    if diskguard:
        log(diskguard.report("weight download", need))
        diskguard.require_free_gb(need, "the kurtosis run")
    out = []
    for mid, gb in MODELS:
        log(f"\n=== {mid} (~{gb}GB) ===")
        try:
            r = summarise(mid)
        except Exception as e:
            log(f"  FAIL {type(e).__name__}: {e}")
            continue
        log(f"  channels {r['n_channels']} over {r['n_layers']} layers "
            f"({r['seconds']}s)")
        log(f"  mean {r['mean_kurtosis']:.3f}  median "
            f"{r['median_kurtosis']:.3f}  p99 {r['p99_kurtosis']:.3f}  "
            f"max {r['max_kurtosis']:.1f}")
        log(f"  channels with excess kurtosis >3: "
            f"{r['pct_channels_kurt_gt_3']:.2f}%   >10: "
            f"{r['pct_channels_kurt_gt_10']:.2f}%")
        out.append(r)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(out, open(OUT, "w"), indent=1)
    log(f"\nWROTE {len(out)} models -> {OUT}")
    log("KURTOSIS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
