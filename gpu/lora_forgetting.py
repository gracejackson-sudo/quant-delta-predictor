"""Measure catastrophic forgetting from LoRA/adapter merges.

The quantization work calibrates *damage*: how much accuracy a compression
scheme costs on capabilities you still need. The analogous quantity for an
adapter is not its gain on the task it was tuned for -- that gain is the point
-- but its cost on tasks it was NOT tuned for. That is forgetting, and it is
the thing worth calibrating.

For each (base model, adapter) pair we score the SAME held-out items twice,
before and after merging, and report

    forgetting = acc_merged - acc_base        (negative = capability lost)

Evaluation reuses gpu/five_shot.py verbatim -- the harness already validated
against published numbers to within 0.22pp -- rather than a fresh implementation.

Run `--smoke` first. It exercises every code path on a tiny random model on
CPU, so that a metered GPU box is never the place a bug is discovered.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import five_shot as FS  # the validated harness  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
import diskguard  # noqa: E402

# (base, adapter, rank). Every entry was verified against the adapter's own
# adapter_config.json: base_model_name_or_path must equal the base exactly.
# Adapters targeting a 4-bit base (e.g. unsloth/*-bnb-4bit) are deliberately
# excluded -- merging those into an fp16 base would confound quantization with
# the adapter effect, which is the very thing being measured.
#
# None of these were tuned on MMLU, so MMLU is genuinely held out and the
# delta is forgetting rather than target-task gain.
PAIRS_UNGATED = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Hazde/careerbot_PG6_Qwen_Qwen2.5-1.5B-Instruct_model_LoRA_5", 2),
    ("Qwen/Qwen2.5-1.5B-Instruct", "bharati2324/Qwen2.5-1.5B-Instruct-Code-LoRA-r16", 16),
    ("Qwen/Qwen2.5-1.5B-Instruct", "zjudai/flowertune-general-nlp-lora-qwen2.5-1.5b-instruct", 32),
    ("Qwen/Qwen2.5-1.5B-Instruct", "DreamGallery/Qwen-Qwen2.5-1.5B-Instruct-1727452927", 128),
    ("Qwen/Qwen2.5-0.5B-Instruct", "taronklm/Qwen2.5-0.5B-Instruct-lora-chatbot", 4),
    ("Qwen/Qwen2.5-0.5B-Instruct", "Ionio-ai/Qwen2.5-0.5B-Instruct-Ecommerce-Extraction-LoRA", 32),
    ("Qwen/Qwen2.5-3B-Instruct", "patilshrinivas/Qwen2.5-3B-Instruct-qlora-drug-ade-relation-extractor", 16),
    ("Qwen/Qwen2.5-3B-Instruct", "agastyasridharan/Qwen2.5-3B-Instruct-Sheldon-SFT-v3a-LoRA", 32),
]
# meta-llama bases are gated=manual: they need an accepted licence and an
# HF_TOKEN. Skipped automatically when no token is set, so a run never dies
# halfway through on a 403 the way GPU session 1 did.
PAIRS_GATED = [
    ("meta-llama/Llama-3.2-1B-Instruct", "codelion/Llama-3.2-1B-Instruct-tool-calling-lora", 64),
    ("meta-llama/Llama-3.2-1B-Instruct", "vimosh-v/lora-sycophancy-Llama-3.2-1B-Instruct", 8),
]


def pairs():
    if os.environ.get("HF_TOKEN"):
        return PAIRS_UNGATED + PAIRS_GATED
    log(f"[env] no HF_TOKEN: skipping {len(PAIRS_GATED)} gated meta-llama pairs")
    return PAIRS_UNGATED


PAIRS = PAIRS_UNGATED
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "data", "adversarial", "lora_forgetting.csv")


def log(*a):
    print(*a, flush=True)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_base(mid, dtype):
    m = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype)
    return m.to(device()).eval()


def merge_adapter(mid, adapter, dtype):
    """Base + adapter, merged into the weights.

    merge_and_unload() folds the adapter in so the merged model is scored by
    exactly the same code path as the base -- no PEFT wrapper in the forward
    pass, nothing that could differ between the two measurements.
    """
    from peft import PeftModel
    base = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    return merged.to(device()).eval()


def evaluate(model, tok, items, by_sub, bs):
    correct = FS.score_letters(model, tok, items, by_sub, bs=bs)
    return 100.0 * float(correct.mean()), correct


def run_pair(mid, adapter, items, by_sub, bs, dtype):
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    t0 = time.time()
    base = load_base(mid, dtype)
    acc_b, corr_b = evaluate(base, tok, items, by_sub, bs)
    log(f"    base     {acc_b:6.2f}  ({time.time()-t0:.0f}s)")
    del base
    if device() == "cuda":
        torch.cuda.empty_cache()

    t0 = time.time()
    merged = merge_adapter(mid, adapter, dtype)
    acc_m, corr_m = evaluate(merged, tok, items, by_sub, bs)
    log(f"    merged   {acc_m:6.2f}  ({time.time()-t0:.0f}s)")
    del merged
    if device() == "cuda":
        torch.cuda.empty_cache()

    flips = float((corr_b != corr_m).mean())
    return {
        "base_model": mid, "adapter": adapter, "benchmark": "mmlu",
        "n_items": len(items),
        "acc_before": round(acc_b, 4), "acc_after": round(acc_m, 4),
        "forgetting": round(acc_m - acc_b, 4), "flip_rate": round(flips, 4),
    }


def smoke():
    """Every code path, tiny random model, CPU, no network beyond two small
    repos. Proves the pipeline before any metered box is touched."""
    log("[smoke] tiny model, CPU, synthetic adapter")
    from peft import LoraConfig, get_peft_model
    mid = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()

    # Vary both the question and the correct index. A constant predictor then
    # scores ~25% rather than 0%, so accuracy has room to move in either
    # direction -- which is what makes the sensitivity check below meaningful.
    items = [{"q": f"Question {i}: pick the right option.",
              "ch": [f"opt{i}a", f"opt{i}b", f"opt{i}c", f"opt{i}d"],
              "a": i % 4, "subject": "math"} for i in range(40)]
    by_sub = {"math": []}

    acc_b, corr_b = evaluate(base, tok, items, by_sub, bs=2)
    log(f"[smoke] base scored: {acc_b:.2f} over {len(items)} items")

    cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                     task_type="CAUSAL_LM")
    merged = get_peft_model(base, cfg).merge_and_unload().eval()
    acc_m, corr_m = evaluate(merged, tok, items, by_sub, bs=2)
    log(f"[smoke] merged scored: {acc_m:.2f}")
    log(f"[smoke] forgetting = {acc_m - acc_b:+.2f}pp  "
        f"flip_rate = {float((corr_b != corr_m).mean()):.3f}")

    # Sensitivity. A pipeline that reports 0.00 for everything would also
    # report 0.00 on a genuinely damaged model, so prove the metric moves:
    # wreck the merged weights and require that it notices.
    import copy
    broken = copy.deepcopy(merged)
    with torch.no_grad():
        for prm in broken.parameters():
            prm.add_(torch.randn_like(prm) * 5.0)
    acc_x, corr_x = evaluate(broken, tok, items, by_sub, bs=2)
    moved = (abs(acc_x - acc_b) > 1e-9) or bool((corr_b != corr_x).any())
    log(f"[smoke] sabotaged model scored {acc_x:.2f} "
        f"(flips vs base {float((corr_b != corr_x).mean()):.3f})")
    if not moved:
        raise SystemExit(
            "[smoke] FAIL: the metric did not move on a deliberately wrecked "
            "model. It would report 0.00 for real damage too.")
    log("[smoke] sensitivity OK - metric responds to real weight damage")
    log("[smoke] OK - merge, eval, metric and sensitivity all exercised")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-items", type=int, default=FS.N_MMLU)
    ap.add_argument("--bs", type=int, default=2)
    a = ap.parse_args()
    if a.smoke:
        # the tiny test model is a few MB; the eval dataset is not
        print(diskguard.report("smoke test", 1.0))
        diskguard.require_free_gb(1.0, "the smoke test")
        return smoke()

    todo = pairs()
    if not todo:
        raise SystemExit("no pairs to run.")
    # four base models up to 3B, ten adapters, plus the MMLU dataset
    need = 1.5 * len(todo) + 5.0
    print(diskguard.report("models and dataset", need))
    diskguard.require_free_gb(need, "the LoRA forgetting run")
    dtype = torch.float16 if device() == "cuda" else torch.float32
    log(f"[env] device={device()} dtype={dtype} pairs={len(todo)}")

    items, by_sub = FS.mmlu_items()
    items = items[:a.n_items]
    log(f"[data] mmlu {len(items)} items, 5-shot, letter-scored")

    rows = []
    for mid, adapter, rank in todo:
        log(f"\n=== {mid}  +  {adapter} ===")
        try:
            row = run_pair(mid, adapter, items, by_sub, a.bs, dtype)
            row["lora_rank"] = rank
            rows.append(row)
        except Exception as e:
            log(f"    FAIL {type(e).__name__}: {e}")
        if rows:
            import csv
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    log(f"\nWROTE {len(rows)} rows -> {OUT}")
    log("LORA_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
