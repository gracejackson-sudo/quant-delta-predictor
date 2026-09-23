"""
TASK 1: replace our hand-rolled GPTQ control with the real thing.

Our control arm loses 2.6x more accuracy than Red Hat's published w4a16 on
matched checkpoints. We attributed that to calibrating on ~171x less data plus
missing activation reordering. Rather than reimplement production GPTQ, this
uses llm-compressor -- the actual tool Red Hat used -- at its documented
defaults (512 samples x 2048 tokens, act-order).

If the gap closes, excess-over-control becomes a trustworthy anchor instead of
a caveated one. If it does not, the 2.6x is something other than calibration
volume and we have learned that cheaply.

Evaluation is deliberately IDENTICAL to session 2: MMLU 5-shot, letter-scored,
same 3000 items, same seed. Only the quantizer changes.
"""
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
SEED = 1234
OUT = "/home/ubuntu/work"
N_MMLU = 3000
LETTERS = ["A", "B", "C", "D"]

# llm-compressor documented defaults for GPTQ W4A16
NUM_CALIB = 512
MAX_SEQ = 2048

MODELS = [("Qwen/Qwen2.5-1.5B-Instruct", 1.5, -2.05)]   # published delta
if os.environ.get("INCLUDE_05B") == "1":
    MODELS.append(("Qwen/Qwen2.5-0.5B-Instruct", 0.5, -2.53))


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------ eval (identical to run 2)
def mmlu_items():
    random.seed(SEED)
    test = load_dataset("cais/mmlu", "all", split="test")
    dev = load_dataset("cais/mmlu", "all", split="dev")
    by_sub = {}
    for r in dev:
        by_sub.setdefault(r["subject"], []).append(r)
    items = [{"q": r["question"], "ch": r["choices"], "a": int(r["answer"]),
              "subject": r["subject"]} for r in test]
    random.shuffle(items)
    return items[:N_MMLU], by_sub


def fmt(r):
    q = r.get("question", r.get("q"))
    ch = r.get("choices", r.get("ch"))
    s = str(q).strip() + "\n"
    for L, c in zip(LETTERS, ch):
        s += f"{L}. {str(c).strip()}\n"
    return s + "Answer:"


def prompt(item, by_sub, k=5):
    sub = item["subject"].replace("_", " ")
    head = (f"The following are multiple choice questions (with answers) "
            f"about {sub}.\n\n")
    body = ""
    for r in by_sub.get(item["subject"], [])[:k]:
        ans = r.get("answer", r.get("a"))
        body += fmt(r) + f" {LETTERS[int(ans)]}\n\n"
    return head + body + fmt(item)


@torch.no_grad()
def score(model, tok, items, by_sub, bs=2):
    ids_letter = [tok(" " + L, add_special_tokens=False)["input_ids"][-1]
                  for L in LETTERS]
    correct = np.zeros(len(items), dtype=bool)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        enc = [tok(prompt(it, by_sub),
                   add_special_tokens=True)["input_ids"][-2048:]
               for it in chunk]
        mx = max(len(e) for e in enc)
        inp = torch.full((len(enc), mx), pad, dtype=torch.long)
        att = torch.zeros((len(enc), mx), dtype=torch.long)
        for k, e in enumerate(enc):
            inp[k, mx - len(e):] = torch.tensor(e)
            att[k, mx - len(e):] = 1
        kw = dict(input_ids=inp.to(DEV), attention_mask=att.to(DEV))
        try:
            lg = model(**kw, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            lg = model(**kw).logits[:, -1, :]
        pred = lg.float().log_softmax(-1)[:, ids_letter].argmax(-1).tolist()
        for k, it in enumerate(chunk):
            correct[i + k] = (pred[k] == it["a"])
        del lg
    return correct


# ------------------------------------------------------ real GPTQ
def quantize_real(mid, workdir):
    """Production llm-compressor GPTQ W4A16 at documented defaults."""
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.transformers import oneshot

    ds = load_dataset("HuggingFaceH4/ultrachat_200k",
                      split=f"train_sft[:{NUM_CALIB}]")
    tok = AutoTokenizer.from_pretrained(mid)

    def prep(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False)
        return tok(text, padding=False, max_length=MAX_SEQ, truncation=True,
                   add_special_tokens=False)

    ds = ds.map(prep, remove_columns=ds.column_names)
    recipe = GPTQModifier(targets="Linear", scheme="W4A16",
                          ignore=["lm_head"])
    out = os.path.join(workdir, os.path.basename(mid) + "-w4a16-real")
    oneshot(model=mid, dataset=ds, recipe=recipe,
            max_seq_length=MAX_SEQ, num_calibration_samples=NUM_CALIB,
            output_dir=out)
    return out


def main():
    items, by_sub = mmlu_items()
    log(f"[data] mmlu {len(items)} items, 5-shot, letter-scored")
    results = []
    for mid, pb, published in MODELS:
        short = mid.split("/")[-1]
        log(f"\n=== {short}  (published w4a16 MMLU delta {published:+.2f}pp) ===")
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        t0 = time.time()
        base = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16).to(DEV).eval()
        bc = score(base, tok, items, by_sub)
        a0 = 100 * float(bc.mean())
        log(f"  base MMLU {a0:.2f}   ({time.time()-t0:.0f}s)")
        del base
        torch.cuda.empty_cache()

        t0 = time.time()
        try:
            path = quantize_real(mid, OUT)
        except Exception as e:
            log(f"  QUANT FAIL {type(e).__name__}: {e}")
            continue
        qt = time.time() - t0
        log(f"  real GPTQ done in {qt:.0f}s -> {path}")

        t0 = time.time()
        qm = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.float16).to(DEV).eval()
        qc = score(qm, tok, items, by_sub)
        a1 = 100 * float(qc.mean())
        delta = a1 - a0
        flips = float((bc != qc).mean())
        log(f"  quantized MMLU {a1:.2f}   delta {delta:+.2f}pp   "
            f"flips {flips:.3f}   ({time.time()-t0:.0f}s)")
        log(f"  PUBLISHED {published:+.2f}pp   OURS {delta:+.2f}pp   "
            f"gap {delta-published:+.2f}pp   ratio {abs(delta/published):.2f}x")
        results.append({
            "model": short, "params_b": pb, "benchmark": "mmlu",
            "acc_before": round(a0, 4), "acc_after": round(a1, 4),
            "delta": round(delta, 4), "published_delta": published,
            "gap_vs_published": round(delta - published, 4),
            "ratio_vs_published": round(abs(delta / published), 4),
            "flip_rate": round(flips, 4), "quantizer": "llm-compressor-GPTQ",
            "num_calibration_samples": NUM_CALIB, "max_seq_length": MAX_SEQ,
            "quant_seconds": round(qt),
        })
        json.dump(results, open(f"{OUT}/task1_real_gptq.json", "w"), indent=1)
        del qm
        torch.cuda.empty_cache()

    log(f"\nWROTE {len(results)} rows")
    log("TASK1_DONE")


if __name__ == "__main__":
    sys.exit(main())
