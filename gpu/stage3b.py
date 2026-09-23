"""
Deliberately-bad quantization runs to anchor the left tail (Track 3).

Design:
 * Paired evaluation: base and quantized scored on the IDENTICAL item set with
   a fixed seed, so only flipped items contribute noise (se = sqrt(flip/n)).
 * Multiple-choice loglikelihood only (arc/hellaswag/mmlu): fast, deterministic.
   GSM8K dropped -- 97% of its variance is measurement noise and it needs
   generation.
 * Every model also gets a CORRECT w4a16 control arm on the same items, so
   broken-vs-good is measured on one harness and cannot be confounded with
   "our harness differs from Red Hat's".
"""
import csv
import json
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
SEED = 1234
OUTDIR = "/home/ubuntu/work"
N = {"arc_challenge": 1172, "hellaswag": 3000, "mmlu": 3000}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ data
def get_items(bench):
    random.seed(SEED)
    if bench == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        it = [{"q": r["question"], "ch": r["choices"]["text"],
               "a": r["choices"]["label"].index(r["answerKey"])}
              for r in ds if r["answerKey"] in r["choices"]["label"]]
    elif bench == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        it = [{"q": r["ctx"], "ch": r["endings"], "a": int(r["label"])}
              for r in ds if str(r["label"]) != ""]
    else:
        ds = load_dataset("cais/mmlu", "all", split="test")
        it = [{"q": r["question"], "ch": r["choices"], "a": int(r["answer"])}
              for r in ds]
    random.shuffle(it)
    return it[:N[bench]]


# --------------------------------------------------------------- scoring
@torch.no_grad()
def score(model, tok, items, bs=8):
    correct = np.zeros(len(items), dtype=bool)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        flat, owner = [], []
        for j, it in enumerate(chunk):
            for c in it["ch"]:
                flat.append((it["q"], " " + str(c).strip()))
                owner.append(j)
        enc = [tok(p, add_special_tokens=True)["input_ids"] for p, _ in flat]
        full = [tok(p + c, add_special_tokens=True)["input_ids"]
                for p, c in flat]
        full = [f[-1024:] for f in full]
        enc = [e[-1024:] for e in enc]
        mx = max(len(f) for f in full)
        ids = torch.full((len(full), mx), pad, dtype=torch.long)
        att = torch.zeros((len(full), mx), dtype=torch.long)
        for k, f in enumerate(full):
            ids[k, :len(f)] = torch.tensor(f)
            att[k, :len(f)] = 1
        ids, att = ids.to(DEV), att.to(DEV)
        lg = model(input_ids=ids, attention_mask=att).logits.float()
        lg = lg.log_softmax(-1)
        sc = []
        for k, f in enumerate(full):
            s = max(len(enc[k]), 1)
            tgt = torch.tensor(f[s:], device=DEV)
            if len(tgt) == 0:
                sc.append(-1e9)
                continue
            lp = lg[k, s - 1:len(f) - 1, :].gather(
                -1, tgt.unsqueeze(-1)).sum().item()
            sc.append(lp / len(tgt))
        best = {}
        for k, o in enumerate(owner):
            best.setdefault(o, []).append(sc[k])
        for o, vals in best.items():
            correct[i + o] = (int(np.argmax(vals)) == chunk[o]["a"])
        del lg
    return correct


# --------------------------------------------------------- quantization
def rtn_quant_(w, bits, group, per_tensor=False):
    qmax = 2 ** (bits - 1) - 1
    if per_tensor:
        s = w.abs().max().clamp(min=1e-8) / qmax
        w.copy_(torch.clamp(torch.round(w / s), -qmax - 1, qmax) * s)
        return
    orig = w.shape
    if w.shape[1] % group != 0:
        group = w.shape[1]
    w2 = w.reshape(-1, group)
    s = w2.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    w2 = torch.clamp(torch.round(w2 / s), -qmax - 1, qmax) * s
    w.copy_(w2.reshape(orig))


def gptq_like_(w, bits, group, H):
    """Error-compensated quantization. Bad calibration -> bad H -> bad rounding."""
    qmax = 2 ** (bits - 1) - 1
    W = w.data.float().clone()
    d = W.shape[1]
    Hd = H + torch.eye(d, device=W.device) * (
        0.01 * torch.diag(H).mean().clamp(min=1e-6))
    try:
        Hinv = torch.linalg.cholesky(torch.linalg.inv(Hd), upper=True)
    except Exception:
        Hinv = torch.eye(d, device=W.device)
    diag = Hinv.diag().clamp(min=1e-6)
    for c0 in range(0, d, group):
        c1 = min(c0 + group, d)
        blk = W[:, c0:c1]
        s = blk.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
        q = torch.clamp(torch.round(blk / s), -qmax - 1, qmax) * s
        err = (blk - q) / diag[c0:c1].unsqueeze(0)
        W[:, c0:c1] = q
        if c1 < d:
            W[:, c1:] -= err @ Hinv[c0:c1, c1:]
    w.data.copy_(W.to(w.dtype))


@torch.no_grad()
def collect_H(model, tok, texts, layers, maxlen=256):
    H = {n: torch.zeros(m.in_features, m.in_features, device=DEV)
         for n, m in layers}
    hooks = []

    def mk(name):
        def f(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            H[name] += x.T @ x
        return f

    for n, m in layers:
        hooks.append(m.register_forward_hook(mk(n)))
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=maxlen).to(DEV)
        model(**ids)
    for h in hooks:
        h.remove()
    return H


def linears(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and "lm_head" not in n]


def calib_texts(kind, tok, n=24):
    random.seed(SEED)
    if kind == "good":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        return [x for x in ds["text"] if len(x) > 400][:n]
    if kind == "random_tokens":
        V = min(getattr(tok, "vocab_size", 30000), 30000)
        return [tok.decode([random.randint(5, V - 1) for _ in range(128)])
                for _ in range(8)]
    # wrong_domain: dense code for a general-purpose chat model
    ds = load_dataset("code-search-net/code_search_net", "python",
                      split="train", streaming=True, trust_remote_code=True)
    out = []
    for r in ds:
        c = r.get("whole_func_string") or ""
        if len(c) > 400:
            out.append(c)
        if len(out) >= n:
            break
    return out


# tag, scheme, bits, group, per_tensor, calib, method
RECIPES = [
    ("control_w4a16_good", "w4a16", 4, 128, False, "good", "gptq"),
    ("w4a16_randcalib", "w4a16", 4, 128, False, "random_tokens", "gptq"),
    ("w4a16_wrongdomain", "w4a16", 4, 128, False, "wrong_domain", "gptq"),
    ("w4a16_pertensor", "w4a16", 4, 0, True, "good", "rtn"),
    ("w8a8_nosmooth", "w8a8_int", 8, 0, True, "good", "rtn"),
]

MODELS = [("Qwen/Qwen2.5-1.5B-Instruct", 1.5)]
CONTROL_ONLY = [("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
                ("HuggingFaceTB/SmolLM-135M-Instruct", 0.135)]


def main():
    out = []
    items = {b: get_items(b) for b in N}
    for b in items:
        log(f"[data] {b}: {len(items[b])} items")

    plan = ([(m, p, [RECIPES[0]]) for m, p in CONTROL_ONLY]
            + [(m, p, RECIPES) for m, p in MODELS])
    for mid, pb, recipes in plan:
        short = mid.split("/")[-1]
        log(f"\n=== {short} ===")
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                mid, dtype=torch.float16).to(DEV).eval()
        except Exception as e:
            log(f"  SKIP {short}: {type(e).__name__} {e}")
            continue

        t0 = time.time()
        base_correct = {b: score(base, tok, items[b]) for b in items}
        for b in items:
            log(f"  base {b:<15} {100*base_correct[b].mean():6.2f}")
        log(f"  base eval {time.time()-t0:.0f}s")
        sd = {k: v.clone() for k, v in base.state_dict().items()}

        for tag, scheme, bits, group, pt, ck, method in recipes:
            t0 = time.time()
            base.load_state_dict(sd)
            try:
                if method == "gptq":
                    texts = calib_texts(ck, tok)
                    lyrs = linears(base)
                    H = collect_H(base, tok, texts, lyrs)
                    for n, m in lyrs:
                        gptq_like_(m.weight, bits, group or 128, H[n])
                    del H
                else:
                    for n, m in linears(base):
                        rtn_quant_(m.weight.data, bits, group or 128,
                                   per_tensor=pt)
                torch.cuda.empty_cache()
                qc = {b: score(base, tok, items[b]) for b in items}
            except Exception as e:
                log(f"  FAIL {tag}: {type(e).__name__} {e}")
                base.load_state_dict(sd)
                torch.cuda.empty_cache()
                continue

            for b in items:
                a0 = 100 * float(base_correct[b].mean())
                a1 = 100 * float(qc[b].mean())
                flips = float((base_correct[b] != qc[b]).mean())
                out.append({
                    "model": f"local/{short}-{tag}",
                    "base_model": short, "scheme": scheme, "recipe": tag,
                    "params_b": pb, "benchmark": b,
                    "acc_before": round(a0, 4), "acc_after": round(a1, 4),
                    "n_items": len(items[b]), "flip_rate": round(flips, 4),
                    "is_control": int(tag.startswith("control")),
                    "method": method, "weight_bits": bits,
                    "act_bits": 8 if scheme == "w8a8_int" else 16,
                    "num_type": "int", "is_instruct": 1,
                    "family": "adversarial",
                    "notes": f"calib={ck} group={group} per_tensor={pt}",
                })
                log(f"  {tag:<22}{b:<15}{a0:6.2f} -> {a1:6.2f}  "
                    f"d={a1-a0:+6.2f}  flip={flips:.3f}")
            log(f"  {tag} {time.time()-t0:.0f}s")
            json.dump(out, open(f"{OUTDIR}/stage3b_runs.json", "w"),
                      indent=1)

        base.load_state_dict(sd)
        del base
        torch.cuda.empty_cache()

    if out:
        with open(f"{OUTDIR}/stage3b_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    log(f"\nWROTE {len(out)} rows")
    log("BATCH_DONE")


if __name__ == "__main__":
    sys.exit(main())
