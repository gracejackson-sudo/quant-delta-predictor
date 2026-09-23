"""
OPTION B: rerun the adversarial recipes under the PUBLISHED harness protocol.

Why. Our first run scored MMLU 0-shot with full-continuation loglikelihood and
got 31.93 for Qwen2.5-0.5B-Instruct, where the RedHatAI cards report 47.42.
The validator flagged 10 such rows as >2pp base drift, which is why every
downstream number had to use excess-over-control rather than the raw delta.
If a 5-shot letter-scored MMLU reproduces the published base accuracy, the
deltas become directly comparable to the published corpus and the
excess-over-control workaround is no longer needed.

Protocol changes vs run 1:
  * MMLU 5-shot, few-shot examples drawn from each subject's dev split, in the
    standard "The following are multiple choice questions (with answers)
    about {subject}." framing.
  * MMLU scored by comparing loglikelihood of the single tokens " A"/" B"/
    " C"/" D" after "Answer:", which is what lm-eval-harness does -- not by
    scoring the full answer text.
  * ARC-Challenge 25-shot, full-continuation scoring (OpenLLM v1 protocol).
  * HellaSwag left at 10-shot continuation scoring; our 0-shot already matched
    published within ~1pp so it is the control on the harness change itself.
"""
import csv
import json
import random
import sys
import time

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda"
SEED = 1234
OUTDIR = "/home/ubuntu/work"
N_MMLU = 3000
N_ARC = 1172
N_HELLA = 3000
LETTERS = ["A", "B", "C", "D"]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------- data
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


def fmt_mmlu(r):
    """Accepts either raw MMLU rows (question/choices/answer) or our
    converted items (q/ch/a), since dev shots are raw and test items are not."""
    q = r.get("question", r.get("q"))
    ch = r.get("choices", r.get("ch"))
    s = str(q).strip() + "\n"
    for L, c in zip(LETTERS, ch):
        s += f"{L}. {str(c).strip()}\n"
    return s + "Answer:"


def mmlu_prompt(item, by_sub, k=5):
    sub = item["subject"].replace("_", " ")
    head = (f"The following are multiple choice questions (with answers) "
            f"about {sub}.\n\n")
    shots = by_sub.get(item["subject"], [])[:k]
    body = ""
    for r in shots:
        ans = r.get("answer", r.get("a"))
        body += fmt_mmlu(r) + f" {LETTERS[int(ans)]}\n\n"
    return head + body + fmt_mmlu(item)


def arc_items():
    random.seed(SEED)
    tr = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    te = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")

    def conv(r):
        if r["answerKey"] not in r["choices"]["label"]:
            return None
        return {"q": r["question"], "ch": r["choices"]["text"],
                "a": r["choices"]["label"].index(r["answerKey"])}
    shots = [c for c in (conv(r) for r in tr) if c][:25]
    items = [c for c in (conv(r) for r in te) if c]
    random.shuffle(items)
    return items[:N_ARC], shots


def hella_items():
    random.seed(SEED)
    ds = load_dataset("Rowan/hellaswag", split="validation")
    tr = load_dataset("Rowan/hellaswag", split="train")
    items = [{"q": r["ctx"], "ch": r["endings"], "a": int(r["label"])}
             for r in ds if str(r["label"]) != ""]
    shots = [{"q": r["ctx"], "ch": r["endings"], "a": int(r["label"])}
             for r in tr if str(r["label"]) != ""][:10]
    random.shuffle(items)
    return items[:N_HELLA], shots


def fewshot_prefix(shots):
    out = ""
    for s in shots:
        out += f"{s['q'].strip()} {str(s['ch'][s['a']]).strip()}\n\n"
    return out


# ---------------------------------------------------------- scoring
@torch.no_grad()
def score_letters(model, tok, items, by_sub, bs=2):
    """MMLU: compare loglikelihood of ' A'..' D' after the prompt."""
    ids_letter = []
    for L in LETTERS:
        t = tok(" " + L, add_special_tokens=False)["input_ids"]
        ids_letter.append(t[-1])
    correct = np.zeros(len(items), dtype=bool)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        prompts = [mmlu_prompt(it, by_sub) for it in chunk]
        enc = [tok(p, add_special_tokens=True)["input_ids"][-2048:]
               for p in prompts]
        mx = max(len(e) for e in enc)
        inp = torch.full((len(enc), mx), pad, dtype=torch.long)
        att = torch.zeros((len(enc), mx), dtype=torch.long)
        for k, e in enumerate(enc):
            inp[k, mx - len(e):] = torch.tensor(e)   # left pad
            att[k, mx - len(e):] = 1
        # Ask the model for only the final position's logits where supported.
        # Without this the full [B, T, V] tensor is materialised (B=2, T=2k,
        # V=152k is already ~1.2 GiB in fp16, and .float() doubles it).
        kw = dict(input_ids=inp.to(DEV), attention_mask=att.to(DEV))
        try:
            lg = model(**kw, logits_to_keep=1).logits[:, -1, :]
        except TypeError:
            lg = model(**kw).logits[:, -1, :]
        lp = lg.float().log_softmax(-1)[:, ids_letter]
        pred = lp.argmax(-1).tolist()
        del lg, lp
        for k, it in enumerate(chunk):
            correct[i + k] = (pred[k] == it["a"])
    return correct


@torch.no_grad()
def score_cont(model, tok, items, prefix, bs=8):
    """ARC / HellaSwag: full-continuation loglikelihood, length-normalised."""
    correct = np.zeros(len(items), dtype=bool)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        flat, owner = [], []
        for j, it in enumerate(chunk):
            for c in it["ch"]:
                flat.append((prefix + it["q"].strip(),
                             " " + str(c).strip()))
                owner.append(j)
        enc = [tok(p, add_special_tokens=True)["input_ids"][-2048:]
               for p, _ in flat]
        full = [tok(p + c, add_special_tokens=True)["input_ids"][-2048:]
                for p, c in flat]
        mx = max(len(f) for f in full)
        inp = torch.full((len(full), mx), pad, dtype=torch.long)
        att = torch.zeros((len(full), mx), dtype=torch.long)
        for k, f in enumerate(full):
            inp[k, :len(f)] = torch.tensor(f)
            att[k, :len(f)] = 1
        lg = model(input_ids=inp.to(DEV),
                   attention_mask=att.to(DEV)).logits.float().log_softmax(-1)
        sc = []
        for k, f in enumerate(full):
            s = max(len(enc[k]), 1)
            tgt = torch.tensor(f[s:], device=DEV)
            if len(tgt) == 0:
                sc.append(-1e9)
                continue
            sc.append(lg[k, s - 1:len(f) - 1, :].gather(
                -1, tgt.unsqueeze(-1)).sum().item() / len(tgt))
        best = {}
        for k, o in enumerate(owner):
            best.setdefault(o, []).append(sc[k])
        for o, v in best.items():
            correct[i + o] = (int(np.argmax(v)) == chunk[o]["a"])
        del lg
    return correct


# ---------------------------------------------------- quantization
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
        model(**tok(t, return_tensors="pt", truncation=True,
                    max_length=maxlen).to(DEV))
    for h in hooks:
        h.remove()
    return H


def linears(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, nn.Linear) and "lm_head" not in n]


def calib_texts(kind, tok, n=24):
    random.seed(SEED)
    if kind == "good":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                          split="train")
        return [x for x in ds["text"] if len(x) > 400][:n]
    if kind == "random_tokens":
        V = min(getattr(tok, "vocab_size", 30000), 30000)
        return [tok.decode([random.randint(5, V - 1) for _ in range(128)])
                for _ in range(8)]
    ds = load_dataset("code-search-net/code_search_net", "python",
                      split="train", streaming=True)
    out = []
    for r in ds:
        c = r.get("whole_func_string") or ""
        if len(c) > 400:
            out.append(c)
        if len(out) >= n:
            break
    return out


RECIPES = [
    ("control_w4a16_good", "w4a16", 4, 128, False, "good", "gptq"),
    ("w4a16_randcalib", "w4a16", 4, 128, False, "random_tokens", "gptq"),
    ("w4a16_wrongdomain", "w4a16", 4, 128, False, "wrong_domain", "gptq"),
    ("w4a16_pertensor", "w4a16", 4, 0, True, "good", "rtn"),
    ("w8a8_nosmooth", "w8a8_int", 8, 0, True, "good", "rtn"),
]
MODELS = [("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
          ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
          ("HuggingFaceTB/SmolLM-135M-Instruct", 0.135)]


def main():
    out = []
    mm, by_sub = mmlu_items()
    log(f"[data] mmlu {len(mm)} (5-shot, letter-scored)")

    def score_all(model, tok):
        # MMLU only. ARC at 25-shot OOM'd (a 4k-token prompt x 4 choices x
        # batch 8 materialises a ~17 GiB logits tensor), and ARC/HellaSwag
        # already matched published within ~1-3pp at 0-shot, so MMLU is the
        # only benchmark where the protocol change actually mattered.
        return {"mmlu": score_letters(model, tok, mm, by_sub)}

    for mid, pb in MODELS:
        short = mid.split("/")[-1]
        log(f"\n=== {short} ===")
        tok = AutoTokenizer.from_pretrained(mid)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            mid, dtype=torch.float16).to(DEV).eval()
        t0 = time.time()
        bc = score_all(base, tok)
        for b in bc:
            log(f"  base {b:<15}{100*bc[b].mean():6.2f}")
        log(f"  base eval {time.time()-t0:.0f}s")
        sd = {k: v.clone() for k, v in base.state_dict().items()}

        for tag, scheme, bits, group, pt, ck, method in RECIPES:
            t0 = time.time()
            base.load_state_dict(sd)
            try:
                if method == "gptq":
                    lyrs = linears(base)
                    H = collect_H(base, tok, calib_texts(ck, tok), lyrs)
                    for n, m in lyrs:
                        gptq_like_(m.weight, bits, group or 128, H[n])
                    del H
                else:
                    for n, m in linears(base):
                        rtn_quant_(m.weight.data, bits, group or 128,
                                   per_tensor=pt)
                torch.cuda.empty_cache()
                qc = score_all(base, tok)
            except Exception as e:
                log(f"  FAIL {tag}: {type(e).__name__} {e}")
                base.load_state_dict(sd)
                torch.cuda.empty_cache()
                continue
            for b in bc:
                a0 = 100 * float(bc[b].mean())
                a1 = 100 * float(qc[b].mean())
                out.append({
                    "model": f"local/{short}-{tag}-5shot",
                    "base_model": short, "scheme": scheme, "recipe": tag,
                    "params_b": pb, "benchmark": b,
                    "acc_before": round(a0, 4), "acc_after": round(a1, 4),
                    "n_items": len(bc[b]),
                    "flip_rate": round(float((bc[b] != qc[b]).mean()), 4),
                    "is_control": int(tag.startswith("control")),
                    "method": method, "weight_bits": bits,
                    "act_bits": 8 if scheme == "w8a8_int" else 16,
                    "num_type": "int", "is_instruct": 1,
                    "family": "adversarial", "protocol": "published_fewshot",
                    "notes": f"calib={ck} group={group} per_tensor={pt}",
                })
                log(f"  {tag:<22}{b:<15}{a0:6.2f} -> {a1:6.2f}  "
                    f"d={a1-a0:+6.2f}")
            log(f"  {tag} {time.time()-t0:.0f}s")
            json.dump(out, open(f"{OUTDIR}/mmlu5shot_runs.json", "w"), indent=1)
        base.load_state_dict(sd)
        del base
        torch.cuda.empty_cache()

    if out:
        with open(f"{OUTDIR}/mmlu5shot_runs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    log(f"\nWROTE {len(out)} rows")
    log("BATCH_DONE")


if __name__ == "__main__":
    sys.exit(main())
