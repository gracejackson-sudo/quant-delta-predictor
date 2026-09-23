#!/usr/bin/env python3
"""
FULLY INDEPENDENT RE-VERIFICATION of the strict held-out coverage number.

Deliberately shares NOTHING with the main project:
  * no imports from src/ (not harvest, not model, not predictor, not audit)
  * no numpy, pandas, scikit-learn or scipy -- Python standard library only
  * table extraction written with regex rather than html.parser, so a bug in
    the main parser cannot be reproduced by copying its approach
  * its own scheme detection, its own recovery verification, its own conformal
    quantile, its own strict-subset rule

It reads only the raw model-card .md files on disk and reports what it gets.
Any disagreement with the pipeline is printed row by row.

Usage:  python3 verify/independent_check.py
"""
import csv
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(ROOT, "data", "cards")
TEST_DIR = os.path.join(ROOT, "data", "prospective_cards")
ALPHA = 0.10
MIN_ACC = 20.0

# --------------------------------------------------------------------------
# independent benchmark vocabulary (written from the card text, not copied)
# --------------------------------------------------------------------------
BENCH = [
    ("mmlu_pro", r"\bmmlu[\s\-_]*pro\b"),
    ("mmlu_cot", r"\bmmlu\b[^a-z]*\(?\s*cot"),
    ("mmlu", r"\bmmlu\b"),
    ("arc_challenge", r"\barc[\s\-_]*(challenge|c)\b"),
    ("gsm8k", r"\bgsm[\s\-_]*8?k\b"),
    ("hellaswag", r"\bhellaswag\b"),
    ("winogrande", r"\bwinogrande\b"),
    ("truthfulqa", r"\btruthful"),
    ("ifeval", r"\bifeval\b"),
    ("bbh", r"\bbbh\b|\bbig[\s\-_]*bench"),
    ("math_lvl5", r"\bmath[\s\-_]*(lvl|lv|vl|v|level)?[\s\-_]*5\b|"
                  r"\bmath[\s\-_]*hard\b"),
    ("gpqa", r"\bgpqa\b"),
    ("musr", r"\bmusr\b"),
    ("humaneval_plus", r"\bhumaneval\s*\+|\bhumaneval[\s\-_]*plus\b"),
    ("humaneval", r"\bhumaneval\b"),
    ("arena_hard", r"\barena[\s\-_]*hard\b"),
]
SKIP_LABEL = re.compile(r"average|recovery|^\s*score\s*$", re.I)

SCHEME_RULES = [
    ("nvfp4", r"nvfp4"),
    ("mxfp4", r"mxfp4"),
    ("w4a16", r"quantized\.w4a16|w4a16|int4"),
    ("w8a16", r"quantized\.w8a16|w8a16"),
    ("w8a8_int", r"quantized\.w8a8|w8a8|int8"),
    ("fp8_dynamic", r"fp8[-_]dynamic"),
    ("fp8", r"fp8"),
]


def scheme_of(model_id):
    s = model_id.lower()
    # order matters: check the most specific spellings first
    if re.search(r"nvfp4", s):
        return "nvfp4"
    if re.search(r"mxfp4", s):
        return "mxfp4"
    if re.search(r"w4a16|int4", s):
        return "w4a16"
    if re.search(r"w8a16", s):
        return "w8a16"
    if re.search(r"w8a8|int8", s):
        return "w8a8_int"
    if re.search(r"fp8[-_]dynamic", s):
        return "fp8_dynamic"
    if re.search(r"fp8", s):
        return "fp8"
    return None


def bench_of(label):
    t = label.lower().replace("|", " ")
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[*`]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or SKIP_LABEL.search(t):
        return None
    for name, pat in BENCH:
        if re.search(pat, t):
            return name
    return None


# --------------------------------------------------------------------------
# independent table extraction, regex based
# --------------------------------------------------------------------------
TAG = re.compile(r"<[^>]+>")


def clean(cell):
    return re.sub(r"\s+", " ", TAG.sub(" ", cell)).strip()


def html_rows(text):
    for tbl in re.findall(r"<table\b.*?</table>", text,
                          re.S | re.I):
        for tr in re.findall(r"<tr\b.*?</tr>", tbl, re.S | re.I):
            cells = [clean(c) for c in
                     re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", tr,
                                re.S | re.I)]
            if cells:
                yield tbl, cells


def md_rows(text):
    block = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("|") and s.count("|") >= 3:
            block.append(s)
        else:
            if len(block) >= 2:
                for b in block:
                    cells = [clean(c) for c in b.strip("|").split("|")]
                    if all(re.fullmatch(r":?-{2,}:?", c)
                           for c in cells if c):
                        continue
                    yield "\n".join(block), cells
            block = []
    if len(block) >= 2:
        for b in block:
            cells = [clean(c) for c in b.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            yield "\n".join(block), cells


PURE_NUM = re.compile(
    r"^\**\s*([-+]?\d+(?:\.\d+)?)\s*\**\s*%?\s*\**"
    r"(?:\(\s*[\d.\s/,+-]+\s*\))?\s*\**$")


def as_number(cell):
    m = PURE_NUM.match(cell.strip())
    return float(m.group(1)) if m else None


def dp(raw):
    m = re.search(r"\d+\.(\d+)", raw)
    return len(m.group(1)) if m else 0


def tol_for(before, after, rb, ra, rr):
    """Max legitimate gap between printed recovery and 100*after/before."""
    ub, ua, ur = (0.5 * 10 ** -dp(rb), 0.5 * 10 ** -dp(ra),
                  0.5 * 10 ** -dp(rr))
    return 100.0 * (ua / before + after * ub / (before * before)) + ur + 1e-9


def read_card(path, model_id):
    """-> list of dicts, using an independently written resolution rule."""
    text = open(path, encoding="utf-8", errors="replace").read()
    sch = scheme_of(model_id)
    if sch is None:
        return []
    out, seen = [], set()
    for tbl, cells in list(html_rows(text)) + list(md_rows(text)):
        vals = [as_number(c) for c in cells]
        # trailing run of purely-numeric cells
        j = len(cells)
        while j > 0 and vals[j - 1] is not None:
            j -= 1
        nums, raws = vals[j:], cells[j:]
        if not nums:
            continue
        label = next((cells[k] for k in range(j - 1, -1, -1)
                      if cells[k].strip()), None)
        if label is None:
            continue
        b = bench_of(label)
        if b is None or b in seen:
            continue

        # where does the header say Recovery is?
        head = tbl[:tbl.find(label)] if label in tbl else tbl
        rec_hint = None
        for hrow in re.findall(r"<tr\b.*?</tr>", head, re.S | re.I) or []:
            hc = [clean(c) for c in re.findall(
                r"<t[dh]\b[^>]*>(.*?)</t[dh]>", hrow, re.S | re.I)]
            hc = [c for c in hc if c]
            if len(hc) >= len(nums):
                tail = hc[-len(nums):]
                for i, c in enumerate(tail):
                    if re.search(r"recovery", c, re.I):
                        rec_hint = i
        if rec_hint is None:
            mline = head.split("\n")[0] if "|" in head else ""
            hc = [clean(c) for c in mline.strip("|").split("|")] \
                if mline.startswith("|") else []
            hc = [c for c in hc if c]
            if len(hc) >= len(nums):
                for i, c in enumerate(hc[-len(nums):]):
                    if re.search(r"recovery", c, re.I):
                        rec_hint = i

        before = after = None
        if len(nums) == 3:
            # brute force: try every assignment of the three numbers
            ok = []
            for bi in range(3):
                for ai in range(3):
                    for ri in range(3):
                        if len({bi, ai, ri}) != 3:
                            continue
                        if rec_hint is not None and ri != rec_hint:
                            continue
                        bv, av, rv = nums[bi], nums[ai], nums[ri]
                        if bv <= 0 or av <= 0:
                            continue
                        if abs(rv - 100.0 * av / bv) <= tol_for(
                                bv, av, raws[bi], raws[ai], raws[ri]):
                            ok.append((bv, av))
            if not ok:
                continue
            if len(set(ok)) == 1:
                before, after = ok[0]
            elif len({frozenset(x) for x in ok}) == 1:
                # ambiguous sign only; |delta| is inside printing noise
                before, after = ok[0]
            else:
                continue
        elif len(nums) == 2:
            before, after = nums[0], nums[1]
        else:
            continue

        if not (0 < before <= 100 and 0 <= after <= 100):
            continue
        seen.add(b)
        out.append({"model": model_id, "scheme": sch, "benchmark": b,
                    "before": before, "after": after,
                    "delta": round(after - before, 4)})
    if out and max(r["before"] for r in out) <= 1.0:
        return []          # 0-1 scale card
    return out


# --------------------------------------------------------------------------
def conformal_q(vals, alpha):
    v = sorted(vals)
    n = len(v)
    k = math.ceil((n + 1) * (1 - alpha))
    return v[k - 1] if k <= n else float("inf")


def load_dir(d, allow_no_size):
    rows = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".md"):
            continue
        p = os.path.join(d, fn)
        if os.path.getsize(p) < 200:
            continue
        mid = fn[:-3].replace("_", "/", 1)
        if not allow_no_size and not re.search(
                r"\d+(\.\d+)?\s*[bB](?![a-zA-Z0-9])", mid):
            continue
        rows += read_card(p, mid)
    return rows


# strict-subset rule, written independently:
# a prospective row is EXCLUDED if its checkpoint belongs to a family that
# appears in training (llama-3.1, qwen3), or if it is Llama-4 (the parser was
# adapted to those cards after they were seen).
def is_strict(model_id):
    n = model_id.split("/")[-1]
    if re.search(r"Llama-3\.1", n, re.I):
        return False
    if re.search(r"(^|[-_])Qwen3(?![.\d])", n, re.I):
        return False
    if re.search(r"Llama-4", n, re.I):
        return False
    return True


def main():
    print("=" * 72)
    print("INDEPENDENT RE-VERIFICATION (stdlib only, no project imports)")
    print("=" * 72)
    print(f"  python {sys.version.split()[0]}   modules: "
          f"{', '.join(sorted(set(['re','os','csv','math','sys'])))}")

    train = load_dir(TRAIN_DIR, allow_no_size=False)
    train = [r for r in train if r["before"] >= MIN_ACC]
    print(f"\n  training rows parsed independently : {len(train)}")

    # build the envelope independently
    means, qhat, counts = {}, {}, {}
    for s in sorted({r["scheme"] for r in train}):
        g = [r["delta"] for r in train if r["scheme"] == s]
        mu = sum(g) / len(g)
        means[s] = mu
        qhat[s] = conformal_q([abs(x - mu) for x in g], ALPHA)
        counts[s] = len(g)
    print(f"\n  {'scheme':<14}{'n':>5}{'mean':>10}{'half-width':>12}")
    for s in sorted(means):
        print(f"  {s:<14}{counts[s]:>5}{means[s]:>+10.4f}{qhat[s]:>12.4f}")

    test = load_dir(TEST_DIR, allow_no_size=False)
    test = [r for r in test if r["before"] >= MIN_ACC]
    print(f"\n  prospective rows parsed independently: {len(test)}")

    for r in test:
        mu, q = means.get(r["scheme"]), qhat.get(r["scheme"])
        if mu is None:
            r["inside"] = None
            continue
        r["lo"], r["hi"] = mu - q, mu + q
        r["inside"] = (r["lo"] <= r["delta"] <= r["hi"])

    strict = [r for r in test if is_strict(r["model"]) and r["inside"]
              is not None]
    k, n = sum(1 for r in strict if r["inside"]), len(strict)
    kk, nn = (sum(1 for r in test if r["inside"]),
              sum(1 for r in test if r["inside"] is not None))
    print(f"\n  ALL prospective : {kk}/{nn} = {100*kk/nn:.1f}%")
    print(f"  STRICT subset   : {k}/{n} = {100*k/n:.1f}%")
    print(f"  strict groups   : "
          f"{sorted({r['model'].split('/')[-1].split('-quantized')[0][:24] for r in strict})}")

    # ---------------------------------------------------------------- compare
    print("\n" + "=" * 72)
    print("COMPARISON WITH THE PIPELINE")
    print("=" * 72)
    pipe_path = os.path.join(ROOT, "out", "real_use_case.csv")
    if not os.path.exists(pipe_path):
        print("  out/real_use_case.csv missing -- run src/real_use_case.py")
        return 1
    pipe = {}
    with open(pipe_path) as f:
        for row in csv.DictReader(f):
            pipe[(row["model"], row["benchmark"])] = row

    mine = {(r["model"], r["benchmark"]): r for r in test}
    only_mine = sorted(set(mine) - set(pipe))
    only_pipe = sorted(set(pipe) - set(mine))
    both = sorted(set(mine) & set(pipe))
    print(f"  rows found by BOTH            : {len(both)}")
    print(f"  found only by this verifier   : {len(only_mine)}")
    for m, b in only_mine[:12]:
        print(f"      + {m.split('/')[-1]:<50} {b}")
    print(f"  found only by the pipeline    : {len(only_pipe)}")
    for m, b in only_pipe[:12]:
        print(f"      - {m.split('/')[-1]:<50} {b}")

    dis_v, dis_i = [], []
    for key in both:
        a, b_ = mine[key], pipe[key]
        if abs(a["delta"] - float(b_["delta"])) > 1e-6:
            dis_v.append((key, a["delta"], float(b_["delta"])))
        if a["inside"] != (b_["inside_90"].strip().lower() == "true"):
            dis_i.append((key, a["inside"], b_["inside_90"]))
    print(f"\n  rows where DELTA disagrees    : {len(dis_v)}")
    for key, x, y in dis_v[:15]:
        print(f"      {key[0].split('/')[-1]:<46}{key[1]:<15}"
              f"mine={x:+.2f} pipeline={y:+.2f}")
    print(f"  rows where VERDICT disagrees  : {len(dis_i)}")
    for key, x, y in dis_i[:15]:
        print(f"      {key[0].split('/')[-1]:<46}{key[1]:<15}"
              f"mine={x} pipeline={y}")

    ok = (not dis_v) and (not dis_i) and not only_mine and not only_pipe
    print("\n" + "=" * 72)
    print(f"INDEPENDENT VERDICT: strict coverage = {k}/{n} = {100*k/n:.1f}%")
    print(f"  full agreement with the pipeline: {ok}")
    print("=" * 72)

    with open(os.path.join(ROOT, "out", "independent_check.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "scheme", "benchmark",
                                          "before", "after", "delta", "lo",
                                          "hi", "inside"])
        w.writeheader()
        for r in test:
            w.writerow({kk2: r.get(kk2) for kk2 in w.fieldnames})
    print("wrote out/independent_check.csv")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
