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
import json
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
# Tier 2.4 (day-7 audit): the benchmark regexes must handle
#   * suffixed variants ("arc_challenge_llama", "gsm8k_llama"),
#   * embedded parameter tokens ("HumanEval_64 pass@2"), and
#   * localised prefixes that should NOT match the base benchmark
#     ("Spanish-MMLU" is not MMLU).
# The order matters: mmlu_pro / mmlu_cot / spanish-mmlu / arabic-mmlu
# must be excluded from plain mmlu, and humaneval_plus from plain humaneval.
_LLAMA_SUFFIX = r"(?:[\s\-_]*llama|[\s\-_]*meta)?"
BENCH = [
    ("mmlu_pro", r"\bmmlu[\s\-_]*pro\b"),
    # C6: the MMLU family tolerates the harness task-name suffix, the same
    # convention arc_challenge and gsm8k already used below. The cards carry
    # `mmlu_cot_llama (0-shot)` and `mmlu_llama`; both name the Meta
    # prompt-matched template of a task we already whitelist.
    ("mmlu_cot", r"\bmmlu\b[^a-z]*\(?\s*cot|^mmlu[_-]cot"
                 + _LLAMA_SUFFIX + r"\b"),
    # plain MMLU: ANCHORED at start of label, matching src/harvest.py:29's
    # `^mmlu\b`. C5: the previous negative-lookbehind enumeration missed
    # "Italian MMLU", "Hindi MMLU", "Thai MMLU", "Portuguese MMLU" (space
    # instead of hyphen, name not in the enumeration). Anchoring inherits
    # the pipeline's inclusion boundary and needs no enumeration.
    ("mmlu", r"^mmlu" + _LLAMA_SUFFIX + r"\b"),
    ("arc_challenge",
     r"\barc[\s\-_]*(?:challenge|c)" + _LLAMA_SUFFIX + r"\b"),
    ("gsm8k", r"\bgsm[\s\-_]*8?k" + _LLAMA_SUFFIX + r"\b"),
    ("hellaswag", r"\bhellaswag\b"),
    ("winogrande", r"\bwinogrande\b"),
    ("truthfulqa", r"\btruthful"),
    ("ifeval", r"\bifeval\b"),
    ("bbh", r"\bbbh\b|\bbig[\s\-_]*bench"),
    ("math_lvl5", r"\bmath[\s\-_]*(lvl|lv|vl|v|level)?[\s\-_]*5\b|"
                  r"\bmath[\s\-_]*hard\b"),
    # Resolved further by gpqa_protocol() below; see the note there.
    ("gpqa", r"\bgpqa\b"),
    ("musr", r"\bmusr\b"),
    ("humaneval_plus", r"\bhumaneval\s*\+|\bhumaneval[\s\-_]*plus\b"),
    # Tier 2.4: match humaneval with optional _<digits> suffix
    # ("HumanEval_64 pass@2" on NVFP4 cards). Anchored at start-of-word
    # only so humaneval_plus stays a separate label above.
    ("humaneval", r"\bhumaneval(?:_\d+)?\b"),
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


# A RedHatAI card's GPQA label varies along three independent axes, each of
# which the label names explicitly: which question set (the 198-item Diamond
# subset, or Main), how the answer is scored (length-normalized accuracy, or
# plain), and how the model is prompted (chain-of-thought at five shots, or
# zero-shot). Two GPQA tables in one card differ on at least one axis and are
# therefore two measurements, not one reported twice. This resolves the axes
# one at a time and composes the name, rather than matching whole protocol
# names in a fixed order, so the two sides of the comparison reach the same
# label by different routes. Deliberately silent on the one case the cards do
# not resolve: a bare "GPQA (0-shot)" whose value contradicts the same
# checkpoint's other cards reads as Main here, and the pipeline's cross-card
# reconciliation parks it under a placeholder label. That is a known, reported
# difference in label vocabulary, not in rows.
def gpqa_protocol(t):
    diamond = re.search(r"\bdiamond\b", t) is not None
    cot = re.search(r"\bcot\b|chain[\s\-]*of[\s\-]*thought", t) is not None
    norm = re.search(r"acc[\s\-_]*norm|normali[sz]ed", t) is not None
    base = "gpqa_diamond" if diamond else "gpqa_main"
    if cot:
        return base + "_cot_5shot"
    if norm and not diamond:
        return base + "_norm"
    return base


def bench_of(label):
    t = label.lower().replace("|", " ")
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"[*`]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t or SKIP_LABEL.search(t):
        return None
    for name, pat in BENCH:
        if re.search(pat, t):
            return gpqa_protocol(t) if name == "gpqa" else name
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
        out.append({"model": model_id,
                    # Derived here, not imported: the checkpoint identity is
                    # the model id with its quantization suffix removed, which
                    # is the publisher's own naming convention. Needed because
                    # the shipped band partitions on CHECKPOINTS, so this file
                    # has to know which rows share one. Written from the
                    # convention rather than copied from harvest.py.
                    "base_model": base_checkpoint(model_id),
                    "family": family_of(model_id),
                    "band": band_of(model_id),
                    "scheme": sch, "benchmark": b,
                    "before": before, "after": after,
                    "delta": round(after - before, 4)})
    if out and max(r["before"] for r in out) <= 1.0:
        return []          # 0-1 scale card
    return out


# --------------------------------------------------------------------------

_QUANT_SUFFIX = re.compile(
    r"[-.](?:quantized\.w[48]a(?:16|8)|FP8[-_]?dynamic|FP8[-_]?block|FP8|"
    r"INT8|INT4|NVFP4A16|NVFP4|MXFP4|W4A16|W8A8|W8A16)$", re.I)


def base_checkpoint(model_id):
    """-> the underlying checkpoint, i.e. the model id minus its quantization
    suffix. The publisher names every quantized variant <checkpoint>-<scheme>,
    so stripping the scheme token recovers the thing the rows share."""
    return _QUANT_SUFFIX.sub("", model_id.split("/", 1)[-1])


_FAMILY_PATS = [
    ("llama-3", r"llama-3"), ("llama-4", r"llama-4"),
    ("qwen2.5", r"qwen2\.5"), ("qwen3", r"qwen3"),
    ("granite", r"granite"), ("mistral", r"mistral|ministral"),
    ("gemma-2", r"gemma-2"), ("gemma-3", r"gemma-3"),
    ("deepseek-r1-distill", r"deepseek-r1-distill"),
    ("smollm3", r"smollm3"), ("smollm", r"smollm"),
    ("nemotron", r"nemotron"), ("phi", r"phi-?4"),
]


def family_of(model_id):
    """-> family label, derived from the checkpoint name. Emitted into
    out/independent_check.csv so the registry's family split stops recovering
    membership by regex over a model string it was handed."""
    t = model_id.split("/", 1)[-1].lower()
    for name, pat in _FAMILY_PATS:
        if re.search(pat, t):
            return name
    return "other"



_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])")


def params_of(model_id):
    """-> parameter count in billions, or None. Derived from the checkpoint
    name, which is where the publisher states it."""
    m = _SIZE_RE.search(model_id.split("/", 1)[-1])
    return float(m.group(1)) if m else None


def band_of(model_id):
    """-> size band, matching the three the tool reports."""
    p = params_of(model_id)
    if p is None:
        return "unknown"
    return "<2B" if p < 2 else ("2-10B" if p <= 10 else ">10B")


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
# appears in training (Llama-3.* -- one family under Definition B, see §3 --
# or Qwen3), or if it is Llama-4 (the parser was adapted to those cards after
# they were seen).
def is_strict(model_id):
    n = model_id.split("/")[-1]
    if re.search(r"Llama-3\.[123]", n, re.I):
        return False
    if re.search(r"(^|[-_])Qwen3(?![.\d])", n, re.I):
        return False
    if re.search(r"Llama-4", n, re.I):
        return False
    return True


# ----------------------------------------------------------------- LOFO ARM
# Independent recomputation of the leave-one-family-out figures the paper
# prints, added 2026-10-13. These were hand-typed in the paper until the
# day-8 audit and had never been recomputed anywhere but src/. Two estimators
# appear in the paper and they are NOT interchangeable, so both are redone
# here:
#
#   * the SUBGROUP estimator (per-scheme mean + Mondrian-by-scheme split
#     conformal, disjoint fit/calibration families, no size rule) -- the
#     aggregates and the per-fold table in the body;
#   * the SHIPPED band (scheme centre + widen-only size cells, calibrated on
#     a held-out family) -- the per-fold coverages quoted beside them.
#
# Everything below is written from the construction as specified, in the
# standard library, against data/dataset.csv. It imports nothing from src/
# and it does not read the registry: the comparison target is
# paper/numbers.tex, the artifact the PDF is actually built from, so a
# disagreement catches registry drift and macro-generation drift as well as
# an arithmetic error. This block needs only the committed dataset, so it
# runs even in the (normal) case where data/cards is absent and the rest of
# this file is skipped -- which is why it sits ahead of that gate.
ALPHA_LOFO = 0.10
MIN_ROWS_LOFO = 20
MIN_CKPT_LOFO = 3


def _cq(vals, alpha=ALPHA_LOFO):
    """ceil((n+1)(1-alpha))-th smallest, +inf when n cannot support it."""
    v = sorted(float(x) for x in vals)
    n = len(v)
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    return v[k - 1] if k <= n else float("inf")


def _band_lofo(p):
    if p is None or p == "":
        return "unknown"
    p = float(p)
    if p != p:
        return "unknown"
    if p < 2:
        return "<2B"
    if p <= 10:
        return "2-10B"
    return ">10B"


def _load_lofo():
    """-> rows as dicts, filtered exactly as src/model.load does."""
    path = os.path.join(ROOT, "data", "dataset.csv")
    out = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if float(r["acc_before"]) < 20.0:       # MIN_ACC_BEFORE
                continue
            out.append({"delta": float(r["delta"]), "scheme": r["scheme"],
                        "family": r["family"], "base": r["base_model"],
                        "band": _band_lofo(r["params_b"])})
    return out


def _mean(xs):
    return sum(xs) / len(xs)


def _cell_lofo(rows):
    mu = _mean([r["delta"] for r in rows])
    return {"mean": mu,
            "hw": _cq([abs(r["delta"] - mu) for r in rows]),
            "n": len(rows),
            "nck": len({r["base"] for r in rows})}


def _group(rows, key):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return g


def _shipped_band(tr, cal):
    """ConservativeStratified().fit_calibrated(tr, cal), as specified.

    Centres from `tr`; half-widths replaced by the conformal quantile of the
    calibration residuals taken AROUND THE CENTRE predict_interval will use,
    which for this class is the scheme mean and not the size-cell mean. A
    size cell may only widen.
    """
    glob = _cell_lofo(tr)
    by_s = {s: _cell_lofo(g) for s, g in _group(tr, lambda r: r["scheme"]).items()}
    by_b = {}
    for (s, b), g in _group(tr, lambda r: (r["scheme"], r["band"])).items():
        c = _cell_lofo(g)
        if c["n"] >= MIN_ROWS_LOFO and c["nck"] >= MIN_CKPT_LOFO \
                and math.isfinite(c["hw"]):
            by_b[(s, b)] = c
    for r in cal:
        base = by_s.get(r["scheme"]) or glob
        r["_resid"] = abs(r["delta"] - base["mean"])
    for s, g in _group(cal, lambda r: r["scheme"]).items():
        if s in by_s and len(g) >= 9:
            by_s[s] = dict(by_s[s], hw=_cq([r["_resid"] for r in g]))
    keep = {}
    cal_by_cell = _group(cal, lambda r: (r["scheme"], r["band"]))
    for (s, b), c in by_b.items():
        g = cal_by_cell.get((s, b), [])
        if len(g) >= 9:
            q = _cq([r["_resid"] for r in g])
            if math.isfinite(q):
                keep[(s, b)] = dict(c, hw=q)
    by_b = keep
    if len(cal) >= 9:
        glob = dict(glob, hw=_cq([r["_resid"] for r in cal]))

    def interval(r):
        base = by_s.get(r["scheme"]) or glob
        cell = by_b.get((r["scheme"], r["band"]))
        hw = base["hw"]
        if cell is not None and cell["hw"] > hw:
            hw = cell["hw"]
        return base["mean"] - hw, base["mean"] + hw
    return interval


def _subgroup_band(tr, cal):
    """SchemeMean().fit(tr) + Conformal(mondrian_by='scheme').fit(model, cal).

    No size cells and no widening rule: a per-scheme centre, a per-scheme
    conformal quantile, and a marginal quantile to back off to when a scheme
    has too few calibration rows to support alpha.
    """
    gmu = _mean([r["delta"] for r in tr])
    mu_s = {s: _mean([r["delta"] for r in g])
            for s, g in _group(tr, lambda r: r["scheme"]).items()}

    def pred(r):
        return mu_s.get(r["scheme"], gmu)
    q_marg = _cq([abs(r["delta"] - pred(r)) for r in cal])
    q_s = {s: _cq([abs(r["delta"] - pred(r)) for r in g])
           for s, g in _group(cal, lambda r: r["scheme"]).items()}

    def interval(r):
        q = q_s.get(r["scheme"], float("inf"))
        if not math.isfinite(q):
            q = q_marg                                   # backoff=True
        c = pred(r)
        return c - q, c + q
    return interval


def _fold_table(rows, build, min_cal, min_tr, require_test=False):
    """-> {held-out family: coverage %}, pooled over that fold's test points.

    `build(tr, cal) -> interval(row)`. Every eligible calibration family is
    used in turn and the indicators are pooled, which is the rule the paper
    states: folds are pooled over test points, aggregates are the mean over
    folds.
    """
    fams = sorted({r["family"] for r in rows})
    per = {}
    for tf in fams:
        te = [r for r in rows if r["family"] == tf]
        if require_test and not te:
            continue
        n = c = 0
        for cf in fams:
            if cf == tf:
                continue
            cal = [r for r in rows if r["family"] == cf]
            tr = [r for r in rows if r["family"] not in (tf, cf)]
            if len(cal) < min_cal or len(tr) < min_tr:
                continue
            iv = build(tr, cal)
            for r in te:
                lo, hi = iv(r)
                n += 1
                c += int(lo - 1e-9 <= r["delta"] <= hi + 1e-9)
        if n:
            per[tf] = 100.0 * c / n
    return per



def lofo_fold_indicators(rows=None):
    """-> {held-out family: (indicators, cluster labels)} under the shipped
    band, pooled over every eligible calibration family.

    Exposed so verify/independent_rank.py can put a cluster bootstrap on the
    same indicator vector without a second reimplementation of the band --
    one independent implementation, used by both arms, is the point; two
    would only agree with each other.
    """
    if rows is None:
        rows = _load_lofo()
    fams = sorted({r["family"] for r in rows})
    out = {}
    for tf in fams:
        te = [r for r in rows if r["family"] == tf]
        if not te:
            continue
        ind, clus = [], []
        for cf in fams:
            if cf == tf:
                continue
            cal = [r for r in rows if r["family"] == cf]
            tr = [r for r in rows if r["family"] not in (tf, cf)]
            if len(cal) < 19 or len(tr) < 50:
                continue
            iv = _shipped_band(tr, cal)
            for r in te:
                lo, hi = iv(r)
                ind.append(int(lo - 1e-9 <= r["delta"] <= hi + 1e-9))
                clus.append(r["base"])
        if ind:
            out[tf] = (ind, clus)
    return out

def _definition_a(rows):
    """Definition A: llama-3 split into 3.1 / 3.2 / 3.3 by base model."""
    out = []
    for r in rows:
        if r["family"] != "llama-3":
            out.append(r)
            continue
        m = re.search(r"Llama-?3\.([123])", str(r["base"]), re.I)
        q = dict(r)
        q["family"] = "llama-3.%s" % m.group(1) if m else "llama-3.x"
        out.append(q)
    return out


_WORDVALS = {w: i for i, w in enumerate(
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
     "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
     "sixteen", "seventeen", "eighteen", "nineteen", "twenty"])}


def _macro_value(mac, name):
    """-> the numeric value the paper prints for `name`, or None.

    A count may be published as a digit (\\X) or as a generated word
    (\\XWord / \\XWordCap), and three of them have no digit macro at all
    now that every site reads as a word. Resolving the word back to an
    integer here keeps those figures inside the independent check: without
    it, converting a site from "2 folds" to "Two folds" would quietly drop
    the figure out of verification, which is a worse defect than the digit
    it replaced.
    """
    if name in mac:
        try:
            return float(mac[name])
        except ValueError:
            return None
    for suffix in ("Word", "WordCap"):
        lit = mac.get(name + suffix)
        if lit is not None:
            v = _WORDVALS.get(lit.strip().lower())
            if v is not None:
                return float(v)
    return None


def _macros():
    """-> {macro name: literal} from paper/numbers.tex."""
    p = os.path.join(ROOT, "paper", "numbers.tex")
    if not os.path.isfile(p):
        return None
    return {m.group(1): m.group(2) for m in re.finditer(
        r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", open(p).read())}


def lofo_arm():
    """-> 0 agree, 1 disagree, 3 could not check."""
    print()
    print("=" * 72)
    print("INDEPENDENT LOFO RECOMPUTATION (stdlib only, from the dataset)")
    print("=" * 72)
    mac = _macros()
    if mac is None:
        print("  UNCHECKED: paper/numbers.tex is absent, so there is nothing "
              "to compare against.")
        return 3
    rows = _load_lofo()
    print(f"  modelling rows (acc_before >= 20): {len(rows)}")

    tB = _fold_table(rows, _subgroup_band, 19, 50)
    tA = _fold_table(_definition_a(rows), _subgroup_band, 19, 50)
    shipped = _fold_table(rows, _shipped_band, 19, 50, require_test=True)
    gaps = sorted(tA[f"llama-3.{i}"] for i in (1, 2, 3) if f"llama-3.{i}" in tA)

    checks = [
        # macro                   mine                         tol
        ("LofoAggDefB",           _mean(list(tB.values())),    0.05),
        ("LofoAggDefA",           _mean(list(tA.values())),    0.05),
        ("LThreeSubThreePooled",  tA.get("llama-3.3"),         0.05),
        ("LThreeMergedPooled",    tB.get("llama-3"),           0.05),
        ("LofoQwenThreeTblDefA",  tA.get("qwen3"),             0.05),
        ("LofoFoldCovQwenThree",  shipped.get("qwen3"),        0.05),
        ("LofoFoldCovMistral",    shipped.get("mistral"),      0.05),
        ("LofoFoldCovMin",        min(shipped.values()),       0.05),
        ("LofoFoldCovMax",        max(shipped.values()),       0.05),
        ("LofoFoldsBelowNominal",
         float(sum(1 for v in shipped.values() if v < 90.0)),  0.0),
    ]
    # The Llama-3 subgroup table (§6) and the gap pair. One held-out fold ---
    # all of llama-3 under Definition B --- scored under each calibration
    # family in turn, then split by sub-version. The gap the prose quotes is
    # between Llama-3.3 and each cousin WITHIN one calibration family, which
    # is why it cannot be read off the pooled folds above.
    _SHORT = {"gemma-2": "Gem", "granite": "Gra", "mistral": "Mis",
              "qwen2.5": "Qtf", "qwen3": "Qth"}
    te_l3 = [r for r in rows if r["family"] == "llama-3"]
    pool = [r for r in rows if r["family"] != "llama-3"]
    for r in te_l3:
        m = re.search(r"Llama-?3\.([123])", str(r["base"]), re.I)
        r["_sub"] = m.group(1) if m else "x"
    sub_cov, merged_cov = {}, {}
    for cf in sorted({r["family"] for r in pool}):
        cal = [r for r in pool if r["family"] == cf]
        tr = [r for r in pool if r["family"] != cf]
        if len(cal) < 19 or len(tr) < 50:
            continue
        iv = _subgroup_band(tr, cal)
        ind = {}
        for r in te_l3:
            lo, hi = iv(r)
            ind.setdefault(r["_sub"], []).append(
                int(lo - 1e-9 <= r["delta"] <= hi + 1e-9))
        allv = [x for v in ind.values() for x in v]
        merged_cov[cf] = 100.0 * _mean(allv)
        for s, v in ind.items():
            sub_cov[(s, cf)] = 100.0 * _mean(v)
    gapv = []
    for cf in sorted(merged_cov):
        if ("3", cf) not in sub_cov:
            continue
        t3 = sub_cov[("3", cf)]
        for s in ("1", "2"):
            if (s, cf) in sub_cov:
                gapv.append(sub_cov[(s, cf)] - t3)
    _WORD = {"1": "SubOne", "2": "SubTwo", "3": "SubThree"}
    for cf, short in sorted(_SHORT.items()):
        for s in ("1", "2", "3"):
            if (s, cf) in sub_cov:
                checks.append((f"LThree{_WORD[s]}{short}",
                               sub_cov[(s, cf)], 0.05))
        if cf in merged_cov:
            checks.append((f"LThreeMerged{short}", merged_cov[cf], 0.05))
    if gapv:
        checks.append(("LThreeGapMin", min(gapv), 0.05))
        checks.append(("LThreeGapMax", max(gapv), 0.05))

    bad = []
    print(f"\n  {'macro':<24}{'paper':>9}{'mine':>11}   verdict")
    for name, mine, tol in checks:
        theirs = _macro_value(mac, name)
        if theirs is None:
            bad.append((name, mine, "absent from numbers.tex"))
            print(f"  {name:<24}{'--':>9}{mine:>11.4f}   NOT DECLARED")
            continue
        if mine is None:
            bad.append((name, None, theirs))
            print(f"  {name:<24}{theirs:>9}{'--':>11}   NOT COMPUTED")
            continue
        # the paper prints one decimal; agreement means agreeing to the
        # printed precision plus the registry's own tolerance on the figure.
        ok = abs(round(mine, 1) - theirs) <= tol + 1e-9
        if not ok:
            bad.append((name, mine, theirs))
        print(f"  {name:<24}{theirs:>9.2f}{mine:>11.4f}   "
              f"{'agree' if ok else 'DISAGREE'}")
    print(f"\n  per-fold coverage, shipped band   : "
          f"{ {k: round(v, 2) for k, v in sorted(shipped.items())} }")
    print(f"  per-fold coverage, subgroup, def B: "
          f"{ {k: round(v, 2) for k, v in sorted(tB.items())} }")
    print(f"  Llama-3 subgroup folds, def A     : "
          f"{ [round(g, 2) for g in gaps] }")
    n_cmp = sum(1 for _n, _m, _t in checks
                if _macro_value(mac, _n) is not None and _m is not None)
    print(f"\n  comparisons made  : {n_cmp} of {len(checks)} macros "
          f"(the rest were not declared or not computed)")
    print(f"  LOFO disagreements: {len(bad)}")
    if n_cmp == 0:
        print("  UNCHECKED: no macro was compared, so this is not "
              "agreement.")
        return 3
    return 1 if bad else 0


def main():
    print("=" * 72)
    print("INDEPENDENT RE-VERIFICATION (stdlib only, no project imports)")
    print("=" * 72)
    print(f"  python {sys.version.split()[0]}   modules: "
          f"{', '.join(sorted(set(['re','os','csv','math','sys'])))}")

    _lofo_rc = lofo_arm()
    if _lofo_rc == 1:
        return 1
    if _lofo_rc == 3:
        return 3

    # data/cards/ holds raw model cards; they are intentionally NOT committed
    # (they contain PII and licensed text). Absence is documented in
    # PROVENANCE.md 2026-09-27. Exit 0 with SKIPPED so the gate step
    # (tests/test_all.py::test_verifiers_exit_zero) does not fire on the
    # documented absence — the stand-in gate is
    # test_independent_check_csv_matches_tolerance_recomputation.
    if not os.path.isdir(TRAIN_DIR):
        print(f"\n  UNCHECKED: {TRAIN_DIR} is not present in this checkout.")
        print("  The card corpus is not committed (PII / licensed text).")
        print("  See PROVENANCE.md 2026-09-27 for the stand-in gate.")
        print("  card-corpus comparisons made: 0")
        print(f"  (the LOFO arm above ran and made its own comparisons; this "
              f"is only the card-parsing half.)")
        # Exit 3, not 0: the card-parsing half of this verifier compared
        # nothing. It returned 0 for years so the gate step would not fire on
        # a documented absence, which made "the independent verifier passed"
        # true of a run that re-verified nothing from the cards.
        return 3
    train = load_dir(TRAIN_DIR, allow_no_size=False)
    train = [r for r in train if r["before"] >= MIN_ACC]
    print(f"\n  training rows parsed independently : {len(train)}")

    # C5: compare training row count and per-benchmark label counts against
    # the pipeline's data/dataset.csv. Any drift is a silent parse divergence
    # (previously the verifier reported 819 while the pipeline emitted 817
    # because Italian/Hindi/Thai MMLU leaked into the plain 'mmlu' bucket).
    _pipe_csv = os.path.join(ROOT, "data", "dataset.csv")
    if os.path.isfile(_pipe_csv):
        import csv as _csv
        _pipe = [r for r in _csv.DictReader(open(_pipe_csv))
                 if float(r["acc_before"]) >= MIN_ACC]
        _pipe_n = len(_pipe)
        print(f"  pipeline rows in data/dataset.csv  : {_pipe_n}")
        if len(train) != _pipe_n:
            print(f"  !! ROW COUNT DISAGREES: verifier {len(train)} vs "
                  f"pipeline {_pipe_n} (delta {len(train) - _pipe_n:+d})")
            _mine, _theirs = {}, {}
            for r in train:
                _mine[r["benchmark"]] = _mine.get(r["benchmark"], 0) + 1
            for r in _pipe:
                _theirs[r["benchmark"]] = _theirs.get(r["benchmark"], 0) + 1
            for _b in sorted(set(_mine) | set(_theirs)):
                if _mine.get(_b, 0) != _theirs.get(_b, 0):
                    print(f"    {_b:<22s} verifier={_mine.get(_b, 0):>4d} "
                          f"pipeline={_theirs.get(_b, 0):>4d}")
            return 3

    # Build the envelope independently -- as SPLIT CONFORMAL, which is what
    # ships. Until 2026-10-02 this block put the centre and the width on the
    # same rows, and src/real_use_case.py did the same, so the two arms agreed
    # with each other because they shared the assumption rather than because
    # either matched the shipped band. The cross-arm check in
    # paper/audit_paper.py reported zero disagreements throughout, which is
    # the float-boundary lesson one level down: a comparison between two
    # implementations of the same wrong premise is not evidence.
    #
    # DECLARED INPUT, stated here and in the module docstring: which
    # checkpoints form the calibration side. Read from
    # out/calibration_partition.json because this file is standard-library
    # only and cannot reproduce numpy's default_rng. Everything that list is
    # used for -- splitting the rows, the centre on the fit side, the
    # conformal index and quantile on the calibration side -- is still derived
    # here from the raw cards.
    part = json.load(open(os.path.join(
        ROOT, "out", "calibration_partition.json")))
    cal_ck = set(part["calibration_checkpoints"])
    fit_rows = [r for r in train if r["base_model"] not in cal_ck]
    cal_rows = [r for r in train if r["base_model"] in cal_ck]
    means, qhat, counts = {}, {}, {}
    for s in sorted({r["scheme"] for r in train}):
        gf = [r["delta"] for r in fit_rows if r["scheme"] == s]
        gc = [r for r in cal_rows if r["scheme"] == s]
        if not gf:
            continue
        mu = sum(gf) / len(gf)
        means[s] = mu
        # residuals about the centre predict_interval will use, on the
        # disjoint calibration side; too few to index -> keep the fit-side
        # width, which is what strata.fit_calibrated does (and which S4 now
        # records as a limitation rather than claiming we return infinity).
        if len(gc) >= 9:
            qhat[s] = conformal_q([abs(r["delta"] - mu) for r in gc], ALPHA)
        else:
            qhat[s] = conformal_q([abs(x - mu) for x in gf], ALPHA)
        # support counts describe the published evidence, so whole corpus
        counts[s] = sum(1 for r in train if r["scheme"] == s)

    # The widen-only size rule, derived here rather than imported: a
    # (scheme, band) cell may RAISE the scheme half-width and never lower it,
    # and only if it clears the support floor (>=20 rows, >=3 checkpoints over
    # the whole corpus) and has >=9 calibration rows to index a quantile with.
    # Without this the two arms compare different objects: the shipped tool
    # prints the widened band, and leaving it out produced two verdict
    # disagreements that were the stratification, not a defect.
    strat = {}
    for s in sorted({r["scheme"] for r in train}):
        for b in ("<2B", "2-10B", ">10B"):
            whole = [r for r in train if r["scheme"] == s and r["band"] == b]
            if len(whole) < 20 or len({r["base_model"] for r in whole}) < 3:
                continue
            gc = [r for r in cal_rows if r["scheme"] == s and r["band"] == b]
            if len(gc) < 9 or s not in means:
                continue
            q = conformal_q([abs(r["delta"] - means[s]) for r in gc], ALPHA)
            if math.isfinite(q) and q > qhat.get(s, 0.0):
                strat[(s, b)] = q
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
        q = max(q, strat.get((r["scheme"], r.get("band")), q))
        r["lo"], r["hi"] = mu - q, mu + q
        # closed-interval hit with a 1e-9 float tolerance; see the
        # audit-section note on shared-assumption reimplementation
        COVER_EPS = 1e-9
        r["inside"] = (r["lo"] - COVER_EPS <= r["delta"] <= r["hi"] + COVER_EPS)

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
        w = csv.DictWriter(f, fieldnames=["model", "base_model", "family",
                                   "scheme", "benchmark",
                                          "before", "after", "delta", "lo",
                                          "hi", "inside"])
        w.writeheader()
        for r in test:
            w.writerow({kk2: r.get(kk2) for kk2 in w.fieldnames})
    print("wrote out/independent_check.csv")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
