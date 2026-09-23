"""
Parse RedHatAI model-card READMEs into rows of
(model, quant_config, benchmark, accuracy_before, accuracy_after).

Integrity design (see RESEARCH.md section 4, threat 6):
the cards print their own `Recovery %` column, so for every 3-number row we can
check  recovery ~= 100 * after / before  and REJECT the row if it disagrees.
That check also tells us the column ORIENTATION (base-first vs quant-first)
without trusting header text. Rows we cannot verify are kept but flagged.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from html.parser import HTMLParser

CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cards")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
REJECT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "rejected_rows.csv")

# ---------------------------------------------------------------- benchmarks
# n_items = number of scored items, used for the analytic noise floor
# (RESEARCH.md thread D). Order matters: more specific patterns first.
BENCHMARKS = [
    ("mmlu_pro", r"^mmlu[\s\-_]*pro", 12032),
    ("mmlu_cot", r"^mmlu\b.*\bcot", 14042),
    ("mmlu", r"^mmlu\b", 14042),
    ("arc_challenge", r"^arc[\s\-_]*(challenge|c\b)", 1172),
    ("gsm8k", r"^gsm[\s\-_]*8?k", 1319),
    ("hellaswag", r"^hellaswag", 10042),
    ("winogrande", r"^winogrande", 1267),
    ("truthfulqa", r"^truthful", 817),
    ("ifeval", r"^ifeval", 541),
    ("bbh", r"^bbh|^big[\s\-_]*bench", 6511),
    # "lv"/"vl"/"v" cover the "Math-|v|-5" spelling after pipes are stripped.
    # The (?!\d) is load-bearing: without it "MATH-500" matches here, and
    # MATH-500 (pass@1, ~95) is a DIFFERENT benchmark from Math-Lvl-5 /
    # Math-Hard (exact-match 4-shot, ~6-59). Caught by verify/independent_check.py.
    ("math_lvl5",
     r"^math[\s\-_]*(lvl|lv|vl|v|level)?[\s\-_]*5(?!\d)|^math[\s\-_]*hard",
     1324),
    ("gpqa", r"^gpqa", 448),
    ("musr", r"^musr", 756),
    ("humaneval_plus", r"^humaneval\+|^humaneval[\s\-_]*plus", 164),
    ("humaneval", r"^humaneval", 164),
    ("arena_hard", r"^arena[\s\-_]*hard", 500),
]
BENCH_N = {k: n for k, _, n in BENCHMARKS}

# a cell that is ENTIRELY a number: "73.5", "**73.5**", "105.4%",
# "25.8 (25.1 / 26.5)".  Deliberately rejects "MMLU (5-shot)".
NUM_CELL = re.compile(
    r"^\**\s*([-+]?\d+(?:\.\d+)?)\s*\**\s*%?\s*\**"
    r"(?:\(\s*[\d\.\s/,+-]+\s*\))?\s*\**$"
)

QUANT_HINT = re.compile(
    r"(fp8|w4a16|w8a8|w8a16|int8|int4|nvfp4|mxfp4|quantized|compressed|"
    r"this model|awq|gptq)", re.I
)


AGGREGATE_RE = re.compile(r"average|recovery|^score$")


def canon_benchmark(label: str):
    s = label.strip().strip("*").strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    # Some cards literally print "Math-|v|-5" where others print "Math-lvl-5".
    # Dropping pipes makes the whitelist tolerant of that source typo, which
    # otherwise silently loses 11 real Math-lvl-5 rows (found by src/census.py).
    s = s.replace("|", "")
    if not s or AGGREGATE_RE.search(s):
        return None
    for name, pat, _ in BENCHMARKS:
        if re.search(pat, s):
            return name
    return None


def is_aggregate(label: str) -> bool:
    s = label.strip().strip("*").strip().lower()
    return bool(AGGREGATE_RE.search(s))


# ---------------------------------------------------------------- HTML tables
class TableGrab(HTMLParser):
    """Collect <table> -> list of <tr> -> list of cell text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables, self._tbl, self._row, self._cell = [], None, None, None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._tbl = []
        elif tag == "tr" and self._tbl is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self._tbl.append(self._row)
            self._row = None
        elif tag == "table" and self._tbl is not None:
            if self._tbl:
                self.tables.append(self._tbl)
            self._tbl = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_tables(text):
    p = TableGrab()
    try:
        p.feed(text)
    except Exception:
        pass
    return p.tables


def md_tables(text):
    """Collect contiguous blocks of pipe-delimited lines as tables."""
    out, cur = [], []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.count("|") >= 3:
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue  # separator row
            cur.append(cells)
        else:
            if len(cur) >= 2:
                out.append(cur)
            cur = []
    if len(cur) >= 2:
        out.append(cur)
    return out


def num(cell):
    m = NUM_CELL.match(cell.strip())
    return float(m.group(1)) if m else None


def decimals(raw):
    """Decimal places the card actually printed, e.g. '81.31' -> 2."""
    m = re.search(r"\d+\.(\d+)", raw)
    return len(m.group(1)) if m else 0


def recovery_tolerance(a, b, raw_a, raw_b, raw_rec):
    """
    How far the printed recovery may legitimately sit from 100*b/a, given that
    a, b and rec were each rounded before printing. Propagating half-ULP:
        d(100 b/a) = 100 * (ulp_b/a + b*ulp_a/a^2)
    This is tight for high-accuracy rows (~0.06pp) and correctly loose for
    near-random rows like GPQA at 3.7 (~2.9pp), where rounding a to one
    decimal moves the ratio a lot.
    """
    ulp_a = 0.5 * 10 ** (-decimals(raw_a))
    ulp_b = 0.5 * 10 ** (-decimals(raw_b))
    ulp_r = 0.5 * 10 ** (-decimals(raw_rec))
    return 100.0 * (ulp_b / a + b * ulp_a / (a * a)) + ulp_r + 1e-9


def parse_row(cells):
    """-> (label, nums) where nums is the trailing run of purely-numeric cells."""
    vals = [num(c) for c in cells]
    j = len(cells)
    while j > 0 and vals[j - 1] is not None:
        j -= 1
    nums = [vals[k] for k in range(j, len(cells))]
    label = None
    for k in range(j - 1, -1, -1):
        if cells[k].strip():
            label = cells[k]
            break
    return label, nums


def table_layout(table):
    """
    Work out, from the header row, which of the numeric columns is Recovery and
    which is the quantized model.

    Necessary because the column order is NOT consistent across cards: most are
    [base, quant, recovery] but the Llama-4 cards are
    [recovery, base, quant]. Assuming a fixed order silently corrupts rows
    (caught by the real-use-case test, where a GPQA 'accuracy' of 100.00 was
    actually a recovery percentage).
    """
    width = None
    for row in table:
        _, nums = parse_row(row)
        if nums:
            width = len(nums)
            break
    if width is None:
        return None
    for row in table:
        _, nums = parse_row(row)
        if nums:
            continue  # a data row, not the header
        cells = [c for c in row if c.strip()]
        if len(cells) < width:
            continue
        tail = cells[-width:]
        rec_idx = next((i for i, c in enumerate(tail)
                        if re.search(r"recovery", c, re.I)), None)
        q_idx = next((i for i, c in enumerate(tail)
                      if QUANT_HINT.search(c)), None)
        if q_idx is not None and q_idx == rec_idx:
            q_idx = None
        return {"width": width, "rec_idx": rec_idx, "quant_idx": q_idx}
    return {"width": width, "rec_idx": None, "quant_idx": None}


def _candidate_tiers(layout):
    """
    Ordered tiers of candidate column assignments as INDEX triples
    (before_idx, after_idx, recovery_idx).

    Tiers are tried in order and the first tier with a passing candidate wins,
    so a reading the header actually supports is never outvoted by a
    numerically-coincidental one. Within a tier the two entries differ only in
    orientation (same pair of numbers), so any residual ambiguity is about the
    sign of delta, never its magnitude.
    """
    tiers = []
    if layout and layout.get("rec_idx") is not None and layout["width"] == 3:
        ri = layout["rec_idx"]
        rest = [i for i in range(3) if i != ri]
        qi = layout.get("quant_idx")
        if qi in rest:
            bi = [i for i in rest if i != qi][0]
            tiers.append([(bi, qi, ri)])
        else:
            tiers.append([(rest[0], rest[1], ri), (rest[1], rest[0], ri)])
    tiers.append([(0, 1, 2), (1, 0, 2)])   # recovery last  (the common case)
    tiers.append([(1, 2, 0), (2, 1, 0)])   # recovery first (Llama-4 cards)
    return tiers


def extract(text, diag=None):
    """
    Yield (benchmark, before, after, orientation) from one card.

    `diag`, if given, is a list that receives (reason, label) for every row
    that is skipped without being yielded, so that nothing is dropped silently.
    """
    # Narrow to the accuracy section: start at the Evaluation/Accuracy heading
    # (deployment instructions come BEFORE it in these cards), then cut at the
    # first throughput/latency heading that follows it.
    start = 0
    for marker in ("## Evaluation", "## Accuracy", "### Accuracy",
                   "## Model Evaluation", "### Evaluation"):
        i = text.find(marker)
        if i > 0:
            start = i
            break
    cut = len(text)
    for marker in ("## Inference Performance", "### Inference Performance",
                   "Inference Performance", "## Performance", "## Deployment"):
        i = text.find(marker, start + 1)
        if i > 0:
            cut = min(cut, i)
    body = text[start:cut]

    tables = [(t, table_layout(t)) for t in html_tables(body)]
    tables += [(t, table_layout(t)) for t in md_tables(body)]

    for table, layout in tables:
        swap = bool(layout and layout.get("quant_idx") == 0
                    and layout.get("rec_idx") != 0)
        for cells in table:
            label, nums = parse_row(cells)
            if label is None or not nums:
                continue
            bench = canon_benchmark(label)
            if bench is None:
                if diag is not None:
                    lab = re.sub(r"\s+", " ", label.strip().strip("*"))[:60]
                    diag.append(("aggregate_row_ignored" if is_aggregate(label)
                                 else "benchmark_not_whitelisted", lab))
                continue

            raw = cells[len(cells) - len(nums):]
            if len(nums) not in (2, 3):
                if diag is not None:
                    diag.append((f"numeric_run_len_{len(nums)}", bench))
                continue

            if len(nums) == 3:
                if min(nums) <= 0:
                    yield bench, None, None, "reject"
                    continue
                # Test candidate column assignments, tier by tier, against the
                # card's own recovery arithmetic. First tier that reproduces it
                # wins.
                passing = []
                for tier in _candidate_tiers(layout):
                    for bi, ai, ri in tier:
                        before, after, rec = nums[bi], nums[ai], nums[ri]
                        tol = recovery_tolerance(before, after, raw[bi],
                                                 raw[ai], raw[ri])
                        if abs(rec - 100.0 * after / before) <= tol:
                            passing.append((before, after))
                    if passing:
                        break
                if not passing:
                    # no reading reproduces the printed recovery -> we have
                    # almost certainly misread the columns. Refuse the row.
                    yield bench, None, None, "reject"
                elif len(set(passing)) == 1:
                    yield bench, passing[0][0], passing[0][1], "arith"
                else:
                    # same pair of numbers, ambiguous sign: only possible when
                    # |delta| is within printing noise. Take sign from header.
                    a, b = passing[0]
                    lo_, hi_ = (b, a) if swap else (a, b)
                    yield bench, lo_, hi_, "header"
            elif len(nums) == 2:
                a, b = nums
                before, after = (b, a) if swap else (a, b)
                yield bench, before, after, "header_only"
            # len(nums) not in (2,3): ambiguous, skip silently


# ---------------------------------------------------------------- config parse
SCHEMES = [
    ("nvfp4a16", r"NVFP4A16", 4, 16, "fp"),
    ("nvfp4", r"NVFP4", 4, 4, "fp"),
    ("mxfp4", r"MXFP4", 4, 4, "fp"),
    ("w4a16", r"quantized\.w4a16|[-.]W4A16|INT4", 4, 16, "int"),
    ("w8a8_int", r"quantized\.w8a8|[-.]W8A8(?!-FP)|[-.]INT8", 8, 8, "int"),
    ("w8a16", r"quantized\.w8a16|[-.]W8A16", 8, 16, "int"),
    ("fp8_dynamic", r"FP8[-_]?dynamic", 8, 8, "fp"),
    ("fp8_block", r"FP8[-_]?block", 8, 8, "fp"),
    ("fp8", r"FP8", 8, 8, "fp"),
]


def parse_config(model_id):
    for name, pat, wb, ab, num_t in SCHEMES:
        if re.search(pat, model_id, re.I):
            return name, wb, ab, num_t
    return None, None, None, None


def parse_params_b(model_id):
    m = re.findall(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z0-9])", model_id)
    return float(m[0]) if m else None


# Family patterns are anchored at a name boundary and forbid a trailing digit,
# so that e.g. "diffusiongemma-26B-A4B" is NOT matched as "gemma-2"
# (audit 1 caught exactly that).
_B = r"(?:^|[-_/])"
FAMILIES = [
    ("llama-3.1", _B + r"(?:Meta-)?Llama-3\.1(?![\d.])"),
    ("llama-3.2", _B + r"(?:Meta-)?Llama-3\.2(?![\d.])"),
    ("llama-3.3", _B + r"(?:Meta-)?Llama-3\.3(?![\d.])"),
    ("llama-4", _B + r"Llama-4(?![\d.])"),
    ("qwen2.5", _B + r"Qwen2\.5(?![\d.])"),
    ("qwen3", _B + r"Qwen3(?![\d.])"),
    ("mistral", _B + r"(?:Mistral|Mixtral)(?![\d.])"),
    ("phi-3", _B + r"Phi-3(?![\d.])"),
    ("gemma-2", _B + r"gemma-2(?![\d.])"),
    ("granite", _B + r"granite(?![\d.])"),
]


def parse_family(model_id):
    for name, pat in FAMILIES:
        if re.search(pat, model_id, re.I):
            return name
    return "other"


def base_model_name(model_id):
    """Strip the quantization suffix -> the underlying checkpoint identity."""
    s = model_id.split("/", 1)[-1]
    s = re.sub(
        r"[-.](quantized\.w[48]a(?:16|8)|FP8[-_]?dynamic|FP8[-_]?block|FP8|"
        r"INT8|INT4|NVFP4A16|NVFP4|MXFP4|W4A16|W8A8|W8A16)$",
        "", s, flags=re.I,
    )
    return s


METHOD_PATS = [
    ("gptq", r"\bGPTQ\b"),
    ("awq", r"\bAWQ\b"),
    ("smoothquant", r"\bSmoothQuant\b"),
    ("rtn", r"round[-\s]?to[-\s]?nearest|\bRTN\b"),
]


def parse_method(card_text, scheme):
    found = {n for n, p in METHOD_PATS if re.search(p, card_text, re.I)}
    if "gptq" in found and "smoothquant" in found:
        return "smoothquant+gptq"
    for m in ("gptq", "awq", "smoothquant", "rtn"):
        if m in found:
            return m
    # documented fallback: 8-bit float schemes are RTN by default in llm-compressor
    return "rtn" if scheme and scheme.startswith(("fp8", "nvfp4", "mxfp4")) else "unknown"


def harvest_card(path, model_id, allow_unknown_family=False):
    """
    Parse one card -> (rows, rejects). Shared by the training harvest and by
    the prospective real-use-case test, so both go through identical logic.
    """
    rows, rejects = [], []
    if os.path.getsize(path) < 200:
        return rows, [[model_id, "-", "empty_or_missing_card"]]
    text = open(path, encoding="utf-8", errors="replace").read()

    scheme, wb, ab, num_t = parse_config(model_id)
    if scheme is None:
        return rows, [[model_id, "-", "no_recognized_scheme"]]
    params = parse_params_b(model_id)
    if params is None:
        return rows, [[model_id, "-", "no_param_count_in_name"]]
    method = parse_method(text, scheme)
    base = base_model_name(model_id)
    is_instruct = int(bool(re.search(r"instruct|-it\b|chat", model_id, re.I)))

    family = parse_family(model_id)
    if family == "other" and not allow_unknown_family:
        return rows, [[model_id, "-", "unrecognized_family"]]

    diag = []
    parsed = list(extract(text, diag=diag))
    rejects += [[model_id, lab, why] for why, lab in diag]
    # Scale guard: a few cards report accuracy on a 0-1 scale instead of
    # 0-100. Mixing the two silently shrinks those deltas ~100x, so reject
    # the whole card rather than guess. (Audit 1 caught this.)
    accs = [b for _, b, _, _ in parsed if b is not None]
    if accs and max(accs) <= 1.0:
        return rows, [[model_id, "-", "accuracy_on_0_1_scale"]]

    seen = set()
    for bench, before, after, orient in parsed:
        if before is None:
            rejects.append([model_id, bench, "recovery_mismatch"])
            continue
        if bench in seen:
            # First table wins. This is a CHOICE, not a no-op: some cards
            # report the same benchmark twice under different conditions
            # (e.g. GPQA 30.12 in OpenLLM-v2 vs 62.94 in a reasoning table).
            # Keeping the first preserves the OpenLLM-v1/v2 protocol
            # consistently, but the discarded value is logged.
            rejects.append([model_id, bench, "duplicate_benchmark_discarded"])
            continue
        seen.add(bench)
        if not (0 < before <= 100 and 0 <= after <= 100):
            rejects.append([model_id, bench, "out_of_range"])
            continue
        rows.append({
            "model": model_id, "base_model": base, "family": family,
            "params_b": params, "is_instruct": is_instruct,
            "scheme": scheme, "weight_bits": wb, "act_bits": ab,
            "num_type": num_t, "method": method,
            "benchmark": bench, "n_items": BENCH_N[bench],
            "acc_before": before, "acc_after": after,
            "delta": round(after - before, 4),
            "orient_src": orient,
            "verified": int(orient in ("arith", "header")),
        })
    return rows, rejects


def main():
    rows, rejects = [], []
    cards = sorted(f for f in os.listdir(CARDS_DIR) if f.endswith(".md"))
    n_cards_used = 0

    for fn in cards:
        path = os.path.join(CARDS_DIR, fn)
        model_id = fn[:-3].replace("_", "/", 1)
        r, rj = harvest_card(path, model_id)
        # keep the original reject log shape: only report cards that had a
        # scheme and a size but still failed
        rejects += [x for x in rj if x[2] not in
                    ("empty_or_missing_card", "no_recognized_scheme",
                     "no_param_count_in_name")]
        rows += r
        kept = len(r)
        if kept:
            n_cards_used += 1

    cols = list(rows[0].keys())
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(REJECT_CSV, "w", newline="") as f:
        csv.writer(f).writerows([["model", "benchmark", "reason"]] + rejects)

    import collections
    osrc = collections.Counter(r["orient_src"] for r in rows)
    print(f"cards on disk      : {len(cards)}")
    print(f"cards contributing : {n_cards_used}")
    print(f"rows               : {len(rows)}")
    print(f"  orientation+magnitude verified by card arithmetic : {osrc['arith']}")
    print(f"  magnitude verified, sign from header (|delta|~0)  : {osrc['header']}")
    print(f"  2-column table, no recovery to check against      : {osrc['header_only']}")
    print(f"rejected rows      : {len(rejects)}")
    print(f"distinct base models: {len({r['base_model'] for r in rows})}")
    print(f"families           : {sorted({r['family'] for r in rows})}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
