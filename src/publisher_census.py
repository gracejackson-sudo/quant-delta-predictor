"""How many publishers actually release PAIRED base/quantized evaluations?

The single-publisher limitation in BIAS_CORRECTION.md was an assertion. This
measures it.

Method. For each publisher we sample quantized model cards from the Hugging
Face API and run them through `harvest.extract` -- the same parser that built
our corpus, not a fresh regex. A card "publishes paired evals" if the parser
yields at least one (benchmark, before, after) triple from it.

Positive control. RedHatAI must come back with a non-zero rate. If it does
not, the detector is broken and the script refuses to report, rather than
reporting a false negative about everyone else. An earlier ad-hoc regex failed
exactly this check.

Cards are cached under out/publisher_cards/ (gitignored) and NOT redistributed.
Only the aggregate counts are published.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harvest  # noqa: E402

OUT = os.path.join(HERE, "..", "out")
CACHE = os.path.join(OUT, "publisher_cards")
RESULT = os.path.join(OUT, "publisher_census.json")

# RedHatAI first: it is the positive control, not just another row.
PUBLISHERS = ["RedHatAI", "ModelCloud", "unsloth", "TheBloke", "Intel"]
PER_PUBLISHER = 40
QUANT_RE = re.compile(r"4bit|8bit|gptq|awq|w4a16|w8a8|w8a16|int4|int8|fp8|"
                      r"nvfp4|quantiz", re.I)
UA = {"User-Agent": "quant-delta-predictor/1.0 (research; contact via repo)"}


def get(url, timeout=25):
    try:
        req = urllib.request.Request(url, headers=UA)
        return urllib.request.urlopen(req, timeout=timeout).read().decode(
            "utf-8", "ignore")
    except Exception:
        return ""


def card_text(mid):
    """Fetch a card's README, caching it locally. Cards are not redistributed."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, mid.replace("/", "_") + ".md")
    if os.path.exists(p):
        return open(p, encoding="utf-8").read()
    t = get(f"https://huggingface.co/{mid}/raw/main/README.md")
    if t:
        open(p, "w", encoding="utf-8").write(t)
    return t


def quantized_models(author, limit):
    raw = get(f"https://huggingface.co/api/models?author={author}&limit=500")
    try:
        data = json.loads(raw)
    except Exception:
        return []
    return [m["id"] for m in data if QUANT_RE.search(m["id"])][:limit]


def paired_rows(text):
    """-> (usable_rows, arith_verified_rows).

    `usable` means the parser recovered a before/after pair at all.
    `arith` means the card also printed a Recovery figure that agrees with
    100*after/before, which is the self-verification our integrity gate needs.
    A publisher can release paired numbers without releasing that column.
    """
    if not text:
        return 0, 0
    try:
        rows = list(harvest.extract(text))
    except Exception:
        return 0, 0
    usable = [r for r in rows if r[1] is not None and r[2] is not None]
    arith = [r for r in usable if r[3] == "arith"]
    return len(usable), len(arith)


def survey(author):
    ids = quantized_models(author, PER_PUBLISHER)
    if not ids:
        return {"publisher": author, "quantized_models_listed": 0,
                "cards_fetched": 0, "cards_with_paired_evals": 0,
                "paired_rows_total": 0}
    with ThreadPoolExecutor(8) as ex:
        texts = list(ex.map(card_text, ids))
    fetched = [t for t in texts if t]
    pairs = [paired_rows(t) for t in fetched]
    return {
        "publisher": author,
        "quantized_models_listed": len(ids),
        "cards_fetched": len(fetched),
        "cards_with_paired_evals": sum(1 for u, _ in pairs if u > 0),
        "paired_rows_total": int(sum(u for u, _ in pairs)),
        "cards_with_verifiable_evals": sum(1 for _, a in pairs if a > 0),
        "arith_verified_rows": int(sum(a for _, a in pairs)),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    results = [survey(p) for p in PUBLISHERS]

    control = next(r for r in results if r["publisher"] == "RedHatAI")
    if control["cards_with_paired_evals"] == 0:
        raise SystemExit(
            "POSITIVE CONTROL FAILED: the parser found no paired evals in "
            "RedHatAI cards, so it cannot be trusted to detect their absence "
            "elsewhere. Refusing to report."
        )

    others = [r for r in results if r["publisher"] != "RedHatAI"]
    summary = {
        "publishers_checked": len(results),
        "other_publishers_checked": len(others),
        "cards_inspected_total": sum(r["cards_fetched"] for r in results),
        "cards_inspected_others": sum(r["cards_fetched"] for r in others),
        "others_with_paired_evals": sum(
            r["cards_with_paired_evals"] for r in others),
        "others_with_verifiable_evals": sum(
            r["cards_with_verifiable_evals"] for r in others),
        "others_arith_verified_rows": sum(
            r["arith_verified_rows"] for r in others),
        "control_arith_verified_rows": control["arith_verified_rows"],
        "control_publisher": "RedHatAI",
        "control_cards_fetched": control["cards_fetched"],
        "control_cards_with_paired_evals": control["cards_with_paired_evals"],
        "per_publisher": results,
    }
    json.dump(summary, open(RESULT, "w"), indent=2)

    w = max(len(r["publisher"]) for r in results)
    print(f"{'publisher':<{w}}  fetched  paired-cards  rows  verifiable-rows")
    for r in results:
        tag = "  <- control" if r["publisher"] == "RedHatAI" else ""
        print(f"{r['publisher']:<{w}}  {r['cards_fetched']:>7}  "
              f"{r['cards_with_paired_evals']:>12}  "
              f"{r['paired_rows_total']:>4}  {r['arith_verified_rows']:>15}{tag}")
    print(f"\nwrote {RESULT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
