"""A minimal reporting standard for quantized model cards, and a scorer.

Motivation. The publisher census (src/publisher_census.py) found that Red Hat
is alone among the publishers sampled in printing a Recovery column -- the
thing that lets a reader check a row against 100*after/before and reject it if
it disagrees. That is not a quirk of their template. It is the difference
between an evaluation a reader can verify and one they must simply believe.

Nobody appears to have written down what a quantized model card should report.
This proposes six checks, each mechanical and each answerable from the card
alone, and scores every cached card against them.

THE STANDARD
  C1 base_named        the unquantized parent is identified
  C2 before_reported   the base model's score is given
  C3 after_reported    the quantized model's score is given
  C4 recovery_printed  a recovery/retention figure is given
  C5 recovery_checks   that figure agrees with 100*after/before
  C6 harness_stated    the eval harness and/or shot count is named

C1-C3 make a result readable. C4-C5 make it *checkable*. C6 makes it
*comparable*. A card scoring 6 can be audited by a stranger; a card scoring 3
has to be taken on trust.

This scores cards, not models. A low score is a statement about reporting
practice and nothing else -- it says nothing about the quality of the
quantization itself.

A KNOWN BIAS IN THIS TOOL, MEASURED
  C2, C3 and C5 depend on harvest.extract, a parser written against Red Hat's
  table layout. It misses tables with more than three numeric columns, which
  other publishers use. Measured on the cached corpus, the parser fails to
  read a benchmark table it should have read on 15 non-Red Hat cards and 0
  Red Hat cards -- a one-directional bias in Red Hat's favour.

  So the parser-dependent scores UNDERSTATE other publishers by an unknown
  amount, and no overall "percentage of cards that are checkable" is reported
  here, because that number would inherit the bias.

  C1, C4 and C6 are plain text searches and do not depend on the parser. The
  C4 result -- whether a recovery figure is printed at all -- is the one
  conclusion this tool can support cleanly.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harvest  # noqa: E402

ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "out", "card_quality.json")
DIRS = [("RedHatAI-corpus", os.path.join(ROOT, "data", "cards")),
        ("census", os.path.join(ROOT, "out", "publisher_cards"))]

BASE_RE = re.compile(r"base_model\s*:|\bbase model\b|unquantized|original model|"
                     r"parent model|dense (?:model|baseline)", re.I)
REC_RE = re.compile(r"\brecovery\b|\bretention\b|\brecovered\b|% of baseline", re.I)
HARNESS_RE = re.compile(r"lm[-_ ]?eval|lm[-_ ]evaluation[-_ ]harness|"
                        r"\b\d+[-\s]?shot\b|zero[-\s]?shot|few[-\s]?shot|"
                        r"evalplus|harness", re.I)
CHECKS = ["base_named", "before_reported", "after_reported",
          "recovery_printed", "recovery_checks", "harness_stated"]


def score_card(text):
    """-> dict of the six checks for one card."""
    rows = []
    try:
        rows = list(harvest.extract(text))
    except Exception:
        rows = []
    usable = [r for r in rows if r[1] is not None and r[2] is not None]
    arith = [r for r in usable if r[3] == "arith"]
    return {
        "base_named": bool(BASE_RE.search(text)),
        "before_reported": bool(usable),
        "after_reported": bool(usable),
        "recovery_printed": bool(REC_RE.search(text)),
        # the gate only returns "arith" when a printed recovery figure was
        # found AND agreed with the arithmetic, so this is the strict check
        "recovery_checks": bool(arith),
        "harness_stated": bool(HARNESS_RE.search(text)),
    }


def publisher_of(fname):
    return fname.split("_", 1)[0]


def main():
    cards = []
    for tag, d in DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            text = open(os.path.join(d, fn), encoding="utf-8",
                        errors="ignore").read()
            cards.append((publisher_of(fn), fn, score_card(text)))

    if not cards:
        raise SystemExit("no cached cards found; run src/publisher_census.py")

    by_pub = {}
    for pub, _, sc in cards:
        b = by_pub.setdefault(pub, {"n": 0, **{c: 0 for c in CHECKS},
                                    "score_sum": 0})
        b["n"] += 1
        got = sum(1 for c in CHECKS if sc[c])
        b["score_sum"] += got
        for c in CHECKS:
            b[c] += int(sc[c])

    result = {"standard": CHECKS, "n_cards": len(cards),
              "n_publishers": len(by_pub), "per_publisher": {}}
    for pub, b in by_pub.items():
        result["per_publisher"][pub] = {
            "cards": b["n"],
            "mean_score_of_6": round(b["score_sum"] / b["n"], 2),
            **{c: round(100.0 * b[c] / b["n"], 1) for c in CHECKS},
        }
    # Bias measurement: cards that plainly contain a benchmark table which the
    # parser nevertheless failed to read. Reported per publisher so the
    # parser-dependent columns can be discounted honestly.
    bench = re.compile(r"mmlu|arc.challenge|hellaswag|gsm8k|winogrande|"
                       r"truthfulqa|humaneval", re.I)
    for tag, d in DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".md"):
                continue
            t = open(os.path.join(d, fn), encoding="utf-8",
                     errors="ignore").read()
            looks_tabular = len([l for l in t.split("\n")
                                 if l.strip().startswith("|")
                                 and bench.search(l)]) >= 2
            if looks_tabular and not score_card(t)["before_reported"]:
                pub = publisher_of(fn)
                result["per_publisher"][pub]["parser_missed"] = \
                    result["per_publisher"][pub].get("parser_missed", 0) + 1
    for v in result["per_publisher"].values():
        v.setdefault("parser_missed", 0)
    result["parser_missed_total"] = sum(
        v["parser_missed"] for v in result["per_publisher"].values())
    result["note"] = ("No overall 'percentage checkable' is reported: checks "
                      "C2/C3/C5 use a parser written for one publisher's "
                      "layout and measurably under-read the others.")
    json.dump(result, open(OUT, "w"), indent=2)

    order = sorted(result["per_publisher"].items(),
                   key=lambda kv: -kv[1]["mean_score_of_6"])
    w = max(len(p) for p, _ in order)
    print(f"{'publisher':<{w}}  cards  score/6  " +
          "  ".join(c[:9] for c in CHECKS))
    for pub, v in order:
        print(f"{pub:<{w}}  {v['cards']:>5}  {v['mean_score_of_6']:>7}  " +
              "  ".join(f"{v[c]:>8.0f}%" for c in CHECKS))
    print("\nparser-dependent columns (before/after/recovery_checks) are biased:")
    for pub, v in order:
        print(f"  {pub:<{w}}  benchmark tables the parser failed to read: "
              f"{v['parser_missed']}")
    print("\nThe clean, parser-independent finding is C4: a recovery figure is")
    print("printed on " +
          ", ".join(f"{v['recovery_printed']:.0f}% of {pub} cards"
                    for pub, v in order) + ".")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
