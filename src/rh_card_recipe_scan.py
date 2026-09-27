"""Scan the cached RedHatAI card READMEs and count which recipe fields
each card actually documents.

Two audits, deliberately separate:
  * A LIVE scan across every RedHatAI quantized-model card published on
    Hugging Face at the time of writing (2026-09-26). Cached under
    data/rh_card_scan_2026_09_26/.
  * The HARVESTED subset: the 102 of those cards that actually contribute at
    least one row to the final dataset (data/dataset.csv). This is the
    subsample that our calibration and validation actually depend on.

The paper's Data paragraph on undocumented pathways uses these counts. Every
number this script emits is date-suffixed in the claims registry, so that a
future re-scan produces a NEW key rather than silently overwriting the number
the paper committed to on this date.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SCAN_DATE = "2026_09_26"
CARDS = os.path.join(ROOT, "data", f"rh_card_scan_{SCAN_DATE}")
DATASET = os.path.join(ROOT, "data", "dataset.csv")
OUT = os.path.join(ROOT, "out", f"rh_card_scan_{SCAN_DATE}.json")

# Detection regexes. Each is conservative: we require an explicit signal on
# the card, not a vague hint. False negatives are preferred over false
# positives so the paper's numbers are lower bounds on what is documented.

LIB_VERSION = re.compile(
    r"(llm[-_ ]?compressor|compressed[-_ ]?tensors|transformers|"
    r"gptqmodel|autogptq|optimum)"
    r"[\s]*[=<>~^]{1,2}\s*[\"']?\d+\.\d+", re.I)
LIB_NAMED = re.compile(
    r"\b(llm[-_ ]?compressor|autogptq|gptqmodel|neuralmagic)\b", re.I)
CALIB_DATASET = re.compile(
    r"calibrat(?:ion|ed).{0,80}(?:ultrachat|c4|wikitext|pile|redpajama|"
    r"openplatypus|lima|dolly|self-instruct|open_platypus|magpie)|"
    r"ultrachat[_-]?200k|neuralmagic/LLM_compression_calibration|"
    r"neural-magic/calibration", re.I)
NUM_CALIB = re.compile(
    r"num[_ ]?calibration[_ ]?samples|\d{2,4}\s*(?:calibration )?samples|"
    r"num_samples", re.I)
MAX_SEQ = re.compile(
    r"max[_ ]?seq(?:uence)?[_ ]?length|max_sequence_length|"
    r"sequence length of", re.I)
ACT_ORDER = re.compile(r"act[_-]?order|actorder|activation reordering", re.I)
DAMP = re.compile(r"damp(?:ening|_frac|_percent)?\s*[:=]", re.I)
GROUP_SIZE = re.compile(
    r"group[_ ]?size\s*[:=]?\s*(?:of\s*)?\d+|group of \d+ parameters", re.I)


def flags(txt):
    return {
        "lib_named": bool(LIB_NAMED.search(txt)),
        "lib_version": bool(LIB_VERSION.search(txt)),
        "calib_dataset": bool(CALIB_DATASET.search(txt)),
        "num_calib": bool(NUM_CALIB.search(txt)),
        "max_seq": bool(MAX_SEQ.search(txt)),
        "act_order": bool(ACT_ORDER.search(txt)),
        "damp": bool(DAMP.search(txt)),
        "group_size": bool(GROUP_SIZE.search(txt)),
    }


def card_id(path):
    """cache filenames encode HuggingFace IDs with '/' rewritten to '_'."""
    base = os.path.basename(path)[:-len(".md")]
    return base.replace("_", "/", 1)


def scan():
    files = sorted(glob.glob(os.path.join(CARDS, "*.md")))
    # Only cards whose README actually fetched (small stubs are 404 fallbacks).
    non_empty = [p for p in files if os.path.getsize(p) > 200]
    records = []
    for p in non_empty:
        with open(p, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        records.append({"id": card_id(p), **flags(txt)})

    with open(DATASET) as f:
        harvested_ids = {r["model"] for r in csv.DictReader(f)
                         if r["model"].startswith("RedHatAI/")}
    harvested_records = [r for r in records if r["id"] in harvested_ids]

    def counts(rs):
        n = len(rs)
        return {
            "n": n,
            "lib_named": sum(r["lib_named"] for r in rs),
            "lib_version": sum(r["lib_version"] for r in rs),
            "calib_dataset": sum(r["calib_dataset"] for r in rs),
            "num_calib": sum(r["num_calib"] for r in rs),
            "max_seq": sum(r["max_seq"] for r in rs),
            "act_order": sum(r["act_order"] for r in rs),
            "damp": sum(r["damp"] for r in rs),
            "group_size": sum(r["group_size"] for r in rs),
            "no_recipe": sum(1 for r in rs if not any(
                (r["num_calib"], r["max_seq"], r["act_order"], r["damp"]))),
            "no_anything": sum(1 for r in rs if not any(
                (r["lib_version"], r["calib_dataset"], r["num_calib"],
                 r["max_seq"], r["act_order"], r["damp"], r["group_size"]))),
        }

    return {
        "scan_date": SCAN_DATE,
        "cards_scanned": counts(records),
        "harvested_subset": counts(harvested_records),
    }


def main():
    result = scan()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    a = result["cards_scanned"]
    h = result["harvested_subset"]
    print(f"scanned  {a['n']} cards, pins version {a['lib_version']}, "
          f"names dataset {a['calib_dataset']}, "
          f"no recipe {a['no_recipe']}")
    print(f"harvested {h['n']} of those, pins version {h['lib_version']}, "
          f"no recipe {h['no_recipe']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
