#!/usr/bin/env python3
"""Pin the committed artifacts behind the 29 keys with unpinned card inputs.

data/CARD_MANIFEST.sha256 covers 178 of the 584 card files. The remaining
406 are not committed, so a clone cannot verify them -- but it can verify
the artifact each of those card sets was reduced to, and that artifact is
what src/verify_claims.py actually reads. This writes those hashes.

    python tools/pin_derived_inputs.py          # check, exit 1 on drift
    python tools/pin_derived_inputs.py --write  # regenerate the manifest

Exits 3 when a pinned artifact is absent: an artifact that is not there has
not been verified, and that is neither a pass nor a drift.
"""
from __future__ import annotations
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MAN = os.path.join(ROOT, "data", "DERIVED_INPUTS.sha256")

FILES = [
    ("out/publisher_census.json",
     "14 pub_* keys; the publisher census, built from the 260 uncommitted "
     "out/publisher_cards/ files"),
    ("out/track2_lowrank.json",
     "11 lk_* keys; the kurtosis/LoRA analysis, built from the 136 "
     "uncommitted out/lora_cards/ files"),
    ("data/qwen35/published_rows.csv",
     "qwen35_pub_rows / qwen35_pub_verified; built from the 10 uncommitted "
     "out/qwen35_cards/ files"),
    ("data/qwen35/published_rejects.csv",
     "qwen35_pub_rejects / qwen35_pub_repos; same source"),
]

HEADER = """# Hashes for the committed artifacts that the 29 registry keys with
# UNPINNED upstream card inputs actually read.
#
# data/CARD_MANIFEST.sha256 pins 178 of the 584 files in the card-corpus
# snapshot. The other 406 -- out/publisher_cards (260), out/lora_cards (136),
# out/qwen35_cards (10) -- are not committed and a clone cannot hash them, so
# this manifest does not pretend to. What a clone CAN verify is the artifact
# each of those card sets was reduced to, which is what the registry reads.
# Pinning those closes the distance between a published figure and the
# nearest input a reader actually has, and leaves exactly one unverifiable
# hop -- cards to artifact -- named rather than implied.
#
# Regenerate with: python tools/pin_derived_inputs.py --write
"""


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main() -> int:
    write = "--write" in sys.argv
    if write:
        out = [HEADER]
        for rel, why in FILES:
            p = os.path.join(ROOT, rel)
            out.append(f"# {why}")
            out.append(f"{_sha(p)}  {rel}" if os.path.isfile(p)
                       else f"# MISSING  {rel}")
        open(MAN, "w").write("\n".join(out) + "\n")
        print(f"wrote {MAN} ({len(FILES)} artifacts)")
        return 0

    if not os.path.isfile(MAN):
        print(f"UNCHECKED: {MAN} is absent; nothing to check against.")
        return 3
    want = {}
    for line in open(MAN, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        h, _, rel = line.partition("  ")
        want[rel.strip()] = h
    drift, absent, checked = [], [], 0
    for rel in sorted(want):
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            absent.append(rel)
            continue
        checked += 1
        got = _sha(p)
        if got != want[rel]:
            drift.append(f"{rel}: pinned {want[rel][:12]}, "
                         f"on disk {got[:12]}")
    print(f"comparisons made: {checked} of {len(want)} pinned artifacts")
    for rel in absent:
        print(f"  UNCHECKED  {rel} is absent")
    for d in drift:
        print(f"  DRIFT      {d}")
    if drift:
        print("FAIL: a derived input moved. Every key it feeds is now a "
              "different figure; re-run src/verify_claims.py and "
              "paper/gen_numbers.py, and re-pin with --write in the same "
              "change.")
        return 1
    if checked == 0:
        print("UNCHECKED: no pinned artifact was present, so nothing was "
              "verified.")
        return 3
    if absent:
        return 3
    print("OK: every derived input matches its pin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
