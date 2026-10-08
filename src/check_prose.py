#!/usr/bin/env python3
"""Catch mangled prose joins in the paper variants.

WHY THIS FILE IS SHAPED THE WAY IT IS. An ad-hoc version of this scan failed
its own planted defects three times in one day: once because `re.I` defeated
the case-sensitivity the patterns depended on and it flagged all 249 normal
sentence boundaries, once because the dangling-conjunction pattern did not
match the defect it had been written for, and once because a plant was
miscalibrated against a pattern that had been removed. A scanner that needs
recalibrating every time it runs is not a check.

So DEFECTS is the primary artifact and PATTERNS is subordinate to it. The
plants are real defects that reached a paper variant, written out verbatim.
The contract is in two halves and both are enforced by
tests/test_all.py::test_prose_scanner_catches_its_own_plants:

    1. every entry in DEFECTS must be caught by some pattern
    2. every entry in CLEAN must be caught by none of them

Adding a pattern is only legitimate if the plants still hold. Adding a plant
is the normal way to extend this: write the defect down, watch it fail, then
write the pattern that catches it. Never the reverse -- a pattern invented
first is a pattern nobody has seen fire.

Usage:  python3 src/check_prose.py [file ...]        (default: both variants)
Exit:   0 clean, 1 if any candidate survives the false-positive filter.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DEFAULTS = [os.path.join(ROOT, "paper", "neurips_main.tex"),
            os.path.join(ROOT, "paper", "main.tex")]

# ---------------------------------------------------------------- the plants
# Real defects. Each one must be caught by at least one pattern.
DEFECTS = [
    # The contribution-2 mangling: a partial string replacement left a clause
    # opening with a conjunction after a semicolon and going nowhere.
    "flagged as one the tool cannot judge; or demoted and flagged.",
    # Doubled function words, the classic join artefact.
    "the the band is in-sample",
    "coverage of of the band",
    "and the width and the width come from",
    # A sentence truncated onto a preposition or article.
    "the measured value is 90 and. The rest follows",
    "we report the coverage of. The interval is",
    # The false-negative plant, 2026-10-08. Identical defect with an
    # unrelated decimal beside it: under the old 140-character exemption
    # window this went unreported, and whether it did depended on where the
    # line happened to wrap, so the same prose was flagged in one variant and
    # not the other. Keep both forms -- the pair is what proves the exemption
    # is tested against the match rather than its neighbourhood.
    "we report 2.583pp coverage of. The interval is",
    # Punctuation collisions from deleting a clause but not its separator.
    "a wider interval, , on the same rows",
    "the half-width is 2.68pp ,and the centre",
    # Emphasis emptied by a replacement that removed its argument.
    "widths come from the \\emph{} partition",
    "\\textbf{} is the shipped band",
    # Prose running straight into a heading because a terminator was lost.
    "that is the whole point \\paragraph{What the envelope cannot do.}",
]

# Text that must NOT be flagged. Every one of these is real prose from the
# paper that an over-broad pattern has flagged at some point.
CLEAN = [
    # Headings legitimately end in a preposition or article.
    "\\paragraph{What this tool is for.} The scenario it answers",
    "\\paragraph{A state we built, could have made fire, and chose not to.}",
    # A percent sign followed by a lowercase word is ordinary.
    "a 95\\% checkpoint-bootstrap CI of",
    "against a nominal 90\\% two-sided level, with a 95\\% cluster-robust CI",
    # Sentences ending on "that." are grammatical.
    "accuracy may still degrade uncertainty quality. We measure none of that.",
    "and we considered changing that. The paper no longer advertises",
    # Abbreviations and filenames contain periods mid-sentence.
    "see e.g. the discussion in",
    "recorded in PROVENANCE.md and in src/strata.py",
    "Tong et al. describe using a 90\\% coverage target",
    # Decimals.
    "the floor is 0.507pp against 0.262pp of headroom",
]

# ------------------------------------------------------------- the patterns
# Subordinate to the plants above. No re.I anywhere: every one of these
# depends on case, and re.I is what broke the first version.
# Subordinate to the plants above. No re.I anywhere: every one of these
# depends on case, and re.I is what broke the first version.
#
# Each entry carries two flags, because three plants proved a single uniform
# pipeline cannot catch them all:
#   strip_headings -- \paragraph{...} titles legitimately end in "for." or
#       "to.", so most patterns run with them removed. The pattern that looks
#       for prose running INTO a heading obviously cannot, so it opts out.
#   honour_exempt  -- the exemption list contains \d\.\d for decimals, and
#       it is matched against a 140-character window. That window blanketed a
#       real comma defect sitting near "2.68pp". Only patterns whose own match
#       contains a period need the decimal exemption, so the rest opt out.
PATTERNS = [
    (r"\b([a-z]{3,})\s+\1\b", "doubled word", True, False),
    # A repeated PHRASE of two to five words. The plant that forced this was
    # "and the width and the width come from", where the repeated unit is
    # three words -- a two-word pattern cannot see it.
    (r"\b((?:[a-z]{2,}\s+){1,4}[a-z]{2,})\s+\1\b",
     "doubled phrase", True, False),
    (r"\b(?:the the|of of|is is|and and|to to|in in)\b",
     "repeated function word", True, False),
    (r";\s*(?:and|or|but)\s+[^.;:]{0,60}\.",
     "clause opening with a conjunction after a semicolon", True, True),
    (r"\b(?:the|a|an|of|to|and|or|in|with|by|for)\s*\.\s+[A-Z]",
     "sentence truncated onto an article or preposition", True, True),
    (r",\s*,|,\s*\.|\.\s*,|\s,and\b", "punctuation collision", True, False),
    (r"\\emph\{\}|\\textbf\{\}|\\texttt\{\}", "emphasis emptied", True, False),
    (r"[a-z]{3,}\s+\\(?:paragraph|section|subsection)\{",
     "prose running into a heading", False, False),
    (r"\(\s*\)|\[\s*\]", "empty delimiter", True, False),
    (r"--- ---|-- --", "doubled dash", True, False),
]

# Contexts in which a match is not a defect.
_EXEMPT = re.compile(
    r"e\.g\.|i\.e\.|et al\.|cf\.|vs\.|\d\.\d|arXiv|"
    r"\.py\b|\.csv\b|\.json\b|\.tex\b|\.md\b|\.sty\b")


def hits(text):
    """-> [(why, matched_text, context)]."""
    body = "\n".join(l for l in text.split("\n")
                     if not l.lstrip().startswith("%"))
    raw = re.sub(r"\s+", " ", body)
    stripped = re.sub(r"\\paragraph\{[^{}]*\}", " ", body)
    stripped = re.sub(r"\\(?:section|subsection)\{[^{}]*\}", " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    out = []
    for pat, why, strip_headings, honour_exempt in PATTERNS:
        flat = stripped if strip_headings else raw
        for m in re.finditer(pat, flat):
            ctx = flat[max(0, m.start() - 70):m.end() + 70]
            # The exemption is tested against the MATCH plus a tight left
            # margin, not against `ctx`. Using the 140-character window was a
            # false negative: a genuine "coverage of. The interval" defect
            # went unreported when an unrelated decimal like 2.583pp happened
            # to fall inside the window, and whether it did depended on where
            # the line wrapped -- so the same prose was flagged in one
            # variant and not the other. The exemptions this guards against
            # (e.g., i.e., et al., Fig., decimals, filenames) all sit AT the
            # match or immediately before it, so a 14-character margin is
            # enough to see them and too tight to swallow a defect.
            probe = flat[max(0, m.start() - 14):m.end()]
            if honour_exempt and _EXEMPT.search(probe):
                continue
            out.append((why, m.group(0), ctx.strip()))
    return out


def self_check():
    """The contract. Returns a list of failures; empty means trustworthy."""
    bad = []
    for d in DEFECTS:
        if not hits(d):
            bad.append(f"PLANT NOT CAUGHT: {d!r}")
    for c in CLEAN:
        h = hits(c)
        if h:
            bad.append(f"FALSE POSITIVE on clean text {c!r}: "
                       f"{[x[0] for x in h]}")
    return bad


def main(argv):
    bad = self_check()
    if bad:
        print("the scanner does not satisfy its own plants:")
        for b in bad:
            print("  " + b)
        return 2
    print(f"self-check: {len(DEFECTS)} plants caught, "
          f"{len(CLEAN)} clean strings not flagged")
    paths = argv or DEFAULTS
    total = 0
    for p in paths:
        if not os.path.exists(p):
            continue
        h = hits(open(p, encoding="utf-8").read())
        total += len(h)
        print(f"\n{os.path.relpath(p, ROOT)}: {len(h)} candidate(s)")
        for why, got, ctx in h:
            print(f"  [{why}] {got!r}\n     ...{ctx}...")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
