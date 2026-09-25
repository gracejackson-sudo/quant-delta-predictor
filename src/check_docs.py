"""
Cross-document consistency check.

verify_claims.py proves each number matches a computation. It does NOT prove
the documents agree with each other, or that their cross-references resolve.
This closes that gap.
"""
from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
DOCS = [f for f in sorted(os.listdir(ROOT)) if f.endswith(".md")]
GENERATED = {"RANKING.md", "NEGATIVE_RESULT.md", "BIAS_CORRECTION.md",
             "TOOL_SUMMARY.md", "ONE_SIDED_COVERAGE.md", "EXTERNAL_FEEDBACK.md"}

# statements that must not reappear anywhere user-facing
BANNED = [
    (r"safe to adopt", "Tier A wording retired after the audit"),
    (r"ignores your model entirely", "false since size-widening landed"),
    (r"\b12 numbers\b", "artifact is 18 numbers"),
    (r"26 of 29", "stale tail statistic"),
    (r"80%, not 90%", "stale sub-2B coverage claim"),
    (r"collapse(s)? to (predicting )?(~|approximately )?0",
     "Track 2 refuted this: low-rank does not collapse to zero"),
]


def read(f):
    return open(os.path.join(ROOT, f), encoding="utf-8").read()


def strip_tags(s):
    return re.sub(r"<!--.*?-->", "", s, flags=re.S)


def main():
    problems, warnings = [], []
    texts = {f: strip_tags(read(f)) for f in DOCS}

    # ---- 1. cross-reference resolution
    print("=" * 70)
    print("1. CROSS-REFERENCES")
    print("=" * 70)
    md_link = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    bare = re.compile(r"`?\b([A-Z_]{3,}\.md)\b`?")
    for f, t in texts.items():
        targets = set()
        for _, href in md_link.findall(t):
            if href.startswith(("http://", "https://", "#")):
                continue
            targets.add(href.split("#")[0])
        targets |= set(bare.findall(t))
        for tgt in sorted(targets):
            if not tgt.endswith(".md"):
                continue
            if not os.path.exists(os.path.join(ROOT, tgt)):
                problems.append(f"{f}: references missing file {tgt}")
    print(f"   checked {len(texts)} docs; "
          f"{len([p for p in problems if 'missing file' in p])} broken refs")

    # ---- 2. banned / retracted statements
    print("\n" + "=" * 70)
    print("2. RETRACTED STATEMENTS")
    print("=" * 70)
    for f, t in texts.items():
        low = t.lower()
        for pat, why in BANNED:
            for m in re.finditer(pat, low):
                seg = t[max(0, m.start() - 70):m.start() + 70].replace("\n", " ")
                # audit docs are allowed to quote what they retracted
                if f in ("ADVERSARIAL_AUDIT.md", "AUDIT_DISCIPLINE.md",
                         "PROVENANCE.md", "FINDINGS.md"):
                    continue
                # a doc is allowed to QUOTE the banned list when documenting
                # the rule itself; detect the quoted-list context
                ctx = t[max(0, m.start() - 200):m.start() + 200]
                if re.search(r"banned|blocks phrases|retracted|reappearing",
                             ctx, re.I):
                    continue
                problems.append(f"{f}: retracted phrasing ({why}) near "
                                f"...{seg.strip()}...")
    print(f"   {len([p for p in problems if 'retracted' in p])} occurrences "
          f"outside the audit/history docs")

    # ---- 3. Track 2 conclusion stated consistently
    print("\n" + "=" * 70)
    print("3. TRACK 2 CONSISTENCY (low-rank result)")
    print("=" * 70)
    mentions = {f: t for f, t in texts.items()
                if re.search(r"low[- ]rank|BenchPress", t, re.I)}
    print(f"   docs mentioning BenchPress / low-rank: {sorted(mentions)}")
    for f, t in mentions.items():
        if re.search(r"low[- ]rank", t, re.I) and \
           not re.search(r"8\.7|order of magnitude|worse than|artefact|"
                         r"\d+\.\d+pp.{0,80}\d+\.\d+pp", t, re.I) and \
           f not in ("README.md", "TOOL_SUMMARY.md", "SCOPE.md"):
            warnings.append(f"{f}: mentions low-rank without stating the "
                            f"measured outcome")

    # ---- 4. does every doc that cites BenchPress give the arXiv id?
    for f, t in mentions.items():
        if "BenchPress" in t and "2606.24020" not in t:
            warnings.append(f"{f}: cites BenchPress without the arXiv id")

    # ---- 5. generated docs must carry the do-not-edit banner
    print("\n" + "=" * 70)
    print("4. GENERATED-DOC PROVENANCE")
    print("=" * 70)
    for f in sorted(GENERATED):
        if f not in texts:
            problems.append(f"{f}: generated doc missing from repo")
            continue
        if not re.search(r"generated|regenerate", texts[f], re.I):
            warnings.append(f"{f}: generated but does not say so")
        print(f"   {f:<24} says it is generated: "
              f"{bool(re.search(r'generated|regenerate', texts[f], re.I))}")

    # ---- report
    print("\n" + "=" * 70)
    print(f"RESULT: {len(problems)} problems, {len(warnings)} warnings")
    print("=" * 70)
    for p in problems:
        print(f"  PROBLEM  {p}")
    for w in warnings:
        print(f"  WARNING  {w}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
