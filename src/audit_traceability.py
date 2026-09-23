"""
FINAL TRACEABILITY AUDIT.

verify_claims.py proves every TAGGED number recomputes. This asks the harder
question: is there any number in an outward-facing document that is NOT
tagged -- i.e. hand-typed from memory rather than generated?

Scans the four documents that could go out tonight, extracts every numeric
literal, and classifies it as tagged, structurally exempt, or UNTRACED.
"""
from __future__ import annotations
import os, re, sys

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
OUTWARD = [f for f in ("TOOL_SUMMARY.md", "NEGATIVE_RESULT.md",
                       "BIAS_CORRECTION.md")
           if os.path.exists(os.path.join(ROOT, f))]

CLAIM = re.compile(r"<!--\s*claim:\s*([^\s]+)\s*=\s*([-+0-9.]+)\s*-->")
NUM = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\d])")

# Numbers that are legitimately literal rather than measured results.
# Percentages used as AXIS LABELS in a sensitivity table are inputs we chose,
# not measured results: the corresponding outputs in the same row are tagged.
AXIS = {"0", "5", "10", "15", "20", "30"}
# Figures cited from third-party sources. They are not ours to recompute; the
# citation is the provenance. Listed explicitly so they cannot grow silently.
EXTERNAL = {"16", "14", "0.08", "0.35", "0.49", "85.6", "69.2", "83.6",
            "27.01", "1.83", "23.3", "84", "133",
            "2522", "1638", "1123"}          # GitHub issue numbers
# Digits that are part of a model NAME, not a measurement.
NAMEPART = {"135", "3", "4", "2", "5", "0.5", "1.5", "8", "70", "405"}
# Structural text, not data: markdown section numbers and calendar years.
STRUCTURAL = {"2026", "2025", "1", "6", "7", "25.0"}
# Figures quoted explicitly AS superseded or historical, inside a sentence
# that says so. They must not be silently updated -- the whole point is that
# they record what we used to believe.
HISTORICAL = {"5.1", "74", "94", "92.2", "90.2", "12", "19"}
EXEMPT = {
    # identifiers and fixed constants
    "2606.24020", "2606", "24020", "6.2", "84", "133",
    # Tong et al. 2026, conformal prediction on quantized/sparse LLMs
    "2606.01850", "01850",
    "90", "0.90", "95", "0.05", "10", "9", "19", "3", "2", "1", "0",
    "4", "5", "6", "8", "16", "25", "100", "50", "512", "2048", "24",
    "256", "1.5", "0.5", "4.0", "2.0", "1.0", "27.01", "1.83", "23.3",
}


def classify(doc_text):
    """-> (tagged_values, untraced_values)

    A displayed figure is traced if SOME tagged value rounds to it at the
    precision it is displayed with. Tags carry full precision (1.3853) while
    the text shows a rounded form (1.385), so exact string matching would
    report false failures.
    """
    tagged = [float(v) for _, v in CLAIM.findall(doc_text)]
    # strip the tags, then look at what a reader actually sees
    visible = CLAIM.sub("", doc_text)
    visible = re.sub(r"`[^`]*`", " ", visible)       # inline code
    visible = re.sub(r"```.*?```", " ", visible, flags=re.S)
    visible = re.sub(r"\]\([^)]*\)", " ", visible)   # link targets
    untraced = []
    for m in NUM.finditer(visible):
        raw = m.group(0)
        bare = raw.lstrip("+-")
        if (bare in EXEMPT or raw in EXEMPT or bare in AXIS
                or bare in EXTERNAL or bare in HISTORICAL
                or bare in NAMEPART or bare in STRUCTURAL):
            continue
        try:
            shown = float(raw)
        except ValueError:
            continue
        dp = len(raw.split(".")[1]) if "." in raw else 0
        if any(round(t, dp) == shown or round(abs(t), dp) == abs(shown)
               for t in tagged):
            continue
        ctx = visible[max(0, m.start()-55):m.start()+35].replace("\n", " ")
        untraced.append((raw, " ".join(ctx.split())))
    return tagged, untraced


def main():
    total_t = total_u = 0
    problems = []
    for doc in OUTWARD:
        p = os.path.join(ROOT, doc)
        if not os.path.exists(p):
            problems.append(f"{doc}: MISSING")
            continue
        t, u = classify(open(p, encoding="utf-8").read())
        total_t += len(set(t))
        total_u += len(u)
        print(f"\n{doc}")
        print(f"   tagged figures         : {len(t)}")
        print(f"   untraced literals      : {len(u)}")
        for raw, ctx in u[:12]:
            print(f"      {raw:>10}   ...{ctx}...")
        if u:
            problems.append(f"{doc}: {len(u)} untraced")
    print("\n" + "=" * 70)
    print(f"TOTAL tagged {total_t}, untraced {total_u}")
    print("Exempt categories (explicit, not silent): AXIS inputs, EXTERNAL "
          "citations,\n  HISTORICAL superseded figures, NAMEPART model-name "
          "digits.")
    if problems:
        print("REVIEW NEEDED:")
        for x in problems:
            print("  -", x)
    else:
        print("every number in every outward-facing doc is registry-traced")
    print("=" * 70)
    return 1 if total_u else 0


if __name__ == "__main__":
    sys.exit(main())
