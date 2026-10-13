"""Audit the paper. Fails loudly; prints every check.

Four checks, matching the standard applied to the rest of the project:
  1. CLAIM TRACING        every number in main.tex is a generated macro
  2. CITATIONS            every \\cite key is defined, every arXiv id resolves
  3. OVERCLAIM SCAN       banned absolutes and retracted phrasings
  4. INDEPENDENT REDERIVE 3+ headline numbers recomputed from raw source,
                          not from the registry the paper was built from
"""
from __future__ import annotations
import json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
TEXNAME = sys.argv[1] if len(sys.argv) > 1 else "main.tex"
TEX = open(os.path.join(HERE, TEXNAME)).read()
NUMS = open(os.path.join(HERE, "numbers.tex")).read()
BIB = open(os.path.join(HERE, "refs.bib")).read()
# `unchecked` is the third state: a check whose INPUT or SERVICE was not
# available, which is not a defect in the paper. Added 2026-10-12 after an
# arXiv rate-limit turned the variants test red on a clean tree.
fail, warn, unchecked = [], [], []


def head(t): print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


# ---------------------------------------------------------------- 1
head(f"1. CLAIM TRACING  [{TEXNAME}]")
defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", NUMS))
used = set(re.findall(r"\\([A-Z]\w+)", TEX))
cand = {m for m in used if m not in defined}
# LaTeX built-ins / environment names we expect to see capitalised
BUILTIN = {"Acc", "Math", "T", "L", "R", "C", "P", "S", "N", "MATH"}
undef = sorted(m for m in cand if m not in BUILTIN and len(m) > 2
               and not m.isupper())
if undef:
    fail.append(f"macros used but not defined: {undef}")
print(f"   macros defined in numbers.tex : {len(defined)}")
print(f"   macros referenced in main.tex : {len(used & defined)}")
print(f"   referenced-but-undefined      : {len(undef)} {undef if undef else ''}")

# hand-typed numbers in prose (exempt: section refs, years, structural).
# The reduction and the list of what it removed both come from
# paper/texbody.py, which tests/test_all.py imports as well -- so the
# span-rule test reads THIS reduction rather than a copy of its
# patterns. It used to hold copies, and planting the escaped-percent
# bug here left that test green.
from texbody import strip_tex  # noqa: E402
body, BODY_SPANS = strip_tex(TEX)
NUM = re.compile(r"(?<![\w.\\])\d+(?:\.\d+)?(?![\d])")
EXEMPT = {"1", "2", "3", "4", "5", "6", "8", "11", "16", "33", "38", "43",
          "68", "84", "90", "95", "100", "133", "500", "2022", "2023", "2024",
          "2026", "0", "36", "0.03", "0.36", "89.6", "93.4", "99", "10",
          "500,000", "118", "131", "83.6", "94.6", "90.1", "68", "9",
          # 119: the strict count immediately after the 2026-09-27 tolerance
          # patch, quoted in the float-boundary bullet beside 118/131 so a
          # reader can see what that patch did and did not change. Same
          # historical-narrative class as 118 and 131 above.
          "119",
          # 6.4: the retracted stratification claim quoted verbatim in the
          # audit section ("costs 6.4 points of coverage and buys no
          # tightening"). The figure measured full stratification while the
          # sentence was about the widen-only rule; quoting it is the point
          # of the entry, and the live figures beside it are macros.
          "6.4",
          # documented exemptions, each checked by hand:
          "000",   # part of "500{,}000", an external figure from Kurtic et al.
          "3.1",   # part of the model name "Llama-3.1", not a measurement
          "3.2",   # part of the model name "Llama-3.2"
          "3.3",   # part of the model name "Llama-3.3"
          "2.5",   # part of the model name "Qwen-2.5"
          "7",     # "7 test rows" -- a historical figure in the audit record
          "1.7",   # "up to 1.7B parameters", external: Kumar et al. abstract (arXiv:2411.04330)
          "0.5",   # part of the model name "Qwen2.5-0.5B-Instruct"
          "1.5",   # part of the model name "Qwen2.5-1.5B-Instruct"
          "09", "26",  # month/day in the RedHatAI-card scan date 2026-09-26
          "27",  # day in the 2026-09-27 A4 fix / independent-verification run
                 # date (PROVENANCE.md; §5 "Independent verification --- what
                 # and when" paragraph). A date, not a metric.
          "128",   # group size documented on RedHatAI/Qwen2.5-1.5B-quantized.w4a16
          "64",    # group size documented on RedHatAI/Qwen2.5-0.5B-quantized.w4a16
          "135",   # SmolLM-135M-Instruct model name digit
          "70",    # 70B model-name digit (Llama-3.1-70B, Llama-3.3-70B)
          "90",    # 90B model-name digit (Llama-3.2-90B-Vision)
          # Tier 2.5 v2 §9 bullet describes the GPQA baseline ranges
          # (~25-32% chance-baseline vs ~46-67% headline) and lists the
          # affected Qwen3 variants (8B, 14B, 32B). These are qualitative
          # ranges and model-name digits, not measured claim-macros.
          "25", "32", "46", "67", "14",
          # §9 "artifact family regenerated" bullet and the rewritten
          # "regression we nearly shipped" paragraph document before/
          # after values for the day-7 regeneration. Each is present
          # in a narrative role, paired with the current live macro,
          # and each is enumerated in PROVENANCE.md's 2026-09-30 entry.
          # Not tagged because they are historical, not "the current value
          # to display" -- the current value IS the macro next to them.
          # alpha itself, and the conformal index requirement AT alpha=0.05:
          # a finite two-sided 95% quantile needs 19 units, the same kind of
          # constant as the 9 already exempt above at alpha=0.10. Neither is
          # a corpus measurement; the count of schemes that HAVE 19 units is
          # a measurement and is \JackknifeSchemesNineteen.
          "0.10", "19",
          # Hidden until the escaped-percent bug was fixed on 2026-10-11:
          # 0.9 is 1-alpha inside the conformal index formula, and 168 /
          # 90.01 / 91.0 are day-7 before-and-after values in the same
          # appendix item as 89.15, 60 and 93.4, which were already exempt.
          # Each sat after a printed percentage on its line, so no check
          # had ever read it.
          # The escaped-percent entry quotes the defect as measured at
          # 708a84d, the last commit with it live: 132 escaped percent
          # signs, 3{,}162 hidden characters, 97.0% of that revision
          # scanned. Historical and deliberately frozen -- recomputing them
          # against the current paper would answer a different question,
          # which is the error the entry itself is about. 7 and 3.0 are
          # already exempt above. See PROVENANCE.md 2026-10-11 (fifth).
          # The escaped-percent entry quotes the defect at 708a84d,
          # measured by executing that revision's own stripper:
          # 132 escaped percent signs, 7 real comments, 3{,}162 of
          # 103{,}841 characters hidden in the anonymous variant and
          # 3{,}162 of 103{,}710 in the named one, 3.05% hidden and
          # 96.96% scanned, over 18 days and 103 commits. Historical and
          # deliberately frozen; tools/measure_percent_defect.py
          # reproduces them from history.
          "132", "162", "841", "710", "103", "18", "3.05", "96.96",
          "98", "91", "253",   # 98 runs removed: 7 comments + 91 of prose,
                               # holding 3{,}253 characters. Same frozen
                               # historical measurement as the figures
                               # above; tools/measure_percent_defect.py
                               # reproduces all of them.
          # 91.0 was exempt here and is now \BandEmpCov: the sentence names
          # the key (band_emp_coverage_pct) and the literal equalled the
          # live value, so exempting it was the wrong disposition. The other
          # three are historical before/after values in the regeneration
          # item, in the same class as 89.15, 60 and 93.4 beside them.
          "0.9", "168", "90.01",
          "0.1",       # +/- 0.1 rounding in "roughly 40%" prose approx
          "0.7545",    # pre-regen mae_global_lofo
          "0.7553",    # post-regen mae_global_lofo (approx; live macro)
          "0.764",     # pre-regen pred_mae::ridge (approx)
          "0.776",     # post-regen pred_mae::ridge (approx; live macro)
          "3.8",       # pre-regen band gap (empirical-conformal ~3.8pp)
          "60",        # Regime C fold count under 6-family
          "89.15",     # pre-regen Regime C mondrian coverage
          "40",        # "roughly 40% narrower" qualitative (already exempt via string)
          # C3 fresh (day-7 audit) §9 gpqa bullet quotes the two acc_before
          # values on the disputed rows: 31.63 (Llama-3.3 NVFP4 "GPQA (0-shot)")
          # and 46.10 (Llama-3.3 FP8-dynamic and w8a8 "GPQA (0-shot)").
          # They are the concrete numbers the ambiguity note points at.
          "31.63", "46.10",
          "40",    # 40 GB VRAM (A100-SXM4-40GB, on which the day-6 rerun OOM'd)
          "80",    # 80 GB VRAM (the hardware that would make the 3B extension fit)
          "0.85",  # <=0.85pp stochastic GPTQ deviation observed in the day-6 regression
          "0.4",   # llmcompressor 0.4.1 in the pinned-retry disclosure
          "4.44",  # transformers 4.44.2 in the pinned-retry disclosure
          # Float-boundary example in the audit note on shared-assumption
          # reimplementation. These are exact float values from the bug and
          # documenting them precisely is the point of the paragraph.
          "1.40", "1.3999999999999997",
          # Ridge and HistGradientBoosting hyperparameters, documented for
          # reproducibility of the negative-result predictors (see
          # src/predictor.py:63 and src/diagnose.py:61). Fixed code constants,
          # not measured figures.
          "3.0", "300", "0.05", "15", "1.0",
          # Tier-0.1 audit bullet in §9 documents the family-definition
          # change and its LOFO effect. The four literals are quoted
          # once, each in a specific narrative role and each traceable to
          # a value shown per-family in a supplement table; they are not
          # generalisable figures, so tagging them would be spurious.
          "87.3",  # aggregate LOFO across 8 folds, before Def B
          "88.2",  # aggregate LOFO across 6 folds, after Def B
          "85.1",  # llama-3.3 held-out mean under 8 folds (weakness now
                   # folded into llama-3, disclosed)
          "92.2",  # llama-3 held-out mean under 6 folds (the fold the
                   # weakness folded into)
          }
hand = sorted({n for n in NUM.findall(body) if n not in EXEMPT})
if hand:
    warn.append(f"hand-typed numerals in prose (verify each): {hand}")
print(f"   hand-typed numerals in prose  : {len(hand)} {hand if hand else ''}")

# ------------------------------------------------- figures written as WORDS
# The numeral check above cannot see these: a spelled-out count is not a
# numeral. Two live claims had been quoting registry values as words since
# the registry existed -- "over twenty seeds" against calib_partitions=20,
# and "a thirteen-name list" against calib_cal_ckpt=13 -- and passed every
# gate in this repository. One of them, "two of the old widenings disappear",
# was also WRONG (the set difference is four; two is the net change), which
# is what an untraced figure buys you.
#
# Matching every word-number flags about 300 sites in this paper and almost
# all of them are rhetorical: "one of two grounds", "both halves", "half the
# corpus", "one level up". An allowlist of that size is the gate nobody
# trusts. So the pattern matches the SHAPE OF A COUNT instead -- a word
# number immediately governing one of the things this project counts, with up
# to two modifiers in between ("twenty-eight calibration rows"). That is 63
# sites, which is reviewable, and it caught every real instance found by hand.
_WUNIT = ("zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
          "twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
          "nineteen|twenty")
_WTENS = "thirty|forty|fifty|sixty|seventy|eighty|ninety"
_WORD = (rf"(?:{_WTENS})-(?:one|two|three|four|five|six|seven|eight|nine)"
         rf"|(?:{_WTENS})|(?:{_WUNIT})|a dozen|a handful|a couple")
# What this project counts. The first version of this list was chosen by
# hand, which made the GATE'S COVERAGE BOUNDARY a guess: a figure counting a
# noun nobody thought of was invisible again. So the list is now checked
# against the paper's own numeral usage -- the nouns this paper counts with
# numerals are the nouns it may count with words -- by
# _derive_counted_nouns() below, which fails the build when the paper starts
# counting something the vocabulary does not cover.
#
# "numbers", "losses", "bases", "plots", "estimates" and "misses" were added
# by that derivation on 2026-10-11; none had been thought of. "misses" is the
# one that matters: "two misses against two" was blocker 2's supporting
# asymmetry.
_WNOUN = ("cells?|rows?|checkpoints?|schemes?|famil(?:y|ies)|seeds?|"
          "partitions?|folds?|cards?|repos?|repositories|benchmarks?|"
          "publishers?|arms?|paths?|keys?|macros?|models?|files?|names?|"
          "units?|evaluations?|figures?|claims?|gates?|tests?|entries|entry|"
          "instances?|sites?|variants?|scripts?|columns?|protocols?|"
          "questions?|verdicts?|states?|thresholds?|points?|parameters?|"
          "pairs?|bands?|intervals?|predictors?|grounds?|numbers?|losses|"
          "bases|plots?|estimates?|misses|hits|reviewers?|commits?|"
          "observations?|characters?|days?|commits?|runs?|comments?")
WORDNUM = re.compile(
    rf"\b(?:{_WORD})(?:[ -](?:[a-z]+[ -]){{0,2}}(?:{_WNOUN}))\b", re.I)
# Hyphenated METHOD NAMES and compound adjectives, not counts.
WORDTERM = re.compile(r"^(?:leave-)?(?:one|two|three|four|six|eight|twenty)-"
                      r"(?:family|checkpoint|sided|state|column|fold|way|"
                      r"directional|fifths|out|base)", re.I)
# Rhetorical or self-evident uses, each checked by hand during the
# 2026-10-11 sweep. ADDING TO THIS SET REQUIRES A REASON: the whole point is
# that a figure cannot be hidden here by spelling it out.
WORD_EXEMPT = {
    # --- enumerated right there in the sentence, so the count is its own
    #     evidence and a macro would add nothing
    "four parameters", "seven files", "three cards", "two grounds",
    "one of two grounds", "two operative verdicts", "two verdicts",
    "three verdicts", "two states", "two size bands",
    # --- definitional, not measured: these sentences DEFINE the object
    "two checkpoints",          # "two checkpoints share a family when..."
    "one model family", "one question", "one checkpoint", "one row",
    "one scheme", "one family", "one gate", "one publisher", "one card",
    "one benchmark", "one figure", "two arms row", "two models",
    "two cards", "zero rows", "zero delta or verdict",
    # --- historical figures in the error record, quoted as what we then
    #     believed. The live values beside them are macros.
    "three separate families", "three separate scripts", "two cells",
    "two specific cells", "two rows", "three scripts", "four arms",
    "three paths", "eight band-constructing paths", "two further instances",
    "four rows", "three schemes",
    # --- mathematical constants of the construction, not corpus figures:
    #     the 90% conformal index needs n >= 9 whatever the corpus is
    "nine rows",
    # --- "one number" as the thing a reader wants, not a count of numbers
    "one number",
    # --- rhetorical singulars in the remediation entry: "converting one
    #     site of a figure", "the one instance in this record"
    "one site", "one instance",
    # A verified interval between two named commits: the key was registered
    # and its first site converted on 2026-10-08 (0d95501), the second site
    # on 2026-10-11 (59f4cd9). An earlier draft said "five days", which the
    # word gate surfaced once "days" entered the counted-noun vocabulary.
    "three days",
    # "one quantization run" is definitional -- it says what a checkpoint
    # IS -- and "let the second one run" uses run as a verb. Surfaced when
    # "runs" entered the vocabulary for the 98-runs reconciliation.
    "one quantization run", "one run",
    # "the one cell whose upper bound does sit below it (w4a16|<2B...)" --
    # definite article, and the cell is named in the same parenthesis
    "one cell",
    # --- deliberately vague quantifiers
    "a handful of rows",
}
wordhits = []
for m in WORDNUM.finditer(body):
    g = m.group(0).strip()
    if WORDTERM.match(g) or " ".join(g.lower().split()) in WORD_EXEMPT:
        continue
    seg = re.sub(r"\s+", " ", body[max(0, m.start()-60):m.start()+70])
    wordhits.append(f"{g!r} ...{seg}...")
if wordhits:
    fail.append("figures written as words (convert to a macro, or add to "
                "WORD_EXEMPT with a reason):\n      "
                + "\n      ".join(wordhits))
print(f"   figures written as words      : {len(wordhits)}")

# ------------------------------------- is the counted-noun list wide enough?
# _WNOUN above is the word gate's COVERAGE BOUNDARY, and the first version of
# it was chosen by hand -- so a figure counting a noun nobody thought of was
# invisible all over again, which is the same failure as "we fixed the paths
# we knew of". This derives the boundary from the paper's own numeral usage
# instead: whatever the paper counts with numerals, it may count with words,
# so every noun that follows a numeral must be in _WNOUN or in NOT_COUNTED
# with a reason. Self-maintaining, and the diff is reviewable.
_DEFINED_MACROS = {m for m in defined if m.isalpha()}
# Abandon the walk on these: the numeral was not counting a noun. Function
# words, finite verbs, bare adjectives and the word-numbers themselves.
_WSTOP = set("""a an the and or but of to in on at by for from with without
within across against under over above below between among about into onto
than then as is are was were be been being has have had do does did not no
nor so if when where while because although though since after before until
remain remains remained report reports reported extracted extract
unless whether that which who whom whose this these those it its their our
your his her they we you there here both each every either neither all any
some most more less fewer other others such own same only also still now
just even yet however therefore thus hence rather instead plus minus per via
cannot can could may might must shall should will would come comes came sits
sit gives give gave keeps keep kept moves move moved carries carry carried
falls fall fell rises rise rose drops drop dropped grows grow grew measures
measure measured reflects reflect reflected survives survive survived
belongs belong belonged affects affect affected treats treat appears appear
appeared qualifies qualify qualified clears clear cleared found finds find
lost lose loses depending spanning dropping reordering dampening quantized
partitioned refused fitted retracted shared paired contribute contributes
contributed approximately essentially nominal strict wider narrower worse
worst better best higher lower larger smaller wide old new global inner
heaviest pooled insufficient separate none why them together times itself
zero one two three four five six seven eight nine ten eleven twelve
thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
thirty forty fifty sixty seventy eighty ninety hundred thousand""".split())
# Follows a numeral but is not a count. Each checked by hand once.
NOT_COUNTED = {
    "logits",    # "Llama-3.2 1B/3B were distillations from 8B/70B logits"
    "weights",   # "derived from the other's pretraining weights"
    "accounts",  # VERB: "evaluation noise accounts for at least a third"
    "looks",     # VERB: "coverage looks comfortable"
    "recipes",   # "all five quantization recipes" -- the five are named
    # verbs and a preposition that follow a printed percentage. Invisible
    # until the escaped-percent bug was fixed; none is a count.
    "covers", "exceeds", "straddles", "versus",
}


def _derive_counted_nouns(text):
    """-> nouns the paper counts with a numeral (or a numeric macro)."""
    numeral = re.compile(
        r"(?:(?<![\w.\\])\d+(?:\.\d+)?(?![\d])"
        r"|\\(?:" + "|".join(sorted(_DEFINED_MACROS, key=len, reverse=True))
        + r")\b\{?\}?)")
    macro_lower = {m.lower() for m in _DEFINED_MACROS}
    found = set()
    for m in numeral.finditer(text):
        tail = re.sub(r"^\\?%", "", text[m.end():m.end() + 70])
        for tok in re.findall(r"[A-Za-z][A-Za-z-]*", tail)[:3]:
            w = tok.lower()
            if w in _WSTOP:
                break          # the numeral was not counting anything
            if len(w) <= 2:
                continue
            # mechanical filters only, no per-word judgement: plural heads,
            # no macro-name leakage from an adjacent macro, no compounds
            if (w not in macro_lower and "-" not in w
                    and w.endswith("s") and not w.endswith("ss")
                    and w[:-1] not in _WSTOP):
                found.add(w)
            break
    return found


_WNOUN_RE = re.compile(rf"^(?:{_WNOUN})$", re.I)
_derived = _derive_counted_nouns(re.sub(r"\s+", " ", body))
_uncovered = sorted(w for w in _derived
                    if w not in NOT_COUNTED and not _WNOUN_RE.match(w))
if _uncovered:
    fail.append(
        "the paper counts these with numerals but the word gate's vocabulary "
        f"does not cover them, so the same figure spelled out would pass: "
        f"{_uncovered}. Add each to _WNOUN, or to NOT_COUNTED with the reason "
        f"it is not a count.")
print(f"   counted nouns derived         : {len(_derived)} "
      f"({len(_uncovered)} uncovered)")

# ------------------------------------------- does the noun match the macro?
# THE DEFECT THIS EXISTS FOR. n_checkpoints counted base models, said so in
# its own description, was rendered as \Nckpt, and the prose around it said
# "checkpoints" -- in the abstract, the method section, and four generated
# documents. Every gate here compares a macro's VALUE to the registry and
# none compared the registry's description to the noun in the sentence, so
# the value was right, the description was right, and the claim was wrong.
#
# This is not general description-versus-prose checking. It is a per-macro
# unit declaration: a numeric macro may declare the noun phrase that is
# allowed to follow it, and the build fails when some OTHER noun does. A
# macro with no declared unit is exempt, so the table grows deliberately.
# It would have failed on "the \Nckpt{} checkpoints" the day it was written.
# Each entry is the HEAD noun, not the whole phrase: up to two modifiers are
# allowed in between, so "\\ProspClusN{} distinct checkpoints" passes and
# "\\NBase{} checkpoints" does not. What must match is the thing counted.
MACRO_UNITS = {
    # the two objects that were conflated
    "NBase":               r"models?",
    "NQuantCkpt":          r"checkpoints?",
    "CalibFitBases":       r"models?",
    "CalibCalBases":       r"models?|names?",
    # the prospective set, where the unit genuinely is the checkpoint
    "ProspClusN":          r"checkpoints?|runs?",
    "ProspBases":          r"models?",
    "ProspDsCkpts":        r"checkpoints?",
    "ProspGThreeCkpts":    r"checkpoints?",
    "ProspTailCkpts":      r"checkpoints?",
    "ProspDsBases":        r"models?",
    "ProspGThreeBases":    r"models?",
    "ProspTailBases":      r"models?",
    # the four cell populations: three different sets, two holding the same
    # number, which is why each is declared separately
    "NCellsTotal":         r"cells?",
    "CalibCellsAttempted": r"cells?",
    "CalibCellsSurvived":  r"cells?",
    "CalibCellsStarved":   r"cells?",
    "StratCellsInSample":  r"cells?",
    # the rest of the corpus-scale figures. "folds" for Nfam because
    # leave-one-family-out runs one fold per family; "intervals" for Nsch
    # because the envelope publishes one per scheme; "observations" for
    # ProspN because the sentence that says the rows are NOT independent
    # observations is the independence guard itself.
    "Nrows":               r"evaluations?|rows?|figure",
    "ProspN":              r"rows?|observations?",
    "Nsch":                r"schemes?|intervals?",
    "Nfam":                r"famil(?:y|ies)|folds?",
    "CalibUnitsFloor":     r"units?",
    "CalibUnitsMin":       r"checkpoints?|units?",
    "CalibUnitsMax":       r"checkpoints?|units?",
    "CalibPartitions":     r"seeds?|partitions?",
    "CalibRowsNvfp":       r"rows?",
    # --- declared 2026-10-11 from the derivation: every macro the paper
    # follows with a counted noun. Where the declaration was taken from the
    # prose the macro already had, this is a regression guard rather than a
    # verification -- see the note on the derivation below. The ones whose
    # name and noun disagreed were read by hand.
    "AdvBadRows":          r"rows?",
    "AdvCtrlRows":         r"rows?",
    "BigLosses":           r"losses|loss",
    "CellsBlocked":        r"cells?",
    "CkptsCellWfourSmall": r"checkpoints?",
    "CkptsCellWsixteenBig": r"checkpoints?",
    "ExcessN":             r"rows?",
    "FiniteN":             r"rows?",
    "GbSensMmluRows":      r"rows?",
    "GbSensNitemsRows":    r"rows?",
    "GemmaTwoFoldRows":    r"rows?",
    "LofoTotalPairs":      r"evaluations?|pairs?",
    "MaxBenchmarksPerRun": r"benchmarks?",
    # "gates" was a VERB at one site ("\MinCkpt{} gates both the coverage
    # verdict"); the sentence was reworded rather than the declaration widened
    "MinCkpt":             r"checkpoints?",
    "NRhCardsHarvested":   r"cards?",
    "NRhCardsListed":      r"cards?",
    # a count of benchmarks at one site and of the noise estimates built from
    # them at another; both are correct readings of the same figure
    "NoiseNBench":         r"benchmarks?|estimates?",
    "NonLThreeFoldRowsMax": r"rows?",
    "NpredNums":           r"numbers?",
    "NrowsRaw":            r"rows?",
    # the number of calibration families each row is scored under
    "PairsPerRow":         r"famil(?:y|ies)",
    "PooledPairs":         r"evaluations?|pairs?",
    "PooledRows":          r"rows?",
    "ProspAllRows":        r"rows?",
    "ProspCkptsFpdyn":     r"checkpoints?",
    "ProspCkptsWeightint": r"checkpoints?",
    "ProspCkptsWeightsixteen": r"checkpoints?",
    "ProspCkptsWithMiss":  r"checkpoints?",
    "ProspDsRows":         r"rows?",
    "ProspGThreeRows":     r"rows?",
    # the same 9 pre-registered repositories, described once as repos and once
    # as the cards that document them
    "ProspNameGatedRepos": r"repos?|repositories|cards?",
    "ProspRegisteredRepos": r"repos?|repositories",
    "ProspRowsFpdyn":      r"rows?",
    "ProspRowsWeightint":  r"rows?",
    "ProspTailRows":       r"rows?",
    "ProspTotalMisses":    r"misses|miss",
    "PubCards":            r"cards?",
    "PubChecked":          r"publishers?",
    "PubOthersCards":      r"cards?",
    "PubOthersCount":      r"publishers?",
    "PubOthersPaired":     r"cards?",
    "PubOthersVerifiableCards": r"cards?",
    "QwenThreeFoldRows":   r"rows?",
    "RowsCellWfourSmall":  r"rows?",
    "RowsCellWsixteenBig": r"rows?",
    "StratBandReplaced":   r"schemes?",
    "StratCellsBoth":      r"cells?",
    "StratFullLost":       r"rows?",
    "StratRows":           r"rows?",
    "StratSchemesNarrower": r"schemes?",
    "StratWidenGained":    r"rows?",
    "StratWidenGainedAll": r"rows?",
    "TuneOuterFolds":      r"folds?",
}
unit_fail = []
for mac, allowed in sorted(MACRO_UNITS.items()):
    pat = re.compile(r"\\" + mac + r"\b\{?\}?\s*(?:\\?%)?\s*(.{0,42})", re.S)
    ok = re.compile(r"^(?:[A-Za-z][A-Za-z0-9-]* ){0,2}(?:" + allowed
                    + r")\b", re.I)
    for m in pat.finditer(body):
        tail = re.sub(r"\s+", " ", m.group(1)).lstrip()
        # a macro followed by punctuation, a function word, another macro or
        # maths makes no claim about a unit here
        first = re.match(r"[A-Za-z][A-Za-z-]*", tail)
        if not first or first.group(0).lower() in _WSTOP:
            continue
        if ok.match(tail):
            continue
        unit_fail.append(f"\\{mac} is declared as {allowed!r} but reads "
                         f"“\\{mac}{{}} {tail}…”")
if unit_fail:
    fail.append("a macro is followed by a noun it does not count (fix the "
                "sentence, or widen the declaration in MACRO_UNITS if the "
                "noun is right):\n      " + "\n      ".join(unit_fail))
print(f"   macro unit declarations       : {len(MACRO_UNITS)} "
      f"({len(unit_fail)} mismatched)")

# -------------------------------- which macros OUGHT to declare a unit?
# MACRO_UNITS started as 26 entries out of ~314 defined macros, so the other
# ~288 were exempt by omission -- the protection covered the macros somebody
# thought of, which is the failure shape the derived noun vocabulary above
# exists to avoid. Same move again: a macro that is followed by a counted
# noun IS a counting macro and must declare its unit. Measured, not chosen.
#
# HONEST LIMIT. Where a declaration was written from the prose the macro
# already had, this gate is a REGRESSION guard: it cannot catch a wrong noun
# that was there when the declaration was made. It catches the noun changing
# under a macro, and it catches a new macro being used without a decision.
# The declarations whose name and noun disagreed were checked by hand.
_counting = {}
for _mac in sorted(_DEFINED_MACROS):
    for _m in re.finditer(r"\\" + _mac + r"\b\{?\}?\s*(?:\\?%)?\s*(.{0,48})",
                          body):
        _tail = re.sub(r"\s+", " ", _m.group(1)).lstrip()
        for _tok in re.findall(r"[A-Za-z][A-Za-z-]*", _tail)[:3]:
            _w = _tok.lower()
            if _w in _WSTOP or _w in {m.lower() for m in _DEFINED_MACROS}:
                break
            if _WNOUN_RE.match(_w):
                _counting.setdefault(_mac, set()).add(_w)
                break
# Counting macros that deliberately do not declare a unit, with the reason.
UNITLESS = {
    # "\\(\\leq\\AdvMaxB\\)B, losses more than four times worse" -- the unit
    # is B for billions of parameters, printed as a letter rather than a
    # noun, and "losses" belongs to the clause after the comma.
    "AdvMaxB": "unit is B (billions of parameters), not a counted noun",
}
_undeclared = sorted(set(_counting) - set(MACRO_UNITS) - set(UNITLESS))
if _undeclared:
    fail.append(
        "these macros are followed by a counted noun but declare no unit, so "
        "nothing checks that the noun is the one they count:\n      "
        + "\n      ".join(f"\\{m} -> {sorted(_counting[m])}"
                          for m in _undeclared)
        + "\n      Add each to MACRO_UNITS, or to UNITLESS with a reason.")
print(f"   counting macros               : {len(_counting)} "
      f"({len(_undeclared)} without a declaration)")

# ---------------------------------------------------------------- 2
head("2. CITATIONS")
keys = set(re.findall(r"@\w+\{([^,]+),", BIB))
cited = set()
for m in re.findall(r"\\cite\w*\{([^}]*)\}", TEX):
    cited |= {c.strip() for c in m.split(",")}
undef_c = sorted(cited - keys)
unused = sorted(keys - cited)
if undef_c:
    fail.append(f"cited but not in refs.bib: {undef_c}")
print(f"   bib entries {len(keys)} | cited {len(cited)} | undefined {undef_c} | uncited {unused}")
ids = sorted(set(re.findall(r"eprint\s*=\s*\{([\d.]+)\}", BIB)))
for i in ids:
    try:
        code = urllib.request.urlopen(
            urllib.request.Request(f"https://arxiv.org/abs/{i}",
                                   headers={"User-Agent": "curl/8"}),
            timeout=25).status
    except Exception as e:
        code = f"ERR {type(e).__name__}"
    ok = code == 200
    print(f"   arXiv:{i:<12} {'OK' if ok else code}")
    if not ok:
        fail.append(f"arXiv id does not resolve: {i}")

# author metadata: compare each bib entry's surnames and title against arXiv's
# own record. An id that resolves says nothing about who wrote the paper.
def _norm(x):
    import unicodedata
    x = re.sub(r"\\[`'^\"~=.]\{?(\w)\}?", r"\1", x)
    x = unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", x.lower()).strip()

try:
    _x = urllib.request.urlopen(urllib.request.Request(
        "https://export.arxiv.org/api/query?max_results=50&id_list=" + ",".join(ids),
        headers={"User-Agent": "curl/8"}), timeout=40).read().decode()
    _rec = {}
    for _e in re.findall(r"<entry>(.*?)</entry>", _x, re.S):
        _id = re.search(r"arxiv.org/abs/([\d.]+)", _e).group(1)
        _rec[_id] = ([_norm(a).split()[-1] for a in
                      re.findall(r"<author>\s*<name>(.*?)</name>", _e, re.S)],
                     _norm(re.sub(r"\s+", " ", re.search(
                         r"<title>(.*?)</title>", _e, re.S).group(1))))
    for _blk in re.findall(r"@\w+\{[^@]*?\n\}", BIB, re.S):
        _m = re.search(r"eprint\s*=\s*\{([\d.]+)\}", _blk)
        if not _m or _m.group(1) not in _rec:
            continue
        _au = re.search(r"author\s*=\s*\{(.*?)\},?\s*\n", _blk, re.S).group(1)
        _sur = [_norm(a.split(",")[0]).split()[-1] for a in
                re.split(r"\s+and\s+", re.sub(r"\s+", " ", _au))]
        _want = _rec[_m.group(1)][0]
        _ok = _sur == _want
        print(f"   authors {_m.group(1):<11} {'OK' if _ok else 'MISMATCH'}"
              + ("" if _ok else f"  bib={_sur} arxiv={_want}"))
        if not _ok:
            fail.append(f"bib authors differ from arXiv for {_m.group(1)}: "
                        f"{_sur} vs {_want}")
except Exception as _e:
    # NOT a warning. arXiv rate-limits (HTTP 429) and times out, and a
    # warning here failed test_both_paper_variants_pass_audit_paper_cleanly
    # on a tree with nothing wrong with it -- observed 2026-10-12. Same
    # confusion src/audit.py had: "could not check" reported as a defect.
    unchecked.append(f"bib authors against arXiv: {_e}")

# ---------------------------------------------------------------- 3
head("3. OVERCLAIM SCAN")
BANNED = [
    (r"\bfirst to\b", "novelty claim"),
    (r"\bnovel\b", "novelty claim"),
    (r"\bstate[- ]of[- ]the[- ]art\b", "SOTA claim"),
    (r"\bproves?\b", "proof language"),
    (r"\bguarantees? that\b", "unconditional guarantee"),
    (r"\balways\b", "absolute"),
    (r"\bnever fails?\b", "absolute"),
    (r"never used to build it", "retracted phrasing"),
    (r"\bunseen families\b", "retracted phrasing"),
    (r"\bsafe to adopt\b", "retired wording"),
]
low = TEX.lower()
hits = 0


def _in_error_record(pos):
    """True inside the Audit section or the process-error appendix.

    Those two are allowed to quote what they retracted. The test used to ask
    only whether \\section{Audit} appeared ANYWHERE earlier in the file, which
    exempted the Conclusion as well -- and, once the audit record was split,
    the whole appendix and everything after it. An exemption that widens
    whenever the document grows a section is the shape of a check that passes
    because nothing happens to reach it.
    """
    a = TEX.find("\\section{Audit}")
    c = TEX.find("\\section{Conclusion}")
    if a >= 0 and (c < 0 or a < pos < c):
        return True
    ap = TEX.find("\\appendix")
    return ap >= 0 and pos > ap


for pat, why in BANNED:
    for m in re.finditer(pat, low):
        seg = re.sub(r"\s+", " ", TEX[max(0, m.start()-70):m.start()+70])
        # the error record is allowed to quote what it retracted
        if "retracted" in why and _in_error_record(m.start()):
            continue
        hits += 1
        fail.append(f"overclaim [{why}]: ...{seg}...")
print(f"   banned patterns matched: {hits}")

# ---------------------------------------------------------------- 4
head("4. CROSS-ARM CHECK (verifier's artifact versus the pipeline's, "
     "both against the LIVE registry)")
# This section used to be numbered with the one below it under the single
# label "INDEPENDENT RE-DERIVATION (from raw source)". That label was true of
# section 5 and false of this one, which is half a section mislabelled, so
# they are split.
#
# What THIS section verifies. The registry derives prosp_n, prosp_inside,
# prosp_cov_pct, prosp_ci_lo and prosp_ci_hi from out/independent_check.csv
# -- the INDEPENDENT VERIFIER's per-row output, stdlib only, no project
# imports. The checks below recompute the same five from
# out/real_use_case.csv, the PIPELINE's own output, with a separate
# implementation of the inside-the-interval test. So this is not a
# re-derivation from raw data: it is a cross-check between the two arms,
# which is worth having and is worth not mistaking for something else.
#
# It is only meaningful when both artifacts are current. On 2026-10-01 a
# stale independent_check.csv against a fresh real_use_case.csv produced five
# simultaneous failures here, all of them correct. Both files are now in the
# artifact-staleness test (tests/test_all.py).
#
# The consequence for how the headline is sourced, stated here because it is
# easy to miss: the prospective coverage the abstract quotes is the
# verifier's figure, not the pipeline's. That is the stronger choice -- the
# number a reader sees was computed by a reimplementation sharing no code
# with the pipeline -- but it means this section is the ONLY place the
# pipeline's own figure is checked against it. Do not weaken it. The paper
# says so in S6 and README.md says so too.
#
# "paper" in every line below comes from the LIVE claims registry
# (src/verify_claims.registry()), the same source that drives
# paper/numbers.tex. A hard-coded literal here would let the audit pass while
# the paper said something different -- the pre-2026-09-27 failure mode this
# rewrite fixed. Tolerances are rounding-scale. If the registry does not
# export a key we fail rather than skip, so a rename cannot make the audit
# vacuous.
#   * "paper" comes from the LIVE claims registry (src/verify_claims.registry()),
#     which is the same source that drives paper/numbers.tex. A hard-coded
#     literal here would let the audit pass while the paper says something
#     different -- that was the pre-2026-09-27 failure mode this rewrite fixes.
#   * Tolerances are tight (rounding-scale) so a real drift is caught.
#   * If the registry does not export a key, we fail rather than silently
#     skip -- otherwise a rename in verify_claims would make the audit vacuous.
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "src"))
from verify_claims import registry as _reg_fn  # noqa: E402
REG = {k: v[0] for k, v in _reg_fn().items()}


def _claim(key):
    if key not in REG:
        fail.append(f"audit references a claim key that the registry does "
                    f"not export: {key!r}. Either the registry lost the key "
                    f"or the audit is out of date.")
        return float("nan")
    return REG[key]


def chk(label, recomputed, key, tol):
    """Compare a from-source recomputation against the LIVE registry value
    for the given key. Fails on absent-key, NaN, or drift."""
    claimed = _claim(key)
    ok = (not (recomputed != recomputed))  # not NaN
    ok = ok and (not (claimed != claimed))
    ok = ok and abs(recomputed - claimed) <= tol
    print(f"   {label:<34} recomputed {recomputed:<12.4f} paper "
          f"{claimed:<12.4f} (key {key})  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        fail.append(f"re-derivation mismatch on {label} vs claim {key}: "
                    f"recomputed {recomputed} paper {claimed} tol {tol}")


# (a) strict prospective coverage, straight from the per-row predictions
r = pd.read_csv(os.path.join(ROOT, "out", "real_use_case.csv"))
TRAIN = {"llama-3", "qwen2.5", "granite", "mistral", "qwen3", "gemma-2"}
s = r[(~r.family.isin(TRAIN)) & (r.group != "llama-4")]
# recomputation inside audit_paper.py is intentionally a SEPARATE
# implementation from the CSV's own inside_90 column; boundary points
# count as inside, matching the tolerance used elsewhere.
_EPS = 1e-9
_inside = ((s.delta >= s.lo - _EPS) & (s.delta <= s.hi + _EPS))
chk("strict coverage %", 100 * _inside.mean(), "prosp_cov_pct", 0.05)
chk("strict covered rows", float(_inside.sum()), "prosp_inside", 0)
chk("strict total rows", float(len(s)), "prosp_n", 0)

# (b) the Clopper-Pearson CI quoted in the abstract
k, n = int(_inside.sum()), len(s)
chk("CP lower bound", 100 * beta.ppf(0.025, k, n - k + 1),
    "prosp_ci_lo", 0.05)
chk("CP upper bound", 100 * beta.ppf(0.975, k + 1, n - k),
    "prosp_ci_hi", 0.05)

head("5. INDEPENDENT RE-DERIVATION FROM RAW SOURCE "
     "(versus the LIVE registry)")
# These two are raw in the sense the old label claimed: the figures are
# recomputed from the GPU run file and from data/dataset.csv, not from any
# artifact a pipeline script wrote.

# (c) adversarial tail, from the raw GPU run file
a = pd.read_csv(os.path.join(ROOT, "data", "adversarial", "adversarial_runs.csv"))
a["delta"] = a.acc_after - a.acc_before
bad = a[a.is_control == 0]
ctl = a[a.is_control == 1]
chk("worst adversarial delta", a.delta.min(), "adv_worst_delta", 0.01)
chk("faulted rows > -3pp (strict)", float((bad.delta < -3).sum()),
    "adv_bad_over_3pp", 0)
chk("faulted rows total", float(len(bad)), "adv_bad_rows", 0)
chk("control rows > -3pp (strict)", float((ctl.delta < -3).sum()),
    "adv_control_over_3pp", 0)

# (d) per-scheme severe rate, from the dataset
d = pd.read_csv(os.path.join(ROOT, "data", "dataset.csv"))
d = d[d.acc_before >= 20]
nv = d[d.scheme == "nvfp4"]
chk("nvfp4 severe-loss %", 100 * (nv.delta < -3).mean(),
    "severe_pct::nvfp4", 0.05)

# ---------------------------------------------------------------- report
head("RESULT")
for w in warn:
    print(f"   WARNING  {w}")
for f in fail:
    print(f"   FAIL     {f}")
for _u in unchecked:
    print(f"   UNCHECKED  {_u}")
print(f"\n   {len(fail)} failure(s), {len(warn)} warning(s), "
      f"{len(unchecked)} unchecked")
# A single machine-readable line, emitted exactly once, so a caller does not
# have to find the human summary by position. tests/test_all.py read
# stdout.splitlines()[-6:] and could therefore go red on a truncated or
# interleaved capture while nothing was wrong -- a gate that cries wolf costs
# more than it catches.
print(f"AUDIT_PAPER_RESULT variant={TEXNAME} "
      f"failures={len(fail)} warnings={len(warn)} "
      f"unchecked={len(unchecked)}")
sys.exit(1 if fail else 0)
