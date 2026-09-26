"""
Rank proposed quantization schemes by how safe they are to adopt without
running your own evaluation.

This ranks SCHEMES, not models. The underlying artifact has no per-model
signal (see ADVERSARIAL_AUDIT.md), so asking it about a specific checkpoint
would imply precision that does not exist.

Ranking inputs, in the order the ranking applies them:
  1. calibrated interval width  -- narrower means more predictable, so more
     defensible to skip your own testing
  2. worst observed delta vs a risk threshold -- the interval never excludes
     zero, so the tail is reported separately and can demote a scheme
  3. sample size behind the calibration -- thin support is stated, not hidden

Where the interval and the tail disagree, the disagreement is printed. A clean
rank that hides a -8.9pp observation would be the dishonest version of this.

  python src/rank.py w4a16 fp8 nvfp4 w8a8 --size 1.5B
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load  # noqa: E402
from strata import ConservativeStratified, annotate, size_band  # noqa: E402
import feedback  # noqa: E402
import cell_coverage  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")

DEFAULT_RISK_PP = 3.0        # a loss worse than this is "severe"
RISK_FLOOR_PP = 3.0          # threshold used for the footer tail statistic
SEVERE_RATE = 0.02           # >2% of observations severe -> demote
THIN_ROWS = 80               # below this, calibration support is thin
THIN_FAMILIES = 4            # below this, it is one vendor's models
POOR_COVERAGE = 0.85         # validated coverage below nominal by >5pp
REFUSE_BELOW = 0.85          # per-cell measured coverage below this -> no number
# Which coverage the refusal is scored on. The risk this tool exists to bound
# is accuracy LOSS, so the guarantee that matters is one-sided: does the true
# delta stay at or above the lower bound? A model that BEATS its envelope has
# not exposed anyone to anything, but two-sided containment counts that as a
# miss and can make a cell look uncalibrated when it is merely outperforming.
# See ONE_SIDED_COVERAGE.md for the external report that prompted this.
# There is deliberately no minimum-row floor here. One existed (50 "scored
# rows") and it was counted in row-by-calibration-family pairs, 7 per real row,
# so it really meant about 7 rows, and it was checked BEFORE the checkpoint
# floor, which let a one-checkpoint cell through as "trusted". The evidence
# floor is now the number of distinct checkpoints plus the checkpoint bootstrap,
# which already reflects how few rows there are.


def classify_cell(v):
    """One shared classifier for a coverage record: 'trusted',
    'insufficient_evidence', or 'refused'. Used for every (scheme, size band)
    cell AND for each scheme's all-sizes record, and by the top-level
    refused/insufficient-evidence lists, so the views cannot silently diverge.

    Order matters: the checkpoint floor comes first. A record with fewer than
    MIN_CELL_CHECKPOINTS distinct checkpoints cannot be judged whatever its row
    count or point estimate, so nothing may return 'trusted' ahead of it."""
    ckpts = v.get("distinct_checkpoints")
    if ckpts is not None and ckpts < MIN_CELL_CHECKPOINTS:
        return "insufficient_evidence"
    judged = v.get("coverage_one_sided")
    if judged is None:
        judged = v.get("coverage")
    if judged is None:
        return "trusted"  # nothing measured; nothing to judge
    boot_lo, boot_hi = v.get("boot90_lo"), v.get("boot90_hi")
    straddles = (boot_lo is not None and boot_hi is not None and
                 boot_lo < REFUSE_BELOW * 100 < boot_hi)
    if straddles:
        return "insufficient_evidence"
    if judged < REFUSE_BELOW:
        return "refused"
    return "trusted"


MIN_CELL_CHECKPOINTS = 3     # distinct checkpoints a cell needs to reach Tier A
# Also the coverage-judgment floor: below this many checkpoints, a cluster
# bootstrap over checkpoints has too few achievable resample compositions to
# be a meaningful interval (e.g. 2 checkpoints -> only 3 possible resample
# mixtures out of 20000 draws), so the tool cannot confidently call a cell
# either "trusted" or "refused" -- it is a third state, "insufficient
# evidence." Independently verified external finding; see ONE_SIDED_COVERAGE.md.

# Why distinct checkpoints and not scored rows: one checkpoint evaluated on 13
# benchmarks produces 13 rows, but they are 13 correlated views of a single
# quantization run, not 13 independent pieces of evidence. A cell can therefore
# measure 100% coverage on many scored rows while resting on one model.
CELL_COVERAGE = os.path.join(HERE, "..", "out", "cell_coverage.json")


def load_cell_coverage():
    """Measured per-(scheme, band) coverage from src/cell_coverage.py."""
    try:
        with open(CELL_COVERAGE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"cells": {}, "bands": {}, "pooled": {}, "widening": {}}

# Why a RATE and not the single worst observation: the worst value grows with
# sample size, so thresholding on it flags every well-studied scheme and tells
# the user to test everything. The rate is comparable across schemes. The worst
# value is still always printed, because it is what actually bites.

ALIASES = {
    "w4a16": "w4a16", "int4": "w4a16", "gptq": "w4a16", "4bit": "w4a16",
    "w8a8": "w8a8_int", "w8a8_int": "w8a8_int", "int8": "w8a8_int",
    "smoothquant": "w8a8_int", "w8a16": "w8a16",
    "fp8": "fp8", "fp8_static": "fp8",
    "fp8_dynamic": "fp8_dynamic", "fp8dynamic": "fp8_dynamic",
    "nvfp4": "nvfp4", "fp4": "nvfp4",
}
LABELS = {
    "w4a16": "W4A16 (4-bit weights, 16-bit activations)",
    "w8a8_int": "W8A8-INT (8-bit int weights + activations)",
    "w8a16": "W8A16 (8-bit int weights, 16-bit activations)",
    "fp8": "FP8 (8-bit float, static scales)",
    "fp8_dynamic": "FP8-dynamic (8-bit float, dynamic scales)",
    "nvfp4": "NVFP4 (4-bit float weights + activations)",
}
AGGRESSIVE = {"w4a16", "nvfp4"}


def resolve(name):
    return ALIASES.get(str(name).strip().lower().replace("-", "_"))


# ------------------------------------------------------------------ evidence
# Coverage evidence comes from cell_coverage.py, the same code and the same
# checkpoint bootstrap for the (scheme, band) cells and for each scheme.

def build_table(d=None):
    """Everything the ranking needs, derived from data rather than hardcoded."""
    if d is None:
        d = load(DATA)
    d = annotate(d)
    m = ConservativeStratified().fit(d)
    cov = cell_coverage.scheme_records(cell_coverage.scored_pairs(d))

    table = {}
    for s, g in d.groupby("scheme"):
        base = m.by_scheme[s]
        # size gradient: is this scheme more damaging on small models?
        grad = {}
        for b in ("<2B", "2-10B", ">10B"):
            gb = g[g.band == b]
            if len(gb) >= 5:
                grad[b] = {"n": int(len(gb)), "mean": float(gb.delta.mean()),
                           "worst": float(gb.delta.min()),
                           "checkpoints": int(gb.base_model.nunique())}
        sev = {}
        for thr in (2.0, 3.0, 4.0):
            sev[str(thr)] = float((g.delta <= -thr).mean())
        table[s] = {
            "scheme": s,
            "severe_rate": sev,
            "n_severe_3pp": int((g.delta <= -3.0).sum()),
            "label": LABELS.get(s, s),
            "mean": base["mean"],
            "half_width": base["half_width"],
            "lo": base["mean"] - base["half_width"],
            "hi": base["mean"] + base["half_width"],
            "worst_observed": base["worst"],
            "p05": base["p05"],
            "n": base["n"],
            "n_checkpoints": base["n_checkpoints"],
            "n_families": base["n_families"],
            # Scheme-level coverage, one-sided, judged by classify_cell just like a cell.
            "coverage_one_sided": float(cov.get(s, {}).get("coverage_one_sided", float("nan"))),
            "coverage_two_sided": float(cov.get(s, {}).get("coverage", float("nan"))),
            "coverage_rows": int(cov.get(s, {}).get("distinct_rows", 0)),
            "coverage_ckpts": int(cov.get(s, {}).get("distinct_checkpoints", 0)),
            "coverage_scored_pairs": int(cov.get(s, {}).get("scored_pairs", 0)),
            "coverage_boot90": [cov.get(s, {}).get("boot90_lo"), cov.get(s, {}).get("boot90_hi")],
            "coverage_state": classify_cell(cov[s]) if s in cov else "trusted",
            "coverage_worst_family": float(cov.get(s, {}).get("worst_family_one_sided", float("nan"))),
            "coverage_worst_family_name": cov.get(s, {}).get("worst_family_name", "?"),
            "coverage_n_families": int(cov.get(s, {}).get("n_families_tested", 0)),
            "coverage_by_family": cov.get(s, {}).get("by_family_one_sided", {}),
            "coverage_spread": float(cov.get(s, {}).get("spread_one_sided", 0.0)),
            "size_gradient": grad,
            "widened_bands": sorted(
                b for (sc, b), c in m.by_stratum.items()
                if sc == s and c["half_width"] > base["half_width"]),
        }
    # Facts quoted in the output footer, recomputed every run so prose can
    # never drift from the data (see CLAIMS.md).
    ex = sum(1 for e in table.values() if not (e["lo"] <= 0 <= e["hi"]))
    big = d[d.delta < -RISK_FLOOR_PP]
    below = int(sum(1 for _, r in big.iterrows()
                    if r.delta < table[r.scheme]["lo"]))
    widen = [k for k, c in m.by_stratum.items()
             if c["half_width"] > m.by_scheme[k[0]]["half_width"]]
    cc = load_cell_coverage()
    table["_meta"] = {
        "n_schemes": len(table),
        "intervals_excluding_zero": ex,
        "n_big_losses": int(len(big)),
        "n_big_losses_below_floor": below,
        "pct_big_losses_below_floor": (100.0 * below / len(big)
                                       if len(big) else float("nan")),
        "n_numbers_in_predictor": 2 * len(m.by_scheme) + len(widen),
        "n_size_dependent_widths": len(widen),
        "n_rows": int(len(d)),
        "n_checkpoints": int(d.base_model.nunique()),
        "n_families": int(d.family.nunique()),
        "pooled_cell_coverage": cc.get("pooled", {}).get("coverage"),
        "pooled_distinct_rows": cc.get("pooled", {}).get("distinct_rows"),
        "pooled_scored_pairs": cc.get("pooled", {}).get("scored_pairs"),
        "pooled_pairs_per_row": cc.get("pooled", {}).get("pairs_per_row"),
        "refused_cells": sorted(
            k for k, v in cc.get("cells", {}).items()
            if classify_cell(v) == "refused"),
        "insufficient_evidence_cells": sorted(
            k for k, v in cc.get("cells", {}).items()
            if classify_cell(v) == "insufficient_evidence"),
    }
    return table


# ------------------------------------------------------------------- ranking
def assess(e, risk_pp, band=None, moe=False, cell_cov=None):
    """Flags and notes for one scheme. Notes are printed, never swallowed."""
    flags, notes = [], []
    e["refused"] = False
    e["insufficient_evidence"] = False

    # Refusal: if this exact (scheme, size band) cell was MEASURED to cover far
    # below nominal, the tool declines to print an interval rather than show a
    # number that looks as confident as a well-calibrated one.
    #
    # Before that judgment is trusted at all, two evidence-floor checks run
    # first. Either one produces a THIRD state -- "insufficient evidence" --
    # distinct from both "trusted" and "refused":
    #   (a) fewer than MIN_CELL_CHECKPOINTS distinct checkpoints. The same
    #       floor already applied to training support now applies to the
    #       coverage judgment itself.
    #   (b) the checkpoint-cluster bootstrap 90% interval on one-sided
    #       coverage straddles REFUSE_BELOW. A point estimate can sit below
    #       the line while the interval around it is too wide to say so with
    #       any confidence.
    # A cell can fail (a) or (b) and still have a point estimate comfortably
    # above REFUSE_BELOW -- that is not a contradiction, it means the point
    # estimate itself is not trustworthy yet, which is the whole reason for
    # this state. (External finding, independently verified; see
    # ONE_SIDED_COVERAGE.md.)
    if band and cell_cov:
        c = cell_cov.get("cells", {}).get(f"{e['scheme']}|{band}")
        # one-sided where available; fall back to two-sided for artifacts
        # written before coverage_one_sided existed
        judged = c.get("coverage_one_sided") if c else None
        if judged is None and c:
            judged = c.get("coverage")
        ckpts = c.get("distinct_checkpoints") if c else None
        boot_lo = c.get("boot90_lo") if c else None
        boot_hi = c.get("boot90_hi") if c else None
        thin_ckpts = ckpts is not None and ckpts < MIN_CELL_CHECKPOINTS
        straddles = (boot_lo is not None and boot_hi is not None and
                     boot_lo < REFUSE_BELOW * 100 < boot_hi)
        state = classify_cell(c) if c else "trusted"
        if state == "insufficient_evidence":
            e["insufficient_evidence"] = True
            e["evidence_coverage"] = judged
            e["evidence_ckpts"] = ckpts
            e["evidence_boot90"] = [boot_lo, boot_hi]
            e["evidence_reason"] = (
                "checkpoint floor" if thin_ckpts and not straddles else
                "bootstrap straddle" if straddles and not thin_ckpts else
                "checkpoint floor and bootstrap straddle")
            flags.append("INSUFFICIENT_EVIDENCE")
            notes.append(
                f"{e['scheme']} at {band} is neither trusted nor refused: "
                f"its measured one-sided coverage is {judged*100:.1f}% from "
                f"only {ckpts} distinct checkpoints, and the checkpoint "
                f"bootstrap 90% interval [{boot_lo:.0f}%, {boot_hi:.0f}%] "
                f"{'straddles' if straddles else 'sits below'} the "
                f"{REFUSE_BELOW*100:.0f}% line the tool judges against "
                f"-- {e['evidence_reason']}. There is not enough independent "
                f"evidence here to call this cell either way; run your own "
                f"evaluation")
        elif state == "refused":
            e["refused"] = True
            e["refusal_coverage"] = judged
            e["refusal_two_sided"] = c["coverage"]
            e["refusal_rows"] = c["distinct_rows"]
            e["refusal_ckpts"] = ckpts
            e["refusal_fams"] = c.get("distinct_families")
            flags.append("INSUFFICIENT_CALIBRATION")
            notes.append(
                f"no interval is shown for {e['scheme']} at {band}: the true "
                f"result stayed at or above the lower bound only "
                f"{judged*100:.1f}% of the time here "
                f"({c['distinct_rows']} rows from "
                f"{c.get('distinct_checkpoints', '?')} distinct checkpoints "
                f"across {c.get('distinct_families', '?')} families), below the "
                f"{REFUSE_BELOW*100:.0f}% refusal threshold (the one-sided "
                f"nominal level is about 95%). Run your own evaluation for "
                f"this combination")

    # Worst observed loss is stated for EVERY scheme, at every rank and tier.
    if e["worst_observed"] <= -risk_pp:
        notes.append(
            f"worst observed loss {e['worst_observed']:+.2f}pp, on at least "
            f"one published checkpoint - ranking high means predictable on "
            f"average, not safe in the worst case")
    else:
        notes.append(
            f"worst observed loss {e['worst_observed']:+.2f}pp across "
            f"{e['n']} evaluations, which does not reach the "
            f"{risk_pp:.0f}pp severe threshold")

    # (2) tail risk: how OFTEN a severe loss happened, not just whether one did
    rate = e["severe_rate"].get(str(float(risk_pp)))
    if rate is None:
        rate = float(np.nan)
    e["severe_rate_used"] = rate
    margin_n = e["n"] - THIN_ROWS
    margin_f = e["n_families"] - THIN_FAMILIES
    if 0 <= margin_n <= 10 or 0 <= margin_f <= 1:
        notes.append(
            f"only just clears the thin-data test ({e['n']} evaluations vs a "
            f"{THIN_ROWS} threshold, {e['n_families']} families vs "
            f"{THIN_FAMILIES}) - its rank rests on barely enough evidence")

    if not np.isnan(rate) and rate > SEVERE_RATE:
        flags.append("TAIL_RISK")
        notes.append(
            f"{rate*100:.1f}% of observed evaluations lost more than "
            f"{risk_pp:.0f}pp ({e['n_severe_3pp']} of {e['n']}) - that is "
            f"{rate/SEVERE_RATE:.0f}x the {SEVERE_RATE*100:.0f}% threshold "
            f"used to trigger this flag")

    # the discrepancy the audit asked to be surfaced: the interval is silent
    # about a tail that extends well past its own floor
    gap = e["lo"] - e["worst_observed"]
    if gap > e["half_width"]:
        notes.append(
            f"interval floor is {e['lo']:+.2f}pp but the worst observed loss "
            f"is {e['worst_observed']:+.2f}pp, {gap:.2f}pp further down - the "
            f"90% interval does not describe this tail and is not meant to")

    # (3) thin support
    if e["n"] < THIN_ROWS or e["n_families"] < THIN_FAMILIES:
        flags.append("THIN_DATA")
        notes.append(
            f"backed by only {e['n']} evaluations from {e['n_checkpoints']} "
            f"checkpoints across {e['n_families']} model families")

    # Scheme-level coverage (all sizes pooled) is judged by the SAME classifier
    # as a size cell: one-sided coverage, checkpoint bootstrap, checkpoint
    # floor. Sharing classify_cell is what keeps the two levels from
    # disagreeing about what counts as evidence.
    cstate = e.get("coverage_state", "trusted")
    if cstate != "trusted":
        blo, bhi = e.get("coverage_boot90", [None, None])
        one = e["coverage_one_sided"]
        if cstate == "insufficient_evidence":
            if "INSUFFICIENT_EVIDENCE" not in flags:
                flags.append("INSUFFICIENT_EVIDENCE")
            why = ("fewer than %d checkpoints" % MIN_CELL_CHECKPOINTS
                   if e["coverage_ckpts"] < MIN_CELL_CHECKPOINTS else
                   "the checkpoint-bootstrap 90%% interval [%.0f%%, %.0f%%] "
                   "straddles the %.0f%% line" % (blo, bhi, REFUSE_BELOW * 100))
            notes.append(
                f"across all sizes, results stayed at or above this scheme's "
                f"lower bound {one*100:.1f}% of the time, from "
                f"{e['coverage_rows']} rows and {e['coverage_ckpts']} "
                f"checkpoints, but {why}, so coverage is not established "
                f"either way")
        else:
            flags.append("UNDERCOVERED")
            notes.append(
                f"across all sizes, results stayed at or above this scheme's "
                f"lower bound only {one*100:.1f}% of the time, from "
                f"{e['coverage_rows']} rows and {e['coverage_ckpts']} "
                f"checkpoints (bootstrap 90% interval [{blo:.0f}%, "
                f"{bhi:.0f}%], entirely below the {REFUSE_BELOW*100:.0f}% line)")

    # instability across families is context, not a verdict: always surfaced
    wf = e.get("coverage_worst_family", float("nan"))
    if not np.isnan(wf) and e.get("coverage_spread", 0) > 0.15:
        notes.append(
            f"loss-side coverage is uneven across model families - averaged "
            f"{e['coverage_one_sided']*100:.0f}% but fell to {wf*100:.0f}% "
            f"when {e['coverage_worst_family_name']} was held out, over "
            f"{e['coverage_n_families']} families of evidence")

    # size interaction
    g = e["size_gradient"]
    if band and band in g and "&gt;10B" not in band:
        if band == "<2B" and e["scheme"] in AGGRESSIVE:
            flags.append("SIZE_RISK")
            small, large = g.get("<2B"), g.get(">10B")
            if small and large:
                notes.append(
                    f"on models under 2B this scheme averaged "
                    f"{small['mean']:+.2f}pp vs {large['mean']:+.2f}pp on "
                    f"models over 10B, roughly "
                    f"{abs(small['mean']/large['mean']):.1f}x the damage - "
                    f"but that is from only {small['n']} evaluations on "
                    f"{small['checkpoints']} small checkpoints, so treat the "
                    f"ratio as a direction, not a number")
    # When a size band is given and we are NOT refusing, state the measured
    # coverage for that exact cell rather than a blanket claim about size.
    if band and cell_cov and not e["refused"] and not e["insufficient_evidence"]:
        c = cell_cov.get("cells", {}).get(f"{e['scheme']}|{band}")
        if c:
            notes.append(
                f"at {band} results stayed at or above this scheme's lower "
                f"bound {c['coverage_one_sided']*100:.1f}% of the time "
                f"({c['distinct_rows']} rows from {c['distinct_checkpoints']} "
                f"checkpoints); the interval shown is the all-sizes one, not "
                f"a {band}-specific one")
        else:
            notes.append(
                f"no coverage has been measured for {e['scheme']} at {band}")

    # Training-support floor: measured coverage alone is not enough to earn
    # Tier A if the cell rests on too few distinct checkpoints.
    if band and cell_cov:
        sup = cell_cov.get("support", {}).get(f"{e['scheme']}|{band}")
        if sup and sup["train_checkpoints"] < MIN_CELL_CHECKPOINTS:
            flags.append("THIN_CELL_SUPPORT")
            notes.append(
                f"this size cell rests on {sup['train_checkpoints']} distinct "
                f"checkpoint(s) ({sup['train_rows']} training rows). "
                f"{MIN_CELL_CHECKPOINTS} are required before a cell can reach "
                f"Tier A, because rows from one checkpoint across many "
                f"benchmarks are correlated views of a single quantization "
                f"run, not independent evidence")

    if moe:
        flags.append("MOE_UNVALIDATED")
        mo = (cell_cov or {}).get("moe", {})
        notes.append(
            f"mixture-of-experts support in the training data is "
            f"{mo.get('rows', 'a small number of')} rows from "
            f"{mo.get('checkpoints', 'few')} checkpoints; no MoE-specific "
            f"interval is fitted")

    return flags, notes


def tier_of(flags):
    if "INSUFFICIENT_CALIBRATION" in flags or "INSUFFICIENT_EVIDENCE" in flags:
        return "C"
    if {"TAIL_RISK", "UNDERCOVERED"} & set(flags):
        return "C"
    if flags:
        return "B"
    return "A"


# Tier wording is deliberately mechanical: each label states the rule that
# produced it, not a judgement about safety. An earlier version read "safe to
# adopt without running your own eval", which was false for FP8-dynamic
# (Tier A, worst observed -8.72pp).
TIER_MEANING = {
    "A": "no flag triggered: severe-loss rate at or below 2%, "
         "coverage established at or above 85%, support above thresholds",
    "B": "one or more non-severe flags triggered (see below)",
    "C": "severe-loss rate above 2%, or coverage not established at 85% "
         "(too few checkpoints, or the bootstrap interval reaches below it)",
}
# One plain sentence a non-expert can act on. The tier states the RULE; this
# states the CONSEQUENCE. Both are shown because the rule alone reads as jargon
# and the consequence alone reads as advice the evidence cannot support.
PLAIN = {
    "A": "Of the schemes compared, this one's published results are the most "
         "consistent. Still read the worst-case line below.",
    "B": "Usable, but something about the evidence is thin or unusual. Read "
         "the notes before relying on it.",
    "C": "Run your own evaluation before shipping this. The published results "
         "are either volatile or too weakly supported to lean on.",
}


BANDS = ("<2B", "2-10B", ">10B")


def band_verdicts(scheme, cell_cov):
    """The coverage verdict for each size band, by the same classifier. Shown
    when no size is given, so an all-sizes tier is never read as a statement
    about a band it cannot support."""
    out = {}
    for b in BANDS:
        c = (cell_cov or {}).get("cells", {}).get(f"{scheme}|{b}")
        if c is None:
            out[b] = {"state": "not_measured"}
        else:
            out[b] = {"state": classify_cell(c),
                      "checkpoints": c.get("distinct_checkpoints"),
                      "rows": c.get("distinct_rows"),
                      "one_sided": c.get("coverage_one_sided"),
                      "boot90": [c.get("boot90_lo"), c.get("boot90_hi")]}
    return out


def _band_text(v):
    st = v["state"]
    if st == "not_measured":
        return "not measured (no evaluations at this size)"
    ck = v["checkpoints"]
    ckt = f"{ck} checkpoint" + ("" if ck == 1 else "s")
    lo, hi = v["boot90"]
    if st == "trusted":
        return f"trusted ({ckt}, {v['rows']} rows, {v['one_sided']*100:.0f}% at or above the lower bound)"
    if ck is not None and ck < MIN_CELL_CHECKPOINTS:
        return f"insufficient evidence ({ckt}, {v['rows']} rows)"
    if st == "insufficient_evidence":
        return f"insufficient evidence ({ckt}; bootstrap 90% [{lo:.0f}, {hi:.0f}]% straddles {REFUSE_BELOW*100:.0f}%)"
    return f"refused ({ckt}; bootstrap 90% [{lo:.0f}, {hi:.0f}]% below {REFUSE_BELOW*100:.0f}%)"


def rank(schemes, table=None, risk_pp=DEFAULT_RISK_PP, band=None, moe=False,
         cell_cov=None):
    if table is None:
        table = build_table()
    if cell_cov is None:
        cell_cov = load_cell_coverage()
    out, unknown = [], []
    for name in schemes:
        key = resolve(name)
        if key is None or key == "_meta" or key not in table:
            unknown.append(name)
            continue
        e = dict(table[key])
        e["flags"], e["notes"] = assess(e, risk_pp, band, moe, cell_cov)
        e["tier"] = tier_of(e["flags"])
        e["band_verdicts"] = band_verdicts(key, cell_cov)
        e["requested_as"] = name
        out.append(e)
    # tier first, then predictability (narrower interval), then milder tail
    out.sort(key=lambda e: (e["tier"], e["half_width"], -e["worst_observed"]))
    for i, e in enumerate(out, 1):
        e["rank"] = i
    return out, unknown


# -------------------------------------------------------------------- report
def report(ranked, unknown, risk_pp, band, moe, meta=None):  # noqa: C901
    w = 78
    print()
    ctx = []
    if band:
        ctx.append(f"model size {band}")
    if moe:
        ctx.append("mixture-of-experts")
    ctx.append(f"severe-loss threshold {risk_pp:.1f}pp")
    print("Ranking quantization schemes  (" + ", ".join(ctx) + ")")
    print("=" * w)
    M = meta or {}
    print(f"Historical baseline over {M.get('n_rows', 0)} published "
          f"evaluations from {M.get('n_checkpoints', 0)} checkpoints across "
          f"{M.get('n_families', 0)} model families.")
    print("Sorted by tier, then by how tightly past results clustered.\n")
    print("How to read the interval: it is the range that contained 9 out of")
    print("10 published results for this scheme. It is centred on the average,")
    print("so it is symmetric - real damage is left-skewed, which is why the")
    print("worst-case line matters more than the interval's lower edge.\n")

    for e in ranked:
        head = (f"{e['rank']}. {e['label']}")
        print(head)
        print("   " + "-" * (len(head) - 3 + 3))
        if e.get("refused"):
            print("   expected change   NOT SHOWN")
            print("   90% interval      INSUFFICIENT CALIBRATION FOR THIS "
                  "COMBINATION")
            print(f"                     losses stayed above the lower bound "
                  f"only {e['refusal_coverage']*100:.1f}% of the time, over "
                  f"{e['refusal_rows']} rows from "
                  f"{e.get('refusal_ckpts', '?')} distinct checkpoints")
            if e.get("refusal_two_sided") is not None:
                print(f"                     (two-sided containment "
                      f"{e['refusal_two_sided']*100:.1f}%; the refusal is "
                      f"scored one-sided - see ONE_SIDED_COVERAGE.md)")
            for ln in feedback.refused_prompt([e["scheme"]], w - 21):
                print("                     " + ln)
        else:
            print(f"   expected change   {e['mean']:+.2f}pp     "
                  f"90% interval  [{e['lo']:+.2f}, {e['hi']:+.2f}]pp "
                  f"(+/-{e['half_width']:.2f})")
        sr = e.get("severe_rate_used", float("nan"))
        srtxt = "n/a" if np.isnan(sr) else f"{sr*100:.1f}%"
        print(f"   worst ever seen   {e['worst_observed']:+.2f}pp     "
              f"5th percentile {e['p05']:+.2f}pp")
        print(f"   severe losses     {srtxt} of evaluations lost more than "
              f"{risk_pp:.0f}pp")
        one, two = e["coverage_one_sided"], e["coverage_two_sided"]
        blo, bhi = e.get("coverage_boot90", [None, None])
        covtxt = ("n/a" if np.isnan(one) else
                  f"{one*100:.0f}% of results at or above the lower bound "
                  f"(bootstrap 90% [{blo:.0f}, {bhi:.0f}]%)")
        wf = e.get("coverage_worst_family", float("nan"))
        wftxt = ("" if np.isnan(wf) else
                 f", worst family {wf*100:.0f}% on "
                 f"{e['coverage_worst_family_name']}")
        print(f"   evidence          {e['n']} evals / "
              f"{e['n_checkpoints']} checkpoints / {e['n_families']} families")
        print(f"   validated cover   {covtxt}")
        print(f"                     {two*100:.0f}% inside the full interval; "
              f"{e['coverage_rows']} rows from {e['coverage_ckpts']} "
              f"checkpoints{wftxt}")
        if band is None:
            bv = e.get("band_verdicts", {})
            for j, b_ in enumerate(BANDS):
                if b_ in bv:
                    print(("   by size band      " if j == 0 else " " * 21)
                          + f"{b_:<6}" + _band_text(bv[b_]))
        for j, line in enumerate(_wrap(PLAIN[e["tier"]], w - 21)):
            print(("   what this means   " if j == 0 else " " * 21) + line)
        print(f"   tier              {e['tier']} - {TIER_MEANING[e['tier']]}")
        if e["flags"]:
            print(f"   flags             {', '.join(e['flags'])}")
        if band is None and e["tier"] in ("A", "B"):
            weak = [b_ for b_, v in e.get("band_verdicts", {}).items()
                    if v["state"] != "trusted"]
            if weak:
                for ln in _wrap(
                        f"!! this tier describes all sizes pooled. "
                        f"{len(weak)} of {len(BANDS)} size bands "
                        f"({', '.join(weak)}) are not trusted on their own "
                        f"evidence, so enter --size for the band you use "
                        f"before relying on it.", w - 5):
                    print("   " + ln)
        for n in e["notes"]:
            for j, line in enumerate(_wrap(n, w - 8)):
                print(("   !! " if j == 0 else "      ") + line)
        print()

    if unknown:
        print(f"not recognised: {', '.join(unknown)}")
        print(f"known schemes : {', '.join(sorted(set(ALIASES.values())))}\n")

    fb = feedback.prompt_line(w - 2)
    if fb:
        for ln in fb:
            print("  " + ln)
        print()
    print("-" * w)
    print("Limits that apply to every row above (recomputed each run):")
    M = meta or {}
    lims = [
        f"{M.get('intervals_excluding_zero', 0)} of "
        f"{M.get('n_schemes', 0)} intervals exclude zero, so this cannot tell "
        f"you a scheme will definitely cost accuracy",
        f"{M.get('n_big_losses_below_floor', 0)} of "
        f"{M.get('n_big_losses', 0)} observed losses worse than "
        f"{RISK_FLOOR_PP:.0f}pp fell below their interval floor "
        f"({M.get('pct_big_losses_below_floor', float('nan')):.0f}%)",
        f"built on {M.get('n_rows', 0)} published evaluations from "
        f"{M.get('n_checkpoints', 0)} checkpoints that Red Hat chose to "
        f"release, so damage from an untuned recipe is not represented",
        f"the predictor is {M.get('n_numbers_in_predictor', 0)} numbers and "
        f"reads exactly 2 inputs: scheme, and size band via "
        f"{M.get('n_size_dependent_widths', 0)} size-dependent widths. It "
        f"reads nothing else about your model",
    ]
    lims.insert(0,
        "TREAT EVERY INTERVAL AS A FLOOR ON RISK, NOT A CEILING. We measured "
        "deliberately-bad quantization configs on a GPU and found real losses "
        "up to -39.5pp, roughly 4x worse than anything in the published data "
        "this tool is calibrated on. Published recipes are the ones that "
        "worked; yours may not be one of them.")
    if M.get("pooled_cell_coverage") is not None:
        lims.append(
            f"pooled measured coverage {M['pooled_cell_coverage']*100:.1f}% "
            f"over {M['pooled_distinct_rows']} rows, each scored under "
            f"{M['pooled_pairs_per_row']:.0f} calibration families; cells refused for "
            f"insufficient calibration: "
            f"{', '.join(M.get('refused_cells', [])) or 'none'}")
        lims.append(
            "cells with insufficient evidence to judge either way (fewer "
            f"than {MIN_CELL_CHECKPOINTS} checkpoints or a checkpoint-"
            f"bootstrap interval straddling {REFUSE_BELOW*100:.0f}%): "
            f"{', '.join(M.get('insufficient_evidence_cells', [])) or 'none'}")
    for L in lims:
        for j, line in enumerate(_wrap(L, w - 4)):
            print(("  - " if j == 0 else "    ") + line)
    print()


def _wrap(t, width):
    words, line, out = t.split(), "", []
    for word in words:
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Rank quantization schemes by adoption risk.")
    ap.add_argument("schemes", nargs="*", help="e.g. w4a16 fp8 nvfp4 w8a8")
    ap.add_argument("--size", help="model size, e.g. 1.5B or 70B "
                                   "(applies the size-risk check)")
    ap.add_argument("--moe", action="store_true",
                    help="model is mixture-of-experts")
    ap.add_argument("--risk", type=float, default=DEFAULT_RISK_PP,
                    help=f"severe-loss threshold in pp "
                         f"(default {DEFAULT_RISK_PP})")
    ap.add_argument("--json", action="store_true", help="machine-readable")
    ap.add_argument("--all", action="store_true", help="rank every scheme")
    ap.add_argument("--form-fields", action="store_true",
                    help="print the fields a result-collection form needs")
    a = ap.parse_args(argv)

    if a.form_fields:
        print("Fields for the result-collection form:\n")
        for name, req, help_ in feedback.form_spec():
            print(f"  {name}  [{req}]")
            print(f"      {help_}")
        dest, ok = feedback.destination()
        print(f"\ndestination: {dest}   configured: {ok}")
        return 0

    table = build_table()
    schemes = (sorted(k for k in table if k != "_meta")
               if (a.all or not a.schemes) else a.schemes)

    band = None
    if a.size:
        try:
            band = size_band(float(str(a.size).lower().rstrip("b")))
        except ValueError:
            print(f"could not read size '{a.size}', ignoring")

    ranked, unknown = rank(schemes, table, a.risk, band, a.moe)
    feedback.log_request([e["scheme"] for e in ranked], band, a.moe,
                         [f"{e['scheme']}|{band}" for e in ranked
                          if e.get("refused")])
    if a.json:
        print(json.dumps({"context": {"size_band": band, "moe": a.moe,
                                      "risk_pp": a.risk},
                          "ranked": ranked, "unknown": unknown},
                         indent=2, default=float))
    else:
        report(ranked, unknown, a.risk, band, a.moe,
               table.get('_meta'))

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ranking_table.json"), "w") as f:
        json.dump(table, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
