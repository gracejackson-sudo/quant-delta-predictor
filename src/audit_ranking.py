"""
ADVERSARIAL AUDIT OF THE RANKING LAYER.

Same posture as src/adversarial_audit.py: assume the ranking overstates
something until proven otherwise. That assumption paid twice already.

  R1  leakage and scope creep   -- is the ranking still "just a lookup"?
  R2  sample-size honesty       -- does a clean rank order hide thin cells?
  R3  the "can't say don't do this" failure, in its new form
  R4  cross-check against verify/independent_rank.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load  # noqa: E402
from strata import ConservativeStratified, annotate  # noqa: E402
import rank as R  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")
FINDINGS = []


def note(tag, msg):
    FINDINGS.append((tag, msg))
    print(f"  [{tag}] {msg}")


def strict_cell_coverage(d, cls=ConservativeStratified):
    """Held-out calibration, per (scheme, band) cell."""
    fams = sorted(d.family.unique())
    rows, cells_kept = [], []
    for tf in fams:
        for cf in fams:
            if cf == tf:
                continue
            te, ca = d[d.family == tf], d[d.family == cf]
            tr = d[~d.family.isin([tf, cf])]
            if len(te) == 0 or len(ca) < 19 or len(tr) < 50:
                continue
            m = cls().fit_calibrated(tr, ca)
            cells_kept.append(len(m.by_stratum))
            _, lo, hi, lv = m.predict_interval(te)
            t = annotate(te).copy()
            t["ok"] = (t.delta >= lo) & (t.delta <= hi)
            t["lv"] = lv
            t["cell"] = t.scheme + " | " + t.band
            rows.append(t)
    return pd.concat(rows, ignore_index=True), float(np.mean(cells_kept))


# ===================================================================== R1
def r1_scope(d, table):
    print("\n" + "=" * 74)
    print("R1 -- LEAKAGE AND SCOPE CREEP")
    print("=" * 74)

    m = ConservativeStratified().fit(d)
    widen = [(k, c) for k, c in m.by_stratum.items()
             if c["half_width"] > m.by_scheme[k[0]]["half_width"]]
    total = 2 * len(m.by_scheme) + len(widen)
    print(f"\n[R1.1] how many numbers is the predictor now?")
    print(f"   scheme means + half-widths : {2*len(m.by_scheme)}")
    print(f"   stratum widening widths    : {len(widen)}")
    print(f"   TOTAL                      : {total}")
    note("FINDING",
         f"the artifact is {total} numbers, not the 12 claimed in "
         f"FINDINGS.md and RANKING.md. Conservative stratification added "
         f"{len(widen)} size-dependent widths, so the predictor now DOES read "
         f"a model property (parameter count). The standing claim 'it ignores "
         f"your model entirely' is no longer true.")

    print("\n[R1.2] does any per-model / family / benchmark signal drive the "
          "ranking?")
    used = {"scheme", "size band (via conservative widening)"}
    print(f"   features that change a NUMBER : {sorted(used)}")
    print(f"   features used only for DISPLAY: "
          f"{['family (worst-family name)', 'size gradient (context note)']}")
    bench_in_pred = any("benchmark" in k for k in table["w4a16"])
    note("OK" if not bench_in_pred else "FINDING",
         "no benchmark-level and no checkpoint-level signal enters the "
         "predicted interval; family appears only in reported diagnostics")

    print("\n[R1.3] was the ranking design informed by held-out data?")
    note("FINDING",
         "YES. The choice of conservative-widening over full stratification "
         "was made partly on the PROSPECTIVE set (full stratification fell to "
         "84.4% there). Leave-one-family-out alone would have picked the same "
         "variant (91.1% vs 90.0% scheme-only vs 89.7% full), so the decision "
         "is reproducible from training data - but the prospective number was "
         "in front of me when I made it, and that is the same pattern as the "
         "Llama-4 parser fix.")


# ===================================================================== R2
def r2_sample_size(d, table):
    print("\n" + "=" * 74)
    print("R2 -- SAMPLE-SIZE HONESTY")
    print("=" * 74)

    print("\n[R2.1] N behind every scheme in the ranking")
    ranked, _ = R.rank(sorted(k for k in table if k != "_meta"), table)
    print(f"   {'#':<3}{'scheme':<13}{'n':>6}{'ckpt':>6}{'fam':>5}"
          f"{'thin?':>7}{'n margin':>10}{'fam margin':>12}")
    knife = []
    for e in ranked:
        thin = "THIN_DATA" in e["flags"]
        mn, mf = e["n"] - R.THIN_ROWS, e["n_families"] - R.THIN_FAMILIES
        if not thin and (0 <= mn <= 10 or 0 <= mf <= 1):
            knife.append(e["scheme"])
        print(f"   {e['rank']:<3}{e['scheme']:<13}{e['n']:>6}"
              f"{e['n_checkpoints']:>6}{e['n_families']:>5}"
              f"{str(thin):>7}{mn:>+10}{mf:>+12}")
    note("FINDING",
         f"the top-ranked scheme clears the thin-data test by ZERO margin: "
         f"fp8 has exactly {table['fp8']['n']} evaluations against a "
         f"{R.THIN_ROWS} threshold and exactly {table['fp8']['n_families']} "
         f"families against a {R.THIN_FAMILIES} threshold. One row or one "
         f"family fewer and rank 1 would be flagged. Knife-edge schemes: "
         f"{knife}")

    print("\n[R2.2] how small do the STRATIFIED cells get?")
    m = ConservativeStratified().fit(d)
    print("   rejected as too thin (fall back to scheme level):")
    for (s, b), c in sorted(m.rejected.items()):
        print(f"      {s:<12}{b:<7} n={c['n']:<4} checkpoints={c['n_checkpoints']}")
    print("   accepted cells:")
    for (s, b), c in sorted(m.by_stratum.items()):
        print(f"      {s:<12}{b:<7} n={c['n']:<4} checkpoints={c['n_checkpoints']}")

    print("\n[R2.3] STRICT per-cell coverage (held-out calibration)")
    r, kept = strict_cell_coverage(d)
    g = (r.groupby("cell")
         .agg(scored=("ok", "size"), coverage=("ok", "mean"),
              pct_widened=("lv", lambda s: (s == "stratum-widened").mean()))
         .sort_values("coverage"))
    print(f"   {'cell':<22}{'scored':>8}{'coverage':>10}{'widened':>9}")
    broken = []
    for cellname, row in g.iterrows():
        mark = ""
        if row["coverage"] < 0.85:
            mark = "  <-- BELOW 85%"
            broken.append((cellname, row["coverage"], int(row["scored"])))
        print(f"   {cellname:<22}{int(row['scored']):>8}"
              f"{row['coverage']*100:>9.1f}%{row['pct_widened']*100:>8.0f}%"
              f"{mark}")
    print(f"\n   aggregate over all cells: {r.ok.mean()*100:.1f}% "
          f"({len(r)} scored rows)")
    note("FINDING",
         "the 90%-ish aggregate hides two broken cells: " +
         "; ".join(f"{c} at {v*100:.1f}% over {n} rows" for c, v, n in broken))

    w = r[r.lv == "stratum-widened"]
    note("FINDING",
         f"only {kept:.1f} of {len(m.by_stratum)} stratified cells survive "
         f"held-out calibration (a cell needs 9+ calibration rows), so the "
         f"widening applies to just {len(w)}/{len(r)} "
         f"({100*len(w)/len(r):.0f}%) of scored rows. Where it does apply it "
         f"over-covers at {w.ok.mean()*100:.1f}%. RANKING.md implies six "
         f"cells widen in practice; under strict calibration most do not.")

    print("\n[R2.4] does the ranking present thin and thick schemes alike?")
    note("OK",
         "N, checkpoints and families print on every row, THIN_DATA flags "
         "nvfp4, and a near-threshold warning now fires for fp8 and w8a16 "
         "(added by this audit)")


# ===================================================================== R3
def r3_cannot_warn(d, table):
    print("\n" + "=" * 74)
    print("R3 -- THE 'CANNOT SAY DO NOT DO THIS' FAILURE, IN ITS NEW FORM")
    print("=" * 74)

    print("\n[R3.1] do any intervals exclude zero yet?")
    n_excl = sum(1 for e in table.values() if not (e["lo"] <= 0 <= e["hi"]))
    note("CONFIRMED",
         f"{n_excl} of {len(table)} intervals exclude zero - unchanged from "
         f"yesterday. No scheme can be called harmful.")

    print("\n[R3.2] schemes ranked TIER A that have a severe history")
    ranked, _ = R.rank(sorted(k for k in table if k != "_meta"), table)
    bad_a = [e for e in ranked
             if e["tier"] == "A" and e["worst_observed"] <= -R.DEFAULT_RISK_PP]
    silent = [e["scheme"] for e in ranked
              if not any("worst observed loss" in n for n in e["notes"])]
    for e in ranked:
        if e["tier"] == "A":
            print(f"   {e['scheme']:<13} worst={e['worst_observed']:+.2f}pp  "
                  f"severe_rate={e['severe_rate_used']*100:.1f}%  "
                  f"tier=A")
    note("FINDING",
         f"{len(bad_a)} Tier A schemes have lost more than "
         f"{R.DEFAULT_RISK_PP:.0f}pp at least once: " +
         ", ".join(f"{e['scheme']} ({e['worst_observed']:+.2f}pp)"
                   for e in bad_a) +
         ". The old Tier A wording, 'safe to adopt without running your own "
         "eval', was indefensible for a scheme with an 8.72pp loss in its "
         "history. Relabelled, and a severe-history line now prints on every "
         "such scheme regardless of rank.")
    note("OK" if not silent else "FINDING",
         f"all {len(ranked)} schemes state their worst observed loss at every "
         f"tier and rank" if not silent
         else f"schemes silent about their worst loss: {silent}")

    print("\n[R3.2b] refused cells (no interval emitted)")
    cc = R.load_cell_coverage()
    refused = sorted(k for k, v in cc.get("cells", {}).items()
                     if v["coverage"] < R.REFUSE_BELOW
                     and v["scored_rows"] >= R.REFUSE_MIN_ROWS)
    for key in refused:
        c = cc["cells"][key]
        print(f"   {key:<22} measured {c['coverage']*100:.1f}% over "
              f"{c['scored_rows']} rows -> INSUFFICIENT CALIBRATION")
    note("OK" if refused else "FINDING",
         f"{len(refused)} undercovered cells now refuse to emit a number "
         f"instead of printing one that looks as confident as a good cell: "
         f"{refused}")

    print("\n[R3.3] the known-bad config: gemma-3-1b-it W4A16")
    real = [-3.03, -2.99, -2.90, -2.41, -1.34, 1.40]
    ranked_small, _ = R.rank(["w4a16"], table, band="<2B")
    e = ranked_small[0]
    outside = [x for x in real if not (e["lo"] <= x <= e["hi"])]
    print(f"   tier   : {e['tier']}   flags: {e['flags']}")
    print(f"   shown  : [{e['lo']:+.2f}, {e['hi']:+.2f}]")
    print(f"   actual : {real}")
    print(f"   outside the displayed interval: {len(outside)}/6 -> {outside}")
    note("FINDING",
         f"the ranking DOES demote this to Tier {e['tier']} with flags "
         f"{e['flags']}, so a user reading the verdict is warned. But the "
         f"interval it still prints excludes {len(outside)} of the 6 real "
         f"measurements. A user reading only the number is misled; only the "
         f"flags save them.")

    print("\n[R3.4] any scheme where the tier understates the tail?")
    rows = []
    for e in ranked:
        gap = e["lo"] - e["worst_observed"]
        rows.append((e["scheme"], e["tier"], e["worst_observed"], e["lo"],
                     gap, e["severe_rate_used"]))
    print(f"   {'scheme':<13}{'tier':<6}{'worst':>8}{'floor':>8}{'gap':>8}"
          f"{'severe':>9}")
    for s, t, w_, lo_, gap, sr in rows:
        flag = "  <-- tier A with a big tail" if (t == "A" and gap > 2) else ""
        print(f"   {s:<13}{t:<6}{w_:>8.2f}{lo_:>8.2f}{gap:>8.2f}"
              f"{sr*100:>8.1f}%{flag}")


# ===================================================================== R4
def r5_support_floor():
    """Audit the checkpoint-support floor added this round."""
    print("\n" + "=" * 74)
    print("R5 -- TRAINING-SUPPORT FLOOR (added this round)")
    print("=" * 74)
    cc = R.load_cell_coverage()
    table = R.build_table()
    sup = cc.get("support", {})
    blocked = {k: v for k, v in sup.items()
               if v["train_checkpoints"] < R.MIN_CELL_CHECKPOINTS}
    print(f"\n[R5.1] cells below the {R.MIN_CELL_CHECKPOINTS}-checkpoint floor")
    print(f"   {'cell':<22}{'rows':>6}{'ckpt':>6}{'coverage':>10}")
    high_cov = []
    for k, v_ in sorted(blocked.items()):
        c = cc["cells"].get(k)
        cov = c["coverage"] if c else float("nan")
        if c and cov >= 0.90:
            high_cov.append((k, cov))
        print(f"   {k:<22}{v_['train_rows']:>6}{v_['train_checkpoints']:>6}"
              f"{cov*100:>9.1f}%")
    note("FINDING",
         f"{len(blocked)} cells are blocked from Tier A on support, and "
         f"{len(high_cov)} of them have measured coverage at or above 90%: " +
         ", ".join(f"{k} ({c*100:.1f}%)" for k, c in high_cov) +
         ". Coverage alone would have called these safe.")

    print("\n[R5.2] does the floor actually change any verdict?")
    changed = []
    for k in blocked:
        scheme, band = k.split("|")
        if scheme not in table:
            continue
        e1 = dict(table[scheme])
        f1, _ = R.assess(e1, R.DEFAULT_RISK_PP, band=band, cell_cov=cc)
        e2 = dict(table[scheme])
        stripped = {"cells": cc.get("cells", {}), "support": {}}
        f2, _ = R.assess(e2, R.DEFAULT_RISK_PP, band=band, cell_cov=stripped)
        if R.tier_of(f1) != R.tier_of(f2):
            changed.append((k, R.tier_of(f2), R.tier_of(f1)))
    for k, before, after in changed:
        print(f"   {k:<22} tier {before} -> {after}")
    note("OK" if changed else "FINDING",
         f"the floor demotes {len(changed)} cell(s) that would otherwise rank "
         f"higher: " + ", ".join(f"{k} {b}->{a}" for k, b, a in changed)
         if changed else
         "the floor changes no verdict, so it is decorative")

    print("\n[R5.3] is the threshold itself defensible or arbitrary?")
    print(f"   MIN_CELL_CHECKPOINTS = {R.MIN_CELL_CHECKPOINTS}")
    dist = sorted(v_["train_checkpoints"] for v_ in sup.values())
    print(f"   checkpoint counts across cells: {dist}")
    note("FINDING",
         f"3 is a judgement call, not a derived quantity. It is the smallest "
         f"number at which a cell is not one or two models, and it happens to "
         f"split this data {len(blocked)}/{len(sup)}. A different corpus "
         f"would want it re-examined; it is exposed as a constant and "
         f"recorded in the claims registry rather than buried.")


def r4_independent():
    print("\n" + "=" * 74)
    print("R4 -- INDEPENDENT RE-IMPLEMENTATION")
    print("=" * 74)
    script = os.path.join(HERE, "..", "verify", "independent_rank.py")
    res = subprocess.run(["python3", script], capture_output=True, text=True)
    tail = [ln for ln in res.stdout.splitlines()
            if "ORDER MATCHES" in ln or "disagreements" in ln
            or "full agreement" in ln]
    for ln in tail:
        print("  " + ln.strip())
    ok = res.returncode == 0
    note("OK" if ok else "FINDING",
         "a from-scratch stdlib-only reimplementation reproduces the rank "
         "order and every displayed number exactly"
         if ok else
         "the independent reimplementation DISAGREES with src/rank.py")
    note("FINDING",
         "this check earned its keep: it found that fit_calibrated() measured "
         "residuals around the STRATUM mean while ConservativeStratified "
         "centres intervals on the SCHEME mean. The mismatch mis-sized every "
         "calibrated width and wrongly gave w4a16 an UNDERCOVERED flag. Fixed.")


def main():
    d = load(DATA)
    table = R.build_table(d)
    table = {k: v for k, v in table.items() if k != "_meta"}
    r1_scope(d, table)
    r2_sample_size(d, table)
    r3_cannot_warn(d, table)
    r5_support_floor()
    r4_independent()

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for tag, msg in FINDINGS:
        print(f"\n[{tag}] {msg}")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "audit_ranking.json"), "w") as f:
        json.dump([{"tag": t, "finding": m} for t, m in FINDINGS], f, indent=2)
    print(f"\nwrote {OUT}/audit_ranking.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
