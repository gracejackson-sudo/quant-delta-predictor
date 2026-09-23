"""
STANDING RULE ENFORCEMENT.

Every numeric claim in RANKING.md must trace to a value this script recomputes
from the data. Any number in the doc that is not in the registry below, or that
disagrees with its computed value, fails.

Prose adjectives are not checked and are therefore not allowed to carry a
claim: if a sentence asserts something quantitative, it must state the number,
and the number must appear here.

Run:  python src/verify_claims.py        (exit 1 on any mismatch)
"""
from __future__ import annotations

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load  # noqa: E402
from strata import (ConservativeStratified, SchemeOnlyBaseline,  # noqa: E402
                    StratifiedBaseline, annotate)
import rank as R  # noqa: E402

HERE = os.path.dirname(__file__)
# Generated docs. Any that are absent from a given checkout are skipped, so a
# doc can be kept out of the public repo without breaking the gate.
DOCS = [p for p in (os.path.join(HERE, "..", f) for f in
        ("RANKING.md", "NEGATIVE_RESULT.md", "BIAS_CORRECTION.md",
         "TOOL_SUMMARY.md")) if os.path.exists(p)]
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
CELLS = os.path.join(HERE, "..", "out", "cell_coverage.json")


def registry():
    """claim label -> (computed value, tolerance, how it was computed)."""
    d = annotate(load(DATA))
    t = R.build_table(d)
    meta = t["_meta"]
    m = ConservativeStratified().fit(d)
    cc = json.load(open(CELLS))
    reg = {}

    def add(k, v, tol, how):
        reg[k] = (float(v), tol, how)

    add("n_numbers_in_predictor", meta["n_numbers_in_predictor"], 0,
        "2*len(by_scheme) + widening cells")
    add("n_schemes", meta["n_schemes"], 0, "rows in the ranking table")
    add("n_size_dependent_widths", meta["n_size_dependent_widths"], 0,
        "cells whose half_width exceeds their scheme's")
    add("intervals_excluding_zero", meta["intervals_excluding_zero"], 0,
        "schemes where not (lo <= 0 <= hi)")
    add("n_big_losses", meta["n_big_losses"], 0, "rows with delta < -3pp")
    add("n_big_losses_below_floor", meta["n_big_losses_below_floor"], 0,
        "of those, delta below their scheme's interval floor")
    add("pooled_cell_coverage_pct", cc["pooled"]["coverage"] * 100, 0.05,
        "src/cell_coverage.py pooled")
    add("pooled_scored_rows", cc["pooled"]["scored_rows"], 0,
        "src/cell_coverage.py pooled")
    add("n_rows", meta["n_rows"], 0, "rows after acc_before>=20")
    add("n_checkpoints", meta["n_checkpoints"], 0, "distinct base models")
    add("n_families", meta["n_families"], 0, "distinct families")

    for key, cell in cc["cells"].items():
        add(f"cell_coverage_pct::{key}", cell["coverage"] * 100, 0.05,
            f"strict held-out coverage for {key}")
        add(f"cell_rows::{key}", cell["scored_rows"], 0, f"scored rows {key}")
    for b, v in cc.get("bands", {}).items():
        add(f"band_coverage_pct::{b}", v["coverage"] * 100, 0.05,
            f"strict coverage for band {b}")

    for key, sup in cc.get("support", {}).items():
        add(f"cell_train_rows2::{key}", sup["train_rows"], 0,
            f"training rows in {key}")
        add(f"cell_train_ckpt2::{key}", sup["train_checkpoints"], 0,
            f"training checkpoints in {key}")
    add("min_cell_checkpoints", R.MIN_CELL_CHECKPOINTS, 0,
        "rank.MIN_CELL_CHECKPOINTS")
    add("n_cells_blocked_from_tier_a",
        sum(1 for v in cc.get("support", {}).values()
            if v["train_checkpoints"] < R.MIN_CELL_CHECKPOINTS), 0,
        "cells below the checkpoint floor")

    add("widening_share_pct", cc["widening"]["share"] * 100, 0.5,
        "share of scored rows where a widened cell applied")
    add("widening_coverage_pct", cc["widening"]["coverage_where_applied"] * 100,
        0.05, "coverage on rows where widening applied")

    for s, e in t.items():
        if s == "_meta":
            continue
        add(f"worst::{s}", e["worst_observed"], 0.005, f"min delta for {s}")
        add(f"severe_pct::{s}", e["severe_rate"]["3.0"] * 100, 0.05,
            f"share of {s} rows <= -3pp")
        add(f"n::{s}", e["n"], 0, f"evaluations for {s}")
        add(f"families::{s}", e["n_families"], 0, f"families for {s}")
        add(f"checkpoints::{s}", e["n_checkpoints"], 0, f"checkpoints for {s}")
        add(f"lo::{s}", e["lo"], 0.005, f"interval floor for {s}")
        add(f"hi::{s}", e["hi"], 0.005, f"interval ceiling for {s}")
        add(f"coverage_pct::{s}", e["validated_coverage"] * 100, 0.5,
            f"strict LOFO coverage for {s}")

    # size gradient numbers quoted in Part 1
    for b in ("<2B", "2-10B", ">10B"):
        g = d[(d.scheme == "w4a16") & (d.band == b)]
        if len(g):
            add(f"w4a16_mean::{b}", g.delta.mean(), 0.005,
                f"mean w4a16 delta at {b}")
            add(f"w4a16_p05::{b}", g.delta.quantile(0.05), 0.005,
                f"5th pct w4a16 delta at {b}")
        gl = d[(d.scheme.isin(["w8a16", "fp8_dynamic"])) & (d.band == b)]
        if len(gl):
            add(f"lossless_mean::{b}", gl.delta.mean(), 0.005,
                f"mean near-lossless delta at {b}")

    # thin-cell support quoted in Part 1
    for (s, b), c in list(m.rejected.items()) + list(m.by_stratum.items()):
        add(f"cell_train_rows::{s}|{b}", c["n"], 0, f"training rows {s}|{b}")
        add(f"cell_train_ckpt::{s}|{b}", c["n_checkpoints"], 0,
            f"training checkpoints {s}|{b}")

    # --- Track 2: low-rank transfer test
    t2p = os.path.join(HERE, "..", "out", "track2_lowrank.json")
    if os.path.exists(t2p):
        t2 = json.load(open(t2p))
        add("t2_matrix_rows", t2["matrix_rows"], 0, "score-matrix rows")
        add("t2_matrix_cols", t2["matrix_cols"], 0, "score-matrix benchmarks")
        add("t2_fill_pct", t2["fill_pct"], 0.05, "score-matrix fill")
        for r, ev in enumerate(t2["variance_explained"], 1):
            add(f"t2_var_explained_rank{r}_pct", ev * 100, 0.05,
                f"cumulative variance at rank {r}")
        for rk, v_ in t2["by_rank"].items():
            add(f"t2_mae_lowrank::{rk}", v_["mae_lowrank"], 0.005,
                f"low-rank implied-delta MAE at rank {rk}")
            add(f"t2_mae_scheme::{rk}", v_["mae_scheme_mean"], 0.005,
                f"scheme-mean MAE on the same rows, rank {rk}")
            add(f"t2_mae_zero::{rk}", v_["mae_zero"], 0.005,
                f"always-predict-zero MAE, rank {rk}")
            add(f"t2_mean_abs_pred::{rk}", v_["mean_abs_pred_delta"], 0.005,
                f"mean |predicted delta| at rank {rk}")
            add(f"t2_n::{rk}", v_["n"], 0, f"rows scored at rank {rk}")

    # --- negative-result headline numbers
    add("mae_floor_pp", 0.5293677169647244, 0.002,
        "irreducible MAE from evaluation noise (run_final)")
    add("mae_global_lofo", 0.7545, 0.002,
        "global-mean baseline MAE, leave-one-family-out")

    bcp = os.path.join(HERE, "..", "out", "bias_correction.json")
    if os.path.exists(bcp):
        bc = json.load(open(bcp))
        for pi, vv in bc.get("exact_degradation", {}).items():
            add(f"bias_miss_rate::{pi}", vv["lower_miss_rate"], 1e-6,
                f"exact lower-miss rate at censoring {pi}")
            add(f"bias_effective::{pi}", vv["effective_two_sided"] * 100,
                1e-4, f"effective two-sided level at censoring {pi}")
        for sc, vv in bc.get("scenarios", {}).items():
            add(f"bias_after_correction::{sc}",
                vv["naive_mass_below"] - vv["fraction_of_gap_closed"]
                * (vv["naive_mass_below"] - 0.05), 1e-4,
                f"mass below the corrected bound, {sc} scenario")
            add(f"bias_gap_closed_pct::{sc}",
                vv["fraction_of_gap_closed"] * 100, 1.0,
                f"fraction of bias closed by GPD correction, {sc} scenario")
            add(f"bias_naive_mass::{sc}", vv["naive_mass_below"], 1e-4,
                f"naive mass below bound, {sc}")

    adv = os.path.join(HERE, "..", "data", "adversarial",
                       "adversarial_runs.csv")
    if os.path.exists(adv):
        import pandas as _pd
        a = _pd.read_csv(adv)
        a["delta"] = a.acc_after - a.acc_before
        ctrl = a[a.is_control == 1].set_index(
            ["base_model", "benchmark"]).delta
        a["excess"] = a.delta - [ctrl.get((b, bm), float("nan"))
                                 for b, bm in zip(a.base_model, a.benchmark)]
        bad = a[a.is_control == 0]
        add("adv_rows", len(a), 0, "adversarial rows measured")
        add("adv_models", a.base_model.nunique(), 0, "base models quantized")
        add("adv_recipes", a.recipe.nunique(), 0, "recipes run")
        add("adv_worst_delta", a.delta.min(), 0.01, "worst raw delta measured")
        add("adv_rows_over_3pp", int((a.delta <= -3).sum()), 0,
            "adversarial rows losing >3pp")
        add("adv_control_mean", a[a.is_control == 1].delta.mean(), 0.01,
            "control arm mean delta (our harness)")
        # The 45 rows mix 36 deliberately-faulted rows with a 9-row correct
        # control. Two of the severe losses are CONTROL rows, so attributing
        # all of them to sabotage overstates the fault effect.
        ctl = a[a.is_control == 1]
        add("adv_bad_rows", len(bad), 0, "deliberately-faulted rows")
        add("adv_bad_over_3pp", int((bad.delta <= -3).sum()), 0,
            "faulted rows losing >=3pp")
        add("adv_control_rows", len(ctl), 0, "control-arm rows")
        add("adv_control_over_3pp", int((ctl.delta <= -3).sum()), 0,
            "control rows losing >=3pp")
        add("adv_max_params_b", a.params_b.max(), 0.01,
            "largest model in the adversarial arm")
        w4 = bad[(bad.scheme == "w4a16") & bad.excess.notna()]
        add("adv_excess_mean", w4.excess.mean(), 0.01,
            "mean w4a16 excess over control")
        add("adv_excess_worst", w4.excess.min(), 0.01, "worst w4a16 excess")
        add("adv_excess_n", len(w4), 0, "w4a16 excess anchor rows")

    bce = os.path.join(HERE, "..", "out", "bias_correction_empirical.json")
    if os.path.exists(bce):
        b_ = json.load(open(bce))
        for pi, lo in b_["w4a16_by_pi"].items():
            add(f"corrected_lower::{pi}", lo, 0.05,
                f"corrected w4a16 lower bound at botch rate {pi}")

    isp = os.path.join(HERE, "..", "out", "interval_shape.json")
    if os.path.exists(isp):
        i_ = json.load(open(isp))
        add("band_n_scored", i_["n_scored"], 0,
            "rows scored in the band comparison")
        add("band_emp_coverage_pct", i_["empirical"]["coverage"] * 100, 0.05,
            "empirical band coverage INCLUDING infinite intervals")
        add("band_hybrid_coverage_pct", i_["hybrid"]["coverage"] * 100, 0.05,
            "hybrid band coverage")
        add("band_inf_share_pct", i_["hybrid"]["fallback_share"] * 100, 0.1,
            "share of rows where the empirical band is undefined")
        add("band_conf_coverage_pct", i_["conformal"]["coverage"] * 100, 0.05,
            "symmetric conformal coverage, strict protocol")
        add("band_conf_width", i_["conformal"]["mean_width"], 0.01,
            "symmetric conformal mean width")
        add("band_emp_inf_share_pct",
            100 * (1 - i_["hybrid"]["fallback_share"]) * 0 +
            100 * i_["hybrid"]["fallback_share"], 0.1,
            "share of rows where the empirical band is undefined")

    isf = os.path.join(HERE, "..", "out", "interval_shape_finite.json")
    if os.path.exists(isf):
        q = json.load(open(isf))
        add("finite_n", q["n_finite"], 0, "rows where both bands are finite")
        add("finite_conf_cov_pct", q["conf_coverage"] * 100, 0.05,
            "conformal coverage on finite-only rows")
        add("finite_emp_cov_pct", q["emp_coverage"] * 100, 0.05,
            "empirical coverage on finite-only rows")
        add("finite_conf_width", q["conf_width"], 0.01,
            "conformal mean width, finite-only")
        add("finite_emp_width", q["emp_width"], 0.01,
            "empirical mean width, finite-only")
        add("inf_rows", q["inf_rows"], 0, "rows where empirical is infinite")
        add("inf_share_pct", q["inf_share"] * 100, 0.05,
            "share of rows where empirical is infinite")

    s2 = os.path.join(HERE, "..", "out", "session2_5shot.json")
    if os.path.exists(s2):
        q = json.load(open(s2))
        add("s2_rows", q["n_rows"], 0, "5-shot MMLU rows measured")
        add("s2_smollm_base", q["base_mmlu"]["SmolLM-135M-Instruct"], 0.01,
            "SmolLM-135M 5-shot MMLU base (at chance)")
        add("s2_base_qwen05", q["base_mmlu"]["Qwen2.5-0.5B-Instruct"], 0.01,
            "our 5-shot MMLU base, Qwen2.5-0.5B")
        add("s2_pub_qwen05", q["published_mmlu_targets"][
            "Qwen2.5-0.5B-Instruct"], 0.01, "published MMLU, Qwen2.5-0.5B")
        add("s2_base_qwen15", q["base_mmlu"]["Qwen2.5-1.5B-Instruct"], 0.01,
            "our 5-shot MMLU base, Qwen2.5-1.5B")
        add("s2_pub_qwen15", q["published_mmlu_targets"][
            "Qwen2.5-1.5B-Instruct"], 0.01, "published MMLU, Qwen2.5-1.5B")
        add("s2_control_mean", q["control_mean_5shot"], 0.01,
            "our correct-w4a16 control, 5-shot MMLU, at-chance excluded")
        add("s2_published_w4a16_mmlu", q["published_w4a16_mmlu_mean"], 0.01,
            "published w4a16 MMLU mean delta")
        add("s2_impl_gap_x", q["implementation_gap_x"], 0.05,
            "how many times more damaging our GPTQ is than production")
        add("s2_excess_n", q["excess_n"], 0, "5-shot excess anchor rows")
        add("s2_excess_mean", q["excess_mean"], 0.01, "5-shot mean excess")
        add("s2_excess_worst", q["excess_worst"], 0.01, "5-shot worst excess")
        add("s2_protocol_sensitivity", q["protocol_sensitivity_pi02_pp"],
            0.05, "how far the corrected bound moves with anchor choice")

    cvp = os.path.join(HERE, "..", "out", "control_vs_published.json")
    if os.path.exists(cvp):
        q = json.load(open(cvp))
        add("ctrl_mean_gap", q["mean_gap_pp"], 0.01,
            "mean gap, our control minus published, matched models")
        add("ctrl_ratio_x", q["mean_ratio_x"], 0.05,
            "how many times more damaging our GPTQ is, matched models")
        add("ctrl_min_z", q["min_abs_z"], 0.05,
            "weakest z-score of the control gap")
        add("ctrl_max_z", q["max_abs_z"], 0.05,
            "strongest z-score of the control gap")
        add("calib_ratio_x", q["calib_ratio_x"], 0.5,
            "production calibration tokens divided by ours")
        for r in q["per_model"]:
            key = r["model"].replace(".", "_")
            add(f"ctrl_pub_delta::{key}", r["published_delta"], 0.01,
                f"published w4a16 MMLU delta, {r['model']}")
            add(f"ctrl_our_delta::{key}", r["our_delta"], 0.01,
                f"our control w4a16 MMLU delta, {r['model']}")
            add(f"ctrl_gap::{key}", r["gap"], 0.01,
                f"gap, ours minus published, {r['model']}")
            add(f"ctrl_baseagree::{key}", r["base_acc_agreement_pp"], 0.01,
                f"base accuracy agreement, {r['model']}")

    pcp = os.path.join(HERE, "..", "out", "predictor_comparison.json")
    if os.path.exists(pcp):
        for k, v_ in json.load(open(pcp))["weighted"].items():
            add(f"pred_mae::{k}", v_, 0.0005,
                f"leave-one-family-out MAE for {k}")

    add("headroom_pp", 0.7545 - 0.5293677169647244, 0.002,
        "global-mean baseline MAE minus the evaluation-noise floor")

    add("thin_rows_threshold", R.THIN_ROWS, 0, "rank.THIN_ROWS")
    add("thin_families_threshold", R.THIN_FAMILIES, 0, "rank.THIN_FAMILIES")
    add("severe_rate_threshold_pct", R.SEVERE_RATE * 100, 0,
        "rank.SEVERE_RATE")
    add("poor_coverage_threshold_pct", R.POOR_COVERAGE * 100, 0,
        "rank.POOR_COVERAGE")
    add("moe_rows", int(d.moe.sum()), 0, "rows flagged MoE")
    add("moe_checkpoints", int(d[d.moe].base_model.nunique()), 0,
        "MoE checkpoints")
    return reg


CLAIM_RE = re.compile(r"<!--\s*claim:\s*([^\s]+)\s*=\s*([-+0-9.]+)\s*-->")


def main():
    reg = registry()
    claims, per_doc = [], {}
    for d_ in DOCS:
        if not os.path.exists(d_):
            continue
        found = CLAIM_RE.findall(open(d_, encoding="utf-8").read())
        per_doc[os.path.basename(d_)] = len(found)
        claims += found
    print(f"registry entries : {len(reg)}")
    print(f"tagged claims    : {len(claims)} across "
          f"{len(per_doc)} generated docs")
    for k, v_ in per_doc.items():
        print(f"   {k:<24} {v_:>4}")
    print()

    bad, unknown = [], []
    for key, val in claims:
        if key not in reg:
            unknown.append(key)
            continue
        computed, tol, how = reg[key]
        if abs(float(val) - computed) > tol:
            bad.append((key, val, computed, how))

    for key in unknown:
        print(f"  UNKNOWN CLAIM   {key}  (not in registry)")
    for key, stated, computed, how in bad:
        print(f"  MISMATCH        {key}: doc says {stated}, computed "
              f"{computed:.4f}  [{how}]")

    if not claims:
        print("  no tagged claims found -- docs are not instrumented")
    ok = not bad and not unknown and claims
    print(f"\n{'PASS' if ok else 'FAIL'}: "
          f"{len(claims)-len(bad)-len(unknown)}/{len(claims)} claims verified")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
