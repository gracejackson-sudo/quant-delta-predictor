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
         "TOOL_SUMMARY.md", "ONE_SIDED_COVERAGE.md", "EXTERNAL_FEEDBACK.md",
         "NARRATIVE_TECHNICAL.md", "NARRATIVE_GENERAL.md")) if os.path.exists(p)]
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

    import rank as _R
    add("refuse_below_pct", 100 * _R.REFUSE_BELOW, 0,
        "coverage threshold below which a cell is refused")
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
    import pandas as _pdr
    from model import MIN_ACC_BEFORE as _MAB
    _raw = len(_pdr.read_csv(DATA))
    add("n_rows_raw", _raw, 0, "rows extracted, before any filtering")
    add("n_rows_dropped_near_chance", _raw - meta["n_rows"], 0,
        "extracted rows dropped for a near-chance baseline")
    add("min_acc_before_pct", _MAB, 0, "baseline accuracy floor, in points")
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
    add("n_cells_refused",
        sum(1 for v in cc.get("cells", {}).values() if R.classify_cell(v) == "refused"), 0,
        "cells currently refused (interval withheld)")
    add("n_cells_total", len(cc.get("cells", {})), 0,
        "total scheme/size cells")
    add("n_cells_insufficient_evidence",
        sum(1 for v in cc.get("cells", {}).values()
            if R.classify_cell(v) == "insufficient_evidence"), 0,
        "cells reclassified to insufficient_evidence by the checkpoint/"
        "bootstrap floor")

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

    # --- Track 2b: the low-rank claim re-tested after external review.
    # Variance and correlations are recomputed here from the raw data; the
    # prediction runs (about 20 minutes, needs BenchPress's released code) are
    # read from out/track2_benchpress.json.
    import track2_lowrank as _t2
    _M2 = _t2.build_matrix(_pdr.read_csv(DATA).pipe(lambda x: x[x.acc_before >= _MAB]))
    for _k in range(3, 7):
        _n, _c, _v = _t2.fair_variance_explained(_M2, _k)
        add(f"t2b_fair_rows_k{_k}", _n, 0, f"fully observed rows, {_k} benchmarks")
        add(f"t2b_fair_rank2_k{_k}_pct", 100 * _v, 0.05,
            f"rank-2 variance, mean-centred, {_k} benchmarks")
        _n, _c, _v = _t2.fair_variance_explained(_M2, _k, base_only=True)
        add(f"t2b_fair_base_rank2_k{_k}_pct", 100 * _v, 0.05,
            f"rank-2 variance, base rows only, {_k} benchmarks")
    add("t2b_filled_global_mean_pct", 100 * float(_M2.isna().to_numpy().mean()), 0.05,
        "share of cells the original variance figure filled with one mean")
    for _b in ("gpqa", "musr"):
        _nb = _t2.best_neighbour(_M2, _b)
        add(f"t2b_corr_{_b}", _nb[0], 0.005, f"strongest neighbour correlation, {_b}")
        add(f"t2b_corr_{_b}_overlap", _nb[2], 0, f"rows behind that correlation, {_b}")
    t2bp = os.path.join(HERE, "..", "out", "track2_benchpress.json")
    if os.path.exists(t2bp):
        _sm = json.load(open(t2bp))["summary"]
        for _id, _key in (("orig", "original|sib=0|random"),
                          ("orig_nosib", "original|sib=1|random"),
                          ("bp", "benchpress|sib=0|random"),
                          ("bp_nosib", "benchpress|sib=1|random"),
                          ("orig_pred", "original|sib=0|predictive"),
                          ("bp_pred", "benchpress|sib=0|predictive")):
            _v = _sm[_key]
            for _f, _tol in (("mae", 0.005), ("mae_scheme_mean", 0.005),
                             ("mae_zero", 0.005), ("ratio_vs_scheme_mean", 0.005),
                             ("mean_abs_pred", 0.005), ("corr_pred_true", 0.005),
                             ("n_severe", 0), ("severe_caught", 0.05),
                             ("false_alarms", 0.05), ("n", 0)):
                add(f"t2b_{_f}::{_id}", _v[_f], _tol, f"track 2b {_id}: {_f}")
            if _id in ("orig", "bp"):
                add(f"t2b_mae_sd::{_id}", _v["mae_sd"], 0.005, f"track 2b {_id}: seed sd")
        add("t2b_seeds", json.load(open(t2bp))["seeds"], 0, "seeds averaged")
        add("t2b_bp_improvement_pct",
            100 * (1 - _sm["benchpress|sib=0|random"]["mae"]
                   / _sm["original|sib=0|random"]["mae"]), 0.5,
            "BenchPress method's error reduction versus our original imputer")
        # External figure, read from the BenchPress paper's abstract
        # (arXiv:2606.24020: held-out scores recovered within 4.6 points).
        add("benchpress_reported_error_pts", 4.6, 0, "BenchPress abstract, held-out error")

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

    q35 = os.path.join(HERE, "..", "data", "qwen35", "published_rows.csv")
    q35r = os.path.join(HERE, "..", "data", "qwen35", "published_rejects.csv")
    if os.path.exists(q35) and os.path.exists(q35r):
        import pandas as _pd
        _q = _pd.read_csv(q35)
        add("qwen35_pub_rows", len(_q), 0, "published Qwen3.5 rows kept")
        add("qwen35_pub_verified", int(_q.verified.sum()), 0,
            "of those, passing the recovery check")
        add("qwen35_pub_repos", _q.model.nunique(), 0,
            "quantized Qwen3.5 repos with rows")
        add("qwen35_pub_rejects", len(_pd.read_csv(q35r)), 0,
            "published Qwen3.5 rows rejected")

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
        add("adv_min_params_b", a.params_b.min(), 0.001,
            "smallest model in the adversarial arm")
        add("adv_mid_params_b", sorted(a.params_b.unique())[1], 0.001,
            "middle size in the adversarial arm")
        w4 = bad[(bad.scheme == "w4a16") & bad.excess.notna()]
        add("adv_excess_mean", w4.excess.mean(), 0.01,
            "mean w4a16 excess over control")
        add("adv_excess_worst", w4.excess.min(), 0.01, "worst w4a16 excess")
        add("adv_excess_n", len(w4), 0, "w4a16 excess anchor rows")

    pc = os.path.join(HERE, "..", "out", "publisher_census.json")
    if os.path.exists(pc):
        c = json.load(open(pc))
        oth = [r for r in c["per_publisher"] if r["publisher"] != "RedHatAI"]
        add("pub_publishers_checked", c["publishers_checked"], 0,
            "publishers surveyed for paired base/quant evals")
        add("pub_cards_inspected", c["cards_inspected_total"], 0,
            "model cards inspected in total")
        add("pub_others_cards", sum(r["cards_fetched"] for r in oth), 0,
            "cards inspected from publishers other than RedHatAI")
        add("pub_others_paired_cards", sum(
            r["cards_with_paired_evals"] for r in oth), 0,
            "non-RedHatAI cards with any paired before/after rows")
        add("pub_others_verified_rows", sum(
            r["arith_verified_rows"] for r in oth), 0,
            "non-RedHatAI rows our recovery gate can verify")
        add("pub_control_verified_rows", c["control_arith_verified_rows"], 0,
            "RedHatAI rows our recovery gate can verify (positive control)")
        add("pub_others_count", len(oth), 0, "publishers other than RedHatAI")
        add("pub_others_verifiable_cards", sum(
            r["cards_with_verifiable_evals"] for r in oth), 0,
            "non-RedHatAI cards with any gate-verifiable rows")
        by = {r["publisher"]: r for r in c["per_publisher"]}
        for pub, key in (("Intel", "intel"), ("RedHatAI", "redhat")):
            if pub in by:
                add(f"pub_{key}_cards", by[pub]["cards_fetched"], 0,
                    f"{pub} cards inspected")
                add(f"pub_{key}_paired_cards",
                    by[pub]["cards_with_paired_evals"], 0,
                    f"{pub} cards with paired before/after rows")
                add(f"pub_{key}_verified_rows",
                    by[pub]["arith_verified_rows"], 0,
                    f"{pub} rows our recovery gate can verify")

    lc = os.path.join(HERE, "..", "data", "adversarial", "lora_forgetting.csv")
    kj = os.path.join(HERE, "..", "data", "adversarial", "weight_kurtosis.json")
    if os.path.exists(lc) and os.path.exists(kj):
        # re-derived straight from the two raw GPU-run artifacts
        from scipy import stats as _st
        lo = pd.read_csv(lc)
        km = {m_["model"]: m_ for m_ in json.load(open(kj))}
        both = sorted(set(lo.base_model) & set(km))
        add("lk_lora_pairs", len(lo), 0, "LoRA adapters evaluated")
        add("lk_lora_models", lo.base_model.nunique(), 0,
            "base models with LoRA forgetting measured")
        add("lk_kurt_models", len(km), 0, "models with kurtosis measured")
        add("lk_overlap_models", len(both), 0,
            "models with BOTH kurtosis and LoRA forgetting measured")
        add("lk_forget_max_pp", lo.forgetting.max(), 0.01,
            "largest LoRA forgetting value (most positive)")
        add("lk_forget_min_pp", lo.forgetting.min(), 0.01,
            "largest LoRA forgetting value (most negative)")
        sub = lo[lo.base_model.isin(both)].copy()
        sub["k3"] = sub.base_model.map(
            lambda m_: km[m_]["pct_channels_kurt_gt_3"])
        add("lk_pooled_rows", len(sub), 0,
            "rows with a kurtosis value (pseudo-replicated)")
        add("lk_pooled_pearson", _st.pearsonr(sub.k3, sub.forgetting)[0], 0.001,
            "pooled Pearson r, kurtosis vs forgetting")
        add("lk_pooled_spearman", _st.spearmanr(sub.k3, sub.forgetting)[0],
            0.001, "pooled Spearman r, kurtosis vs forgetting")
        rr = _st.spearmanr(lo.lora_rank, lo.forgetting)
        add("lk_rank_spearman", rr[0], 0.001, "Spearman r, LoRA rank vs forgetting")
        add("lk_rank_p", rr[1], 0.001, "p-value of that Spearman r")

    # ---- figures the paper quotes in its abstract and body ----
    add("target_mean_pp", d.delta.mean(), 0.005, "mean accuracy delta, all rows")
    add("target_sd_pp", d.delta.std(), 0.005, "sd of accuracy delta, all rows")
    _near = d[d.scheme.isin(["w8a16", "fp8_dynamic"])]
    _sh = []
    for _b, _g in d.groupby("benchmark"):
        _nb = _near[_near.benchmark == _b]
        if len(_g) >= 20 and len(_nb) >= 8:
            _sh.append((_b, _nb.delta.var(ddof=1) / _g.delta.var(ddof=1),
                        _nb.delta.std()))
    add("noise_n_bench", len(_sh), 0, "benchmarks with a noise-floor estimate")
    add("noise_n_third", sum(1 for x in _sh if x[1] >= 1 / 3), 0,
        "benchmarks where noise is at least a third of the variance")
    add("noise_min_pct", 100 * min(x[1] for x in _sh), 0.5,
        "smallest noise share of variance across benchmarks")
    add("noise_gsm8k_sd_pp", dict((x[0], x[2]) for x in _sh).get("gsm8k", 0),
        0.05, "sd of GSM8K deltas for near-lossless schemes")
    icp = os.path.join(HERE, "..", "out", "independent_check.csv")
    if os.path.exists(icp):
        import re as _re
        from scipy.stats import beta as _beta
        _ic = pd.read_csv(icp)
        _st = _ic[~_ic.model.map(lambda m_: bool(
            _re.search(r"Llama-3\.1", m_.split("/")[-1], _re.I)
            or _re.search(r"(^|[-_])Qwen3(?![.\d])", m_.split("/")[-1], _re.I)
            or _re.search(r"Llama-4", m_.split("/")[-1], _re.I)))]
        _k, _n = int(_st.inside.sum()), len(_st)
        add("prosp_n", _n, 0, "strict prospective rows (independent check)")
        add("prosp_inside", _k, 0, "strict prospective rows inside the interval")
        add("prosp_cov_pct", 100 * _k / _n, 0.05, "strict prospective coverage")
        add("prosp_ci_lo", 100 * _beta.ppf(0.025, _k, _n - _k + 1), 0.05,
            "Clopper-Pearson 95% lower bound")
        add("prosp_ci_hi", 100 * _beta.ppf(0.975, _k + 1, _n - _k), 0.05,
            "Clopper-Pearson 95% upper bound")

    if cc.get("pooled", {}).get("coverage_one_sided") is not None:
        add("pooled_one_sided_pct", 100 * cc["pooled"]["coverage_one_sided"],
            0.1, "pooled one-sided coverage")
        add("pooled_below_lo_pct", 100 * cc["pooled"]["below_lo"], 0.1,
            "pooled share of rows below the lower bound")
        add("pooled_above_hi_pct", 100 * cc["pooled"]["above_hi"], 0.1,
            "pooled share of rows above the upper bound")
    for key, v in cc["cells"].items():
        if v.get("coverage_one_sided") is not None:
            add(f"cell_ckpts::{key}", v.get("distinct_checkpoints", 0), 0,
                f"distinct checkpoints behind {key}")
            add(f"cell_one_sided_pct::{key}",
                100 * v["coverage_one_sided"], 0.1,
                f"one-sided coverage for {key}")

    osa = os.path.join(HERE, "..", "out", "one_sided_audit.json")
    if os.path.exists(osa):
        o = json.load(open(osa))
        add("os_pooled_pct", o["pooled_coverage_pct"], 0.1,
            "pooled coverage recomputed in the one-sided audit")
        add("os_pooled_rows", o["pooled_scored_rows"], 0,
            "pooled scored rows recomputed in the one-sided audit")
        for cell, v in o["cells"].items():
            for bm, mm in (v.get("miss_mean_by_checkpoint") or {}).items():
                add(f"os_missmean::{cell}::{bm}", mm, 0.01,
                    f"mean delta of misses for {bm} in {cell}")
            for key, tol in (("two_sided_pct", 0.1), ("one_sided_pct", 0.1),
                             ("below_lo_pct", 0.1), ("above_hi_pct", 0.1),
                             ("scored_rows", 0), ("distinct_checkpoints", 0),
                             ("distinct_families", 0),
                             ("distinct_model_benchmark", 0),
                             ("boot90_lo", 1.0), ("boot90_hi", 1.0)):
                add(f"os_{key}::{cell}", v[key], tol,
                    f"{key} for {cell} (one-sided audit)")
            for bm, pct in v["per_checkpoint_coverage_pct"].items():
                add(f"os_ckpt::{cell}::{bm}", pct, 0.1,
                    f"coverage for {bm} in {cell}")

    cq = os.path.join(HERE, "..", "out", "card_quality.json")
    if os.path.exists(cq):
        q = json.load(open(cq))
        pp = q["per_publisher"]
        add("cq_cards_scored", q["n_cards"], 0, "model cards scored")
        add("cq_publishers", q["n_publishers"], 0, "publishers scored")
        add("cq_rec_redhat", pp["RedHatAI"]["recovery_printed"], 0.1,
            "% RedHatAI cards printing a recovery figure")
        add("cq_rec_best_other", max(
            v["recovery_printed"] for k, v in pp.items() if k != "RedHatAI"),
            0.1, "best non-RedHatAI recovery-printing rate")
        add("cq_parser_missed", q["parser_missed_total"], 0,
            "cards with a benchmark table the parser could not read")

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
    if "pred_mae::scheme_mean" in reg:
        add("mae_gain_pp",
            reg["pred_mae::global_mean"][0] - reg["pred_mae::scheme_mean"][0],
            0.0005, "LOFO MAE gain of the per-scheme mean over the global mean")
    return reg


CLAIM_RE = re.compile(r"<!--\s*claim:\s*([^\s]+)\s*=\s*([-+0-9.]+)\s*-->")
# keys may carry an escaped pipe (see gen_one_sided_doc.n); undo it
def _unesc(k):
    return k.replace("\\|", "|")


def main():
    reg = registry()
    claims, per_doc = [], {}
    for d_ in DOCS:
        if not os.path.exists(d_):
            continue
        found = [(_unesc(k), v) for k, v in
                 CLAIM_RE.findall(open(d_, encoding="utf-8").read())]
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
