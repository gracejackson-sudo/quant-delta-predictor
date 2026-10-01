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
# Docs whose tagged figures this gate verifies. Any that are absent from a
# given checkout are skipped, so a doc can be kept out of the public repo
# without breaking the gate.
#
# README.md and ACCOUNTING.md are hand-written, not generated, and were
# outside this list until 2026-10-01. That was a real hole and not a
# deliberate exemption: audit_traceability.py already required every numeral
# in README.md to carry a claim tag or an explicit exemption, so its figures
# looked verified, but nothing ever compared a tagged value against the
# computed one. README.md's raw row count sat two corpus changes stale behind
# its own tag, and ACCOUNTING.md's funnel -- which carried no tags at all --
# sat stale behind the paper. Traceability asks whether a number is sourced;
# this asks whether it is right. They are different questions and the
# most-read file in the repository was only answering the first.
DOCS = [p for p in (os.path.join(HERE, "..", f) for f in
        ("RANKING.md", "NEGATIVE_RESULT.md", "BIAS_CORRECTION.md",
         "TOOL_SUMMARY.md", "ONE_SIDED_COVERAGE.md", "EXTERNAL_FEEDBACK.md",
         "NARRATIVE_TECHNICAL.md", "NARRATIVE_GENERAL.md",
         "README.md", "ACCOUNTING.md")) if os.path.exists(p)]
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
    add("pooled_scored_pairs", cc["pooled"]["scored_pairs"], 0,
        "row-by-calibration-family evaluations (NOT independent rows)")
    add("pooled_distinct_rows", cc["pooled"]["distinct_rows"], 0,
        "distinct rows behind the pooled coverage")
    add("pooled_pairs_per_row", cc["pooled"]["pairs_per_row"], 0.001,
        "calibration families each row is scored under")
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
    add("n_benchmarks", int(d.benchmark.nunique()), 0,
        "distinct benchmarks in the modeling set (Tier 2.5 v4 split "
        "the plain 'gpqa' label into five card-verified protocol "
        "labels, so the count grew from 16 to 20)")
    # Tier 2.5 v4 check #2 (day-7 audit): §9 says "up to N benchmark
    # rows come from a single quantization run", where N used to be
    # the hand-typed "sixteen". Register the max live so a future
    # card that carries more benchmarks flows through, and pin a test
    # that fails if the max moves without §9 being updated.
    # Per-fold MAE spread for the §5 "how LOFO looks fold-by-fold"
    # paragraph. Registered live so the paragraph's fold-min /
    # fold-max / winners-vs-losers claims are traceable and cannot
    # drift silently as the corpus evolves.
    from model import Predictor as _Pred_pf
    import numpy as _np_pf
    _pf = []
    for _fam in sorted(d.family.unique()):
        _tr = d[d.family != _fam]; _te = d[d.family == _fam]
        _y = _te.delta.to_numpy(float)
        _pg = _tr.delta.mean()
        _sm = _tr.groupby("scheme").delta.mean()
        _psc = _te.scheme.map(_sm).fillna(_pg).to_numpy(float)
        _mrid = _Pred_pf(alpha=3.0).fit(_tr)
        _pri = _mrid.predict(_te)
        _pf.append((_fam, len(_te),
                    float(_np_pf.mean(_np_pf.abs(_y - _pg))),
                    float(_np_pf.mean(_np_pf.abs(_y - _psc))),
                    float(_np_pf.mean(_np_pf.abs(_y - _pri)))))
    add("lofo_scheme_mean_wins_folds",
        int(sum(1 for _, _n, _g, _s, _ in _pf if _s < _g)), 0,
        "held-out folds where per-scheme mean beats global mean")
    add("lofo_scheme_mean_loses_folds",
        int(sum(1 for _, _n, _g, _s, _ in _pf if _s >= _g)), 0,
        "held-out folds where per-scheme mean does not beat global")
    _loss_fold = [f for f, _n, _g, _s, _ in _pf if _s >= _g]
    add("lofo_scheme_mean_loss_fold_min",
        min(_s for _f, _n, _g, _s, _ in _pf if _f in _loss_fold) if _loss_fold else float("nan"), 0.005,
        "scheme-mean MAE on the loss fold (max, if multiple)")
    add("lofo_scheme_mean_min",
        min(_s for _, _n, _, _s, _ in _pf), 0.005,
        "smallest per-fold scheme-mean MAE (best fold)")
    add("lofo_scheme_mean_max",
        max(_s for _, _n, _, _s, _ in _pf), 0.005,
        "largest per-fold scheme-mean MAE (worst fold)")
    add("lofo_ridge_min",
        min(_r for _, _n, _, _, _r in _pf), 0.005,
        "smallest per-fold ridge MAE (best fold)")
    add("lofo_ridge_max",
        max(_r for _, _n, _, _, _r in _pf), 0.005,
        "largest per-fold ridge MAE (worst fold)")
    _best_fam = min(_pf, key=lambda x: x[3])[0]
    _worst_fam = max(_pf, key=lambda x: x[3])[0]
    add("lofo_best_fold_size_pct", 100 * min(_pf, key=lambda x: x[3])[1] / len(d), 0.05,
        f"training-set share of the best fold ({_best_fam})")
    add("lofo_worst_fold_size_pct", 100 * max(_pf, key=lambda x: x[3])[1] / len(d), 0.05,
        f"training-set share of the worst fold ({_worst_fam})")
    add("llama3_fold_rows",
        int((d.family == "llama-3").sum()), 0,
        "rows in the merged Llama-3 fold")
    add("llama3_fold_share_pct",
        100 * int((d.family == "llama-3").sum()) / len(d), 0.05,
        "Llama-3 fold share of the modeling corpus")
    add("llama3_holdout_train_share_pct",
        100 * (1 - int((d.family == "llama-3").sum()) / len(d)), 0.05,
        "training-set share when Llama-3 is held out")
    _other_train_shares = [100 * (1 - _n / len(d))
                           for _f, _n, _g, _s, _ in _pf if _f != "llama-3"]
    add("nonllama3_holdout_train_min_pct",
        min(_other_train_shares), 0.05,
        "smallest training-set share when a non-Llama-3 family is held out")
    add("nonllama3_holdout_train_max_pct",
        max(_other_train_shares), 0.05,
        "largest training-set share when a non-Llama-3 family is held out")
    _other_pf = [(_f, _n) for _f, _n, *_ in _pf if _f != "llama-3"]
    add("gemma2_fold_rows",
        int(next(_n for _f, _n in _other_pf if _f == "gemma-2")), 0,
        "rows in the gemma-2 fold")
    add("qwen3_fold_rows",
        int(next(_n for _f, _n in _other_pf if _f == "qwen3")), 0,
        "rows in the qwen3 fold")
    add("nonllama3_fold_rows_min",
        min(_n for _, _n in _other_pf), 0,
        "smallest non-Llama-3 fold size in rows")
    add("nonllama3_fold_rows_max",
        max(_n for _, _n in _other_pf), 0,
        "largest non-Llama-3 fold size in rows")
    add("nonllama3_fold_share_min_pct",
        100 * min(_n for _, _n in _other_pf) / len(d), 0.05,
        "smallest non-Llama-3 fold share of the corpus")
    add("nonllama3_fold_share_max_pct",
        100 * max(_n for _, _n in _other_pf) / len(d), 0.05,
        "largest non-Llama-3 fold share of the corpus")

    add("max_benchmarks_per_run",
        int(d.groupby(["base_model", "scheme"]).benchmark.nunique().max()),
        0, "maximum distinct benchmarks reported for a single "
           "(base_model, scheme) quantization run")

    for key, cell in cc["cells"].items():
        add(f"cell_coverage_pct::{key}", cell["coverage"] * 100, 0.05,
            f"strict held-out coverage for {key}")
        add(f"cell_rows::{key}", cell["distinct_rows"], 0, f"distinct rows {key}")
        add(f"cell_pairs::{key}", cell["scored_pairs"], 0,
            f"row-by-calibration-family evaluations {key}")
        add(f"cell_boot_lo::{key}", cell["boot90_lo"], 1.0, f"bootstrap 5th percentile, {key}")
        add(f"cell_boot_hi::{key}", cell["boot90_hi"], 1.0, f"bootstrap 95th percentile, {key}")
    for b, v in cc.get("bands", {}).items():
        add(f"band_coverage_pct::{b}", v["coverage"] * 100, 0.05,
            f"strict coverage for band {b}")
    # Marginal (no-Mondrian) LOFO coverage for the NVFP4 scheme, live from
    # the stored final_marginal_C_leave_family_out.csv. This is what the paper
    # cites when it says a marginal 90% interval undercovered NVFP4 users.
    import csv as _csv
    _mcsv = os.path.join(HERE, "..", "out",
                         "final_marginal_C_leave_family_out.csv")
    if os.path.exists(_mcsv):
        _c = {"covered": 0, "n": 0}
        for _r in _csv.DictReader(open(_mcsv)):
            if _r["scheme"] != "nvfp4":
                continue
            _c["n"] += 1
            _c["covered"] += 1 if _r["covered"] == "True" else 0
        if _c["n"]:
            add("nvfp_marginal_coverage_pct",
                100 * _c["covered"] / _c["n"], 0.05,
                "marginal (no-Mondrian) LOFO coverage on the NVFP4 slice")

    # Failed task-1 calibration-data test (day-6). The two quantized-model
    # MMLU accuracies from the anomalous first run, kept as the range the
    # paper's Limitations paragraph cites.
    _t1 = os.path.join(HERE, "..", "out", "task1_real_gptq.json")
    if os.path.exists(_t1):
        _t = json.load(open(_t1))
        _accs = sorted(r["acc_after"] for r in _t)
        if len(_accs) >= 2:
            add("t1_anom_min_pct", _accs[0], 0.02,
                "day-6 task-1 lowest quantized MMLU on the anomalous first run")
            add("t1_anom_max_pct", _accs[-1], 0.02,
                "day-6 task-1 highest quantized MMLU on the anomalous first run")
        # Per-model before/after so EXTERNAL_FEEDBACK.md can name them
        # without hand-typing (Step 3b, day-7 audit).
        for _row in _t:
            _slug = _row["model"].replace("Qwen2.5-", "qwen25_").replace("-Instruct", "").lower().replace(".", "_")
            add(f"t1_before_{_slug}", _row["acc_before"], 0.02,
                f"day-6 task-1 acc_before for {_row['model']}")
            add(f"t1_after_{_slug}", _row["acc_after"], 0.02,
                f"day-6 task-1 acc_after for {_row['model']}")

    # RedHatAI-card recipe-documentation scan (see src/rh_card_recipe_scan.py).
    # Keys carry the scan date because the underlying count of published
    # cards will drift; a future re-scan should mint a new set of keys
    # instead of silently overwriting the number the paper committed to.
    _rh = os.path.join(HERE, "..", "out", "rh_card_scan_2026_09_26.json")
    if os.path.exists(_rh):
        _r = json.load(open(_rh))
        _a, _h = _r["cards_scanned"], _r["harvested_subset"]
        add("n_rh_cards_listed_2026_09_26", _a["n"], 0,
            "RedHatAI quantized cards with a non-empty README on 2026-09-26")
        add("n_rh_cards_harvested", _h["n"], 0,
            "of those, cards contributing rows to data/dataset.csv")
        add("n_rh_cards_with_lib_version_2026_09_26", _a["lib_version"], 0,
            "cards pinning a quantization-library version (all 439)")
        add("n_rh_cards_with_act_order_2026_09_26", _a["act_order"], 0,
            "cards mentioning activation reordering (all 439)")
        add("n_rh_cards_with_damp_2026_09_26", _a["damp"], 0,
            "cards mentioning dampening / damp_frac (all 439)")
        add("n_rh_cards_no_recipe_2026_09_26", _a["no_recipe"], 0,
            "cards documenting NONE of {num_calib, max_seq, act_order, damp}")
        add("n_rh_harvested_with_lib_version_2026_09_26", _h["lib_version"], 0,
            "harvested subset: cards pinning a library version")
        add("n_rh_harvested_no_recipe_2026_09_26", _h["no_recipe"], 0,
            "harvested subset: cards with none of the four recipe fields")

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
        "share of evaluations where a widened cell applied")
    # scheme-level coverage: same measurement, same classifier, as the cells
    _sch = cc.get("schemes", {})
    for s_, v_ in _sch.items():
        add(f"scheme_rows::{s_}", v_["distinct_rows"], 0, f"distinct rows behind {s_} coverage")
        add(f"scheme_pairs::{s_}", v_["scored_pairs"], 0, f"evaluations behind {s_} coverage")
        add(f"scheme_ckpts::{s_}", v_["distinct_checkpoints"], 0, f"checkpoints behind {s_} coverage")
        add(f"scheme_cov_one::{s_}", v_["coverage_one_sided"] * 100, 0.05, f"one-sided coverage, {s_}")
        add(f"scheme_cov_two::{s_}", v_["coverage"] * 100, 0.05, f"two-sided coverage, {s_}")
        add(f"scheme_boot_lo::{s_}", v_["boot90_lo"], 0.3, f"bootstrap 5th percentile, one-sided, {s_}")
        add(f"scheme_boot_hi::{s_}", v_["boot90_hi"], 0.3, f"bootstrap 95th percentile, one-sided, {s_}")
    add("n_schemes_insufficient_evidence",
        sum(1 for v_ in _sch.values() if R.classify_cell(v_) == "insufficient_evidence"), 0,
        "schemes whose all-sizes coverage is insufficient evidence")
    add("n_schemes_refused",
        sum(1 for v_ in _sch.values() if R.classify_cell(v_) == "refused"), 0,
        "schemes whose all-sizes coverage is refused")
    add("widening_coverage_pct", cc["widening"]["coverage_where_applied"] * 100,
        0.05, "coverage on rows where widening applied")

    for s, e in t.items():
        if s == "_meta":
            continue
        add(f"worst::{s}", e["worst_observed"], 0.005, f"min delta for {s}")
        add(f"severe_pct::{s}", e["severe_rate"]["3.0"] * 100, 0.05,
            f"share of {s} rows losing more than 3pp (strict, delta < -3)")
        add(f"n::{s}", e["n"], 0, f"evaluations for {s}")
        add(f"families::{s}", e["n_families"], 0, f"families for {s}")
        add(f"checkpoints::{s}", e["n_checkpoints"], 0, f"checkpoints for {s}")
        add(f"lo::{s}", e["lo"], 0.005, f"interval floor for {s}")
        add(f"hi::{s}", e["hi"], 0.005, f"interval ceiling for {s}")
        add(f"coverage_pct::{s}", e["coverage_one_sided"] * 100, 0.5,
            f"strict LOFO one-sided coverage for {s}")

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
    # Tier 2.5 v4 (day-7 audit): the plain 'gpqa' column no longer
    # exists in the corpus (split into gpqa_main / gpqa_main_norm /
    # gpqa_main_cot_5shot / gpqa_diamond / gpqa_diamond_cot_5shot per
    # the RedHatAI cards). Use gpqa_main as the representative here --
    # it is the largest of the five (15 rows) and matches the label the
    # external commenter's remark was about; the other four are
    # separately available under their own names.
    for _b in ("gpqa_main", "musr"):
        if _b in _M2.columns:
            _nb = _t2.best_neighbour(_M2, _b)
            add(f"t2b_corr_{_b}", _nb[0], 0.005,
                f"strongest neighbour correlation, {_b}")
            add(f"t2b_corr_{_b}_overlap", _nb[2], 0,
                f"rows behind that correlation, {_b}")
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

    # --- negative-result headline numbers.
    # Tier 2.7 (day-7 audit): these were previously hard-coded literals
    # from a specific run_final run. A data change (Def B, gpqa dedup,
    # Mixtral rebinning) left the values still "verified" but silently
    # stale. Both now compute live from the current corpus. The floor
    # excludes arena_hard, gpqa and musr because those benchmarks have
    # fewer than 4 lossless-scheme rows; the exclusion is a property of
    # the current data, not of the method, so we record which benchmarks
    # were included so the exclusion is visible.
    from run_final import mae_floor as _mae_floor_fn
    _floor, _floor_tab = _mae_floor_fn(d)
    add("mae_floor_pp", _floor, 0.002,
        "irreducible MAE from evaluation noise (mean absolute deviation "
        "on lossless-scheme rows, pooled by row count; computed live)")
    add("mae_floor_n_benchmarks", int(len(_floor_tab)), 0,
        "benchmarks included in the MAE floor (needs 4+ lossless rows)")
    add("mae_floor_excluded_benchmarks",
        int(d.benchmark.nunique() - len(_floor_tab)), 0,
        "benchmarks EXCLUDED from the MAE floor (fewer than 4 lossless "
        "rows -- currently arena_hard, gpqa and musr)")

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
        add("adv_rows_over_3pp", int((a.delta < -3).sum()), 0,
            "adversarial rows losing more than 3pp (strict; Tier 1.2 fix)")
        add("adv_control_mean", a[a.is_control == 1].delta.mean(), 0.01,
            "control arm mean delta (our harness)")
        # The 45 rows mix 36 deliberately-faulted rows with a 9-row correct
        # control. Two of the severe losses are CONTROL rows, so attributing
        # all of them to sabotage overstates the fault effect.
        ctl = a[a.is_control == 1]
        add("adv_bad_rows", len(bad), 0, "deliberately-faulted rows")
        add("adv_bad_over_3pp", int((bad.delta < -3).sum()), 0,
            "faulted rows losing more than 3pp (strict; Tier 1.2 fix)")
        add("adv_control_rows", len(ctl), 0, "control-arm rows")
        add("adv_control_over_3pp", int((ctl.delta < -3).sum()), 0,
            "control rows losing more than 3pp (strict; Tier 1.2 fix)")
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
    add("noise_max_pct", 100 * max(x[1] for x in _sh), 0.5,
        "largest noise share of variance across benchmarks")
    add("noise_n_ge100", sum(1 for x in _sh if x[1] >= 1), 0,
        "benchmarks whose noise estimate is at or above the total variance")
    add("noise_n_small", sum(1 for _b_, _g_ in d.groupby("benchmark")
                              if len(_g_) >= 20 and 8 <= len(_near[_near.benchmark == _b_]) <= 10),
        0, "benchmarks whose noise estimate rests on 10 or fewer near-lossless rows")
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
        # Composition of the strict prospective set, and a cluster-aware interval.
        import numpy as _np
        add("prosp_n_schemes", _st.scheme.nunique(), 0,
            "schemes with any strict prospective row")
        add("prosp_all_rows", len(_ic), 0, "all prospective rows, before the strict filter")
        add("prosp_all_cov_pct", 100 * _ic.inside.mean(), 0.05,
            "coverage over all prospective rows")
        for _s, _g in _st.groupby("scheme"):
            add(f"prosp_scheme_rows::{_s}", len(_g), 0, f"strict prospective rows, {_s}")
            add(f"prosp_scheme_ckpts::{_s}", _g.model.nunique(), 0,
                f"quantized checkpoints behind those rows, {_s}")
            add(f"prosp_scheme_cov_pct::{_s}", 100 * _g.inside.mean(), 0.05,
                f"strict prospective coverage, {_s}")
        # W4A16 share of the strict prospective rows, for the Simpson-effect
        # disclosure on the pooled coverage figure in the Results section.
        _wfour = _st[_st.scheme == "w4a16"]
        add("prosp_wfour_share_pct", 100 * len(_wfour) / max(len(_st), 1), 0.5,
            "W4A16 share of the strict prospective set")
        # Family split under the mechanistic definition (Section 3): a
        # checkpoint is in-family if its pretraining weights are inherited
        # from a training family (distillation, continued pretraining,
        # quantization, fine-tuning). On this corpus the in-family label
        # applies exactly to the seven DeepSeek-R1-Distill checkpoints, whose
        # bases are Llama-3.1/3.3 and Qwen-2.5 checkpoints in the training
        # data. Every other strict checkpoint (Gemma-3, SmolLM, SmolLM3,
        # Nemotron-Nano) is a new pretraining run relative to the training
        # families and is treated as out-of-family.
        _in_mask = _st.model.map(lambda m_: bool(
            _re.search(r"DeepSeek-R1-Distill-(?:Llama|Qwen)",
                       m_.split("/")[-1], _re.I)))
        _in = _st[_in_mask]; _out = _st[~_in_mask]
        add("prosp_in_family_rows", len(_in), 0,
            "rows in the strict prospective set that are in-family under the "
            "Section 3 definition")
        add("prosp_in_family_inside", int(_in.inside.sum()), 0,
            "of those rows, how many the shipped interval covered")
        add("prosp_in_family_cov_pct", 100 * _in.inside.mean(), 0.05,
            "in-family two-sided coverage on the strict prospective set")
        add("prosp_in_family_ckpts", _in.model.nunique(), 0,
            "in-family checkpoints in the strict prospective set")
        add("prosp_out_family_rows", len(_out), 0,
            "rows in the strict prospective set that are out-of-family")
        add("prosp_out_family_inside", int(_out.inside.sum()), 0,
            "of those rows, how many the shipped interval covered")
        add("prosp_out_family_cov_pct", 100 * _out.inside.mean(), 0.05,
            "out-of-family two-sided coverage on the strict prospective set")
        add("prosp_out_family_ckpts", _out.model.nunique(), 0,
            "out-of-family checkpoints in the strict prospective set")
        # Gemma-3 as its own sub-group of the out-of-family split, because
        # SmolLM/SmolLM3 and Nemotron-Nano contribute only 10 and 1 rows
        # respectively and pull the aggregate up. Report Gemma-3 on its own.
        _gm3 = _out[_out.model.map(lambda m_: bool(
            _re.search(r"gemma-3", m_.split("/")[-1], _re.I)))]
        add("prosp_gemma3_rows", len(_gm3), 0, "strict prospective Gemma-3 rows")
        add("prosp_gemma3_inside", int(_gm3.inside.sum()), 0,
            "of those Gemma-3 rows, how many the shipped interval covered")
        add("prosp_gemma3_cov_pct", 100 * _gm3.inside.mean(), 0.05,
            "Gemma-3 two-sided coverage on the strict prospective set")
        add("prosp_gemma3_ckpts", _gm3.model.nunique(), 0,
            "Gemma-3 checkpoints in the strict prospective set")
        add("prosp_gemma3_gap_pp", 90.0 - 100 * _gm3.inside.mean(), 0.05,
            "how many pp below the nominal 90% Gemma-3 lands (paper: "
            "the largest out-of-family sub-group)")
        # The remaining tiny out-of-family sub-groups (SmolLM* and Nemotron)
        # contribute rows at 100% on small samples. Report jointly.
        _small = _out[~_out.index.isin(_gm3.index)]
        add("prosp_outfam_small_rows", len(_small), 0,
            "out-of-family rows outside Gemma-3 (SmolLM/SmolLM3 + Nemotron-Nano)")
        add("prosp_outfam_small_ckpts", _small.model.nunique(), 0,
            "corresponding checkpoint count")
        add("prosp_outfam_small_pullup_pp",
            100 * _out.inside.mean() - 100 * _gm3.inside.mean(), 0.05,
            "how many pp the small-sample groups raise the out-of-family "
            "aggregate above Gemma-3 alone")
        # Cluster-bootstrap 95% CI on the in-family minus out-of-family
        # difference. Resamples whole checkpoints separately from each side.
        import numpy as _npf
        _rng2 = _npf.random.default_rng(0)
        _in_ks = _in.model.unique().tolist()
        _out_ks = _out.model.unique().tolist()
        _in_g = {m: _in[_in.model == m].inside.to_numpy(float) for m in _in_ks}
        _out_g = {m: _out[_out.model == m].inside.to_numpy(float) for m in _out_ks}
        _diffs = []
        for _ in range(20000):
            _si = _rng2.choice(len(_in_ks), size=len(_in_ks), replace=True)
            _so = _rng2.choice(len(_out_ks), size=len(_out_ks), replace=True)
            _ki = sum(_in_g[_in_ks[_i]].sum() for _i in _si)
            _ni = sum(len(_in_g[_in_ks[_i]]) for _i in _si)
            _ko = sum(_out_g[_out_ks[_i]].sum() for _i in _so)
            _no = sum(len(_out_g[_out_ks[_i]]) for _i in _so)
            if _ni and _no:
                _diffs.append(_ki / _ni - _ko / _no)
        _diffs = _npf.array(_diffs)
        add("prosp_infam_outfam_diff_lo",
            100 * _npf.percentile(_diffs, 2.5), 0.5,
            "cluster-bootstrap 95% lower bound on the in-family minus "
            "out-of-family coverage difference")
        add("prosp_infam_outfam_diff_hi",
            100 * _npf.percentile(_diffs, 97.5), 0.5,
            "cluster-bootstrap 95% upper bound on that difference")
        _cl = [_x.inside.to_numpy(float) for _, _x in _st.groupby("model")]
        _S = _np.array([_c.sum() for _c in _cl]); _N = _np.array([len(_c) for _c in _cl])
        _rng = _np.random.default_rng(0)
        _bs = [_S[_i].sum() / _N[_i].sum() for _i in
               (_rng.integers(0, len(_cl), len(_cl)) for _ in range(10000))]
        add("prosp_clus_lo", 100 * _np.percentile(_bs, 2.5), 0.3,
            "cluster bootstrap 95% lower bound, resampling quantized checkpoints")
        add("prosp_clus_hi", 100 * _np.percentile(_bs, 97.5), 0.3,
            "cluster bootstrap 95% upper bound, resampling quantized checkpoints")
        add("prosp_clus_n", len(_cl), 0, "quantized checkpoints resampled")

        # --- Round item 2: the checkpoint-level headline.
        #
        # The shipped headline was an exact Clopper-Pearson interval on rows.
        # Two reviewers objected that the rows are not independent: 131 rows
        # come from 19 quantization runs, up to 13 from one. The honest unit
        # is the checkpoint, so the headline interval is now the standard
        # cluster-robust (sandwich) one for a ratio estimator under cluster
        # sampling. The row-level Clopper-Pearson keys above are kept and
        # still computed, so the two can be compared and the old number
        # reproduced.
        #
        # This interval comes out NARROWER than Clopper-Pearson, which looks
        # like interval-shopping until you measure why, so the reason is
        # registered beside it rather than asserted in prose: the coverage
        # indicator is not clustered by checkpoint. Everything below is
        # deterministic -- no seed, no resampling -- precisely because a
        # published interval should not depend on a random stream.
        _p = _k / _n
        _n_cl = len(_cl)
        # Cluster-robust SE of the ratio estimator sum(S_i)/sum(N_i).
        _num = sum((_S[_j] - _p * _N[_j]) ** 2 for _j in range(_n_cl))
        _se_cr = (_n_cl / (_n_cl - 1) * _num) ** 0.5 / _n
        _se_naive = (_p * (1 - _p) / _n) ** 0.5
        add("prosp_cr_lo", 100 * (_p - 1.96 * _se_cr), 0.05,
            "cluster-robust (sandwich) 95% lower bound on prospective "
            "coverage, resampling unit = quantized checkpoint [HEADLINE]")
        add("prosp_cr_hi", 100 * (_p + 1.96 * _se_cr), 0.05,
            "cluster-robust (sandwich) 95% upper bound [HEADLINE]")
        add("prosp_cr_se_pp", 100 * _se_cr, 0.02,
            "cluster-robust standard error of prospective coverage, in points")
        add("prosp_naive_se_pp", 100 * _se_naive, 0.02,
            "naive binomial standard error, in points, for comparison")
        add("prosp_se_ratio", _se_cr / _se_naive, 0.005,
            "cluster-robust SE divided by the naive binomial SE; below 1 "
            "means clustering REDUCES the standard error on this statistic")

        # Why: the intraclass correlation of the coverage INDICATOR, by a
        # one-way ANOVA on the binary outcome. Negative means rows inside a
        # checkpoint are no more alike than rows across checkpoints.
        _msb = sum(len(_c) * (_c.mean() - _p) ** 2 for _c in _cl) / (_n_cl - 1)
        _msw = sum(((_c - _c.mean()) ** 2).sum() for _c in _cl) / (_n - _n_cl)
        _m0 = (_n - (_N ** 2).sum() / _n) / (_n_cl - 1)
        _icc = (_msb - _msw) / (_msb + (_m0 - 1) * _msw)
        add("prosp_icc", _icc, 0.002,
            "intraclass correlation of the coverage indicator across "
            "checkpoints; negative means no within-checkpoint clustering")
        add("prosp_design_effect", 1 + (_N.mean() - 1) * max(0.0, _icc), 0.005,
            "design effect 1 + (mbar-1)*ICC, clamped at ICC=0")
        add("prosp_rows_per_ckpt_max", int(_N.max()), 0,
            "most prospective rows contributed by a single checkpoint")

        # The miss distribution, which is what shows the ICC is real.
        _miss_by_ck = sorted(((len(_c) - _c.sum()) for _c in _cl), reverse=True)
        add("prosp_ckpts_with_miss", sum(1 for _m in _miss_by_ck if _m > 0), 0,
            "prospective checkpoints carrying at least one miss")
        add("prosp_ckpts_clean", sum(1 for _m in _miss_by_ck if _m == 0), 0,
            "prospective checkpoints with no miss")
        add("prosp_worst_ckpt_misses", int(_miss_by_ck[0]), 0,
            "misses held by the single worst prospective checkpoint")
        add("prosp_total_misses", _n - _k, 0, "prospective misses in total")
        _bm = _st.groupby("benchmark").inside.apply(lambda _c: int((~_c).sum()))
        add("prosp_worst_bench_misses", int(_bm.max()), 0,
            "misses held by the single worst benchmark; larger than the "
            "worst checkpoint is the mechanism behind the negative ICC")
        add("prosp_bench_with_miss", int((_bm > 0).sum()), 0,
            "benchmarks carrying at least one prospective miss")

        # The decomposition, so the narrowing cannot read as shopping.
        # Analytic throughout, so it is reproducible: exact CP -> naive Wald
        # isolates the estimator (CP is conservative); naive Wald ->
        # cluster-robust Wald isolates the change of unit.
        _w_cp = 100 * (_beta.ppf(0.975, _k + 1, _n - _k)
                       - _beta.ppf(0.025, _k, _n - _k + 1))
        _w_wald = 2 * 1.96 * 100 * _se_naive
        _w_cr = 2 * 1.96 * 100 * _se_cr
        add("prosp_width_cp", _w_cp, 0.05, "width of the row-level exact CP interval")
        add("prosp_width_wald", _w_wald, 0.05,
            "width of the row-level naive Wald interval")
        add("prosp_width_cr", _w_cr, 0.05,
            "width of the checkpoint-level cluster-robust interval [HEADLINE]")
        add("prosp_narrowing_estimator", _w_cp - _w_wald, 0.05,
            "points of the CP-to-headline narrowing attributable to the "
            "estimator, i.e. Clopper-Pearson being exact-conservative")
        add("prosp_narrowing_unit", _w_wald - _w_cr, 0.05,
            "points attributable to changing the unit from row to checkpoint")

    if cc.get("pooled", {}).get("coverage_one_sided") is not None:
        _one_sided = 100 * cc["pooled"]["coverage_one_sided"]
        add("pooled_one_sided_pct", _one_sided,
            0.1, "pooled one-sided coverage")
        add("m3_gap_pp", abs(_one_sided - 95.1), 0.05,
            "|pooled_one_sided_pct - 95.1| (the engineer's paraphrased "
            "figure in M3); dynamic on the corpus")
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
        add("os_pooled_pairs", o["pooled_scored_pairs"], 0,
            "pooled evaluations recomputed in the one-sided audit")
        add("os_pooled_rows", o["pooled_distinct_rows"], 0,
            "pooled distinct rows recomputed in the one-sided audit")
        for cell, v in o["cells"].items():
            for bm, mm in (v.get("miss_mean_by_checkpoint") or {}).items():
                add(f"os_missmean::{cell}::{bm}", mm, 0.01,
                    f"mean delta of misses for {bm} in {cell}")
            for key, tol in (("two_sided_pct", 0.1), ("one_sided_pct", 0.1),
                             ("below_lo_pct", 0.1), ("above_hi_pct", 0.1),
                             ("scored_pairs", 0), ("distinct_rows", 0),
                             ("distinct_checkpoints", 0),
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
        add("band_n_pairs", i_["n_scored_pairs"], 0,
            "evaluations in the band comparison")
        add("band_n_rows", i_["n_distinct_rows"], 0,
            "distinct rows in the band comparison")
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
            "Qwen2.5-0.5B-Instruct"], 0.01,
            "published MMLU, Qwen2.5-0.5B-Instruct card (5-shot)")
        add("s2_pub_qwen05_base", q["published_mmlu_targets_base_card"][
            "Qwen2.5-0.5B"], 0.01,
            "published MMLU, Qwen2.5-0.5B base card (5-shot), for context")
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

    # Is the per-scheme mean's small advantage distinguishable from zero?
    # Leave-one-family-out, recomputed here; bootstrap resamples base checkpoints.
    import numpy as _np2
    _rows = []
    for _f in sorted(d.family.unique()):
        _tr, _te = d[d.family != _f], d[d.family == _f]
        _sm2, _g2 = _tr.groupby("scheme").delta.mean(), _tr.delta.mean()
        _t = _te.copy()
        _t["diff"] = ((_t.delta - _g2).abs()
                      - (_t.delta - _t.scheme.map(_sm2).fillna(_g2)).abs())
        _rows.append(_t)
    _r = pd.concat(_rows)
    _cl2 = [_x["diff"].to_numpy() for _, _x in _r.groupby("base_model")]
    _S2 = _np2.array([_c.sum() for _c in _cl2]); _N2 = _np2.array([len(_c) for _c in _cl2])
    _rng2 = _np2.random.default_rng(0)
    _b2 = [_S2[_i].sum() / _N2[_i].sum() for _i in
           (_rng2.integers(0, len(_cl2), len(_cl2)) for _ in range(20000))]
    add("mae_gain_ci_lo", _np2.percentile(_b2, 2.5), 0.003,
        "checkpoint-bootstrap 95% lower bound of the per-scheme-mean MAE gain")
    add("mae_gain_ci_hi", _np2.percentile(_b2, 97.5), 0.003,
        "checkpoint-bootstrap 95% upper bound of the per-scheme-mean MAE gain")
    add("mae_gain_fams_pos", int((_r.groupby("family")["diff"].mean() > 0).sum()), 0,
        "held-out families where the per-scheme mean beats the global mean")

    # Tier 2.7: also computed live so a data change updates the headroom.
    if "pred_mae::global_mean" in reg and "mae_floor_pp" in reg:
        add("headroom_pp",
            reg["pred_mae::global_mean"][0] - reg["mae_floor_pp"][0],
            0.002,
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
    # --- Round item 4: nested-CV tuned baselines.
    #
    # Protocol pre-registered in TUNING_PREREGISTRATION.md and committed before
    # the run. Outcome 4 fired: both tuned baselines beat the per-scheme mean,
    # so the paper's claim that ridge and gradient boosting are worse than
    # guessing the average is retracted rather than reworded. The untuned
    # figures stay registered beside the tuned ones, because running the
    # obvious objection and reporting what it did is what makes the result
    # credible.
    _tbp = os.path.join(HERE, "..", "out", "tuned_baselines.json")
    if os.path.exists(_tbp):
        _tb = json.load(open(_tbp))
        for _k in ("ridge_tuned", "grad_boost_tuned",
                   "ridge_fixed", "grad_boost_fixed"):
            add(f"pred_mae::{_k}", _tb[f"weighted_mae::{_k}"], 0.0005,
                f"leave-one-family-out MAE for {_k}, nested-CV tuned"
                if _k.endswith("tuned") else
                f"leave-one-family-out MAE for {_k}, fixed hyperparameters")
        # How far tuning moved each model class.
        add("tune_shift_ridge", _tb["weighted_mae::ridge_fixed"]
            - _tb["weighted_mae::ridge_tuned"], 0.0005,
            "points of MAE that nested-CV tuning recovered for ridge")
        add("tune_shift_gb", _tb["weighted_mae::grad_boost_fixed"]
            - _tb["weighted_mae::grad_boost_tuned"], 0.0005,
            "points of MAE that nested-CV tuning recovered for gradient boosting")
        # The grid-edge finding: nested CV's answer is maximum regularisation.
        _ra = _tb["ridge_alphas"]
        add("tune_ridge_alpha_max", max(_ra), 0,
            "largest ridge alpha selected by any outer fold")
        add("tune_ridge_folds_at_edge",
            sum(1 for _a in _ra if float(_a) >= 1000), 0,
            "outer folds whose selected ridge alpha sat at the top of the "
            "pre-registered grid")
        add("tune_gb_folds_min_lr",
            sum(1 for _o in _tb["outer_folds"]
                if float(_o["gb_params_selected"]["learning_rate"]) <= 0.01), 0,
            "outer folds whose selected gradient-boosting learning rate sat "
            "at the bottom of the grid")
        add("tune_gb_folds_max_l2",
            sum(1 for _o in _tb["outer_folds"]
                if float(_o["gb_params_selected"]["l2_regularization"]) >= 10.0), 0,
            "outer folds whose selected L2 regularisation sat at the top of "
            "the grid")
        add("tune_outer_folds", len(_tb["outer_folds"]), 0,
            "outer leave-one-family-out folds in the tuning run")
        add("tune_gb_grid", _tb["grid_sizes"]["grad_boost"], 0,
            "gradient-boosting settings per inner fold")
        add("tune_ridge_grid", _tb["grid_sizes"]["ridge"], 0,
            "ridge settings per inner fold")
        # Round item 1 / deferred item 1: the untuned gradient-boosting
        # figure has now moved sharply twice on small corpus corrections,
        # which is a property of the estimator worth stating rather than a
        # one-off. Both instances are measured, not recalled.
        add("gb_sens_nitems_pp", 0.009327, 0.0002,
            "pp the untuned gradient-boosting LOFO MAE moved when the "
            "gpqa_diamond n_items correction changed 6 of 819 rows")
        add("gb_sens_nitems_rows", 6, 0,
            "rows whose feature changed in that correction")
        add("gb_sens_mmlu_pct", 23, 1,
            "percent the untuned gradient-boosting losing margin narrowed "
            "when the mmlu_llama fix added 2 of 819 rows")
        add("gb_sens_mmlu_rows", 2, 0, "rows added by that fix")
        # Checkpoint-bootstrap CIs on the differences (src/tune_baseline_cis.py).
        # The tuned models beat the GLOBAL mean resolvably; they beat the
        # per-scheme mean by an amount whose interval includes zero, and the
        # paper says so.
        for _dk, _label in (("scheme_vs_ridge_tuned", "per-scheme mean minus tuned ridge"),
                            ("scheme_vs_gb_tuned", "per-scheme mean minus tuned gradient boosting"),
                            ("global_vs_ridge_tuned", "global mean minus tuned ridge")):
            _dv = _tb.get("differences", {}).get(_dk)
            if _dv:
                add(f"tune_diff::{_dk}", _dv["estimate"], 0.0008, _label)
                add(f"tune_diff_lo::{_dk}", _dv["lo"], 0.003,
                    f"checkpoint-bootstrap 95% lower bound, {_label}")
                add(f"tune_diff_hi::{_dk}", _dv["hi"], 0.003,
                    f"checkpoint-bootstrap 95% upper bound, {_label}")

    # The framing that survives any reordering: how far the furthest
    # card-feature predictor sits from the global mean, against the noise
    # floor. Computed rather than hand-rounded, so the abstract quotes the
    # real bound instead of a generous one.
    if "pred_mae::global_mean" in reg and "mae_floor_pp" in reg:
        _g = reg["pred_mae::global_mean"][0]
        _cands = [reg[f"pred_mae::{_m}"][0] for _m in
                  ("scheme_mean", "scheme_x_bench", "bench_mean", "ridge",
                   "grad_boost", "ridge_tuned", "grad_boost_tuned")
                  if f"pred_mae::{_m}" in reg]
        _gap = max(abs(_v - _g) for _v in _cands)
        add("pred_max_abs_gap_pp", _gap, 0.0005,
            "largest absolute MAE gap between any card-feature predictor and "
            "the global mean, tuned and untuned")
        add("pred_gap_vs_floor_ratio", reg["mae_floor_pp"][0] / _gap, 0.05,
            "evaluation-noise floor divided by that largest gap")

    if "pred_mae::global_mean" in reg:
        # Tier 2.7: registered here (after pred_mae is populated) so the
        # stale literal 0.7545 cannot come back.
        add("mae_global_lofo", reg["pred_mae::global_mean"][0], 0.002,
            "global-mean baseline MAE, LOFO (live from "
            "out/predictor_comparison.json)")

    # --- Tier 0.1 disclosure: per-subgroup coverage of the held-out
    #     llama-3 fold. Definition B collapses Llama-3.1/3.2/3.3 into a
    #     single family (§3). Under 6-fold LOFO the merged fold hides
    #     that Llama-3.3 alone runs 5-10pp below Llama-3.1 and Llama-3.2
    #     under every calibration family, so we expose the breakdown.
    from predictor import SchemeMean as _SM_l3, Conformal as _Co_l3  # noqa: E402
    import re as _re_l3
    _tag = {"3.1": _re_l3.compile(r"Llama-3\.1", _re_l3.I),
            "3.2": _re_l3.compile(r"Llama-3\.2", _re_l3.I),
            "3.3": _re_l3.compile(r"Llama-3\.3", _re_l3.I)}

    def _sub_l3(_m):
        for _k, _p in _tag.items():
            if _p.search(_m):
                return _k
        return "other"

    _d_l3 = d.copy()
    _d_l3["_l3sub"] = _d_l3.base_model.map(_sub_l3)
    _te_l3 = _d_l3[_d_l3.family == "llama-3"].copy()
    _tr_pool = _d_l3[_d_l3.family != "llama-3"]
    _cal_fams = sorted(_tr_pool.family.unique())
    _EPS_l3 = 1e-9
    for _cf in _cal_fams:
        _ca_l3 = _tr_pool[_tr_pool.family == _cf]
        _tr_l3 = _tr_pool[_tr_pool.family != _cf]
        if len(_ca_l3) < 19 or len(_tr_l3) < 50:
            continue
        _m_l3 = _SM_l3().fit(_tr_l3)
        _c_l3 = _Co_l3(alpha=0.10, mondrian_by="scheme").fit(_m_l3, _ca_l3)
        _, _lo_l3, _hi_l3, _ = _c_l3.predict_interval(_te_l3)
        _y_l3 = _te_l3.delta.to_numpy(float)
        _in_l3 = (_y_l3 >= _lo_l3 - _EPS_l3) & (_y_l3 <= _hi_l3 + _EPS_l3)
        for _s in ("3.1", "3.2", "3.3"):
            _mask = (_te_l3["_l3sub"].to_numpy() == _s)
            if _mask.any():
                add(f"lofo_l3sub_cov::{_s}::{_cf}",
                    100 * float(_in_l3[_mask].mean()), 0.05,
                    f"held-out llama-3 sub-{_s} coverage under {_cf} "
                    f"calibration (Def B breakdown; §6 subgroup table)")
        # merged (all-3.x) fold coverage under this cal family, for the
        # table's rightmost column
        add(f"lofo_l3merged_cov::{_cf}",
            100 * float(_in_l3.mean()), 0.05,
            f"held-out llama-3 merged coverage under {_cf} calibration")

    # --- Tier 1.1 disclosure: LOFO fallback rate under the 6-family corpus
    #     (day-7 audit). ConservativeStratified.fit_calibrated keeps the
    #     training-set half-width whenever the calibration family has <9
    #     rows of a scheme, and labels the result as if it had been
    #     calibrated. We disclose the pooled rate and the per-scheme rate;
    #     the paper's §6 states plainly that fp8, nvfp4 and w8a16 fall back
    #     on ~60% of their evaluations while w4a16 never does.
    from strata import ConservativeStratified as _CS_lo
    _fams_lo = sorted(d.family.unique())
    _lo_total = 0
    _lo_fallback = 0
    _lo_by_scheme = {}
    _lo_held_covered = 0
    _lo_held_total = 0
    _lo_held_ok1 = 0
    for _tf in _fams_lo:
        _te_lo = d[d.family == _tf]
        for _cf_lo in _fams_lo:
            if _cf_lo == _tf:
                continue
            _ca_lo = d[d.family == _cf_lo]
            _tr_lo = d[~d.family.isin([_tf, _cf_lo])]
            if len(_te_lo) == 0 or len(_ca_lo) < 19 or len(_tr_lo) < 50:
                continue
            _m_lo = _CS_lo().fit_calibrated(_tr_lo, _ca_lo)
            _, _lo_, _hi_, _ = _m_lo.predict_interval(_te_lo)
            _y_lo = _te_lo.delta.to_numpy(float)
            _EPS_lo = 1e-9
            _in2 = (_y_lo >= _lo_ - _EPS_lo) & (_y_lo <= _hi_ + _EPS_lo)
            _os_ = (_y_lo >= _lo_ - _EPS_lo)
            _sch_cnt = _ca_lo.groupby("scheme").size().to_dict()
            _te2 = _te_lo.reset_index(drop=True)
            for _i in range(len(_te2)):
                _sc = _te2.iloc[_i].scheme
                _fb = _sch_cnt.get(_sc, 0) < 9
                _lo_total += 1
                _lo_fallback += int(_fb)
                _lo_by_scheme.setdefault(_sc, [0, 0])
                _lo_by_scheme[_sc][0] += 1
                _lo_by_scheme[_sc][1] += int(_fb)
                if not _fb:
                    _lo_held_total += 1
                    _lo_held_covered += int(_in2[_i])
                    _lo_held_ok1 += int(_os_[_i])
    add("lofo_fallback_pairs", _lo_fallback, 0,
        "LOFO pairs where calibration family had <9 rows of the row's scheme")
    add("lofo_total_pairs", _lo_total, 0,
        "total LOFO (test row, cal family) pairs")
    add("lofo_fallback_rate_pct", 100 * _lo_fallback / _lo_total, 0.05,
        "share of LOFO pairs using the training-set half-width fallback")
    for _sc, (_n, _nf) in _lo_by_scheme.items():
        add(f"lofo_fallback_rate_pct::{_sc}", 100 * _nf / _n, 0.05,
            f"fallback rate for {_sc} under 6-fam LOFO")
    add("lofo_heldout_pairs", _lo_held_total, 0,
        "LOFO pairs after excluding fallback rows")
    add("lofo_heldout_cov_pct", 100 * _lo_held_covered / _lo_held_total, 0.05,
        "pooled two-sided coverage on held-out (fallback-excluded) LOFO evals")
    add("lofo_heldout_one_sided_pct", 100 * _lo_held_ok1 / _lo_held_total, 0.05,
        "pooled one-sided coverage on held-out LOFO evals")

    # --- Tier 1.3 disclosure: pre-registered set had 34 repos; 9 are
    #     name-gated (no <n>B token). The frozen-set headline is 119/131
    #     (registry). The with-them figure 143/156 comes from an external
    #     audit and is NOT locally reproducible (the 9 cards are not in
    #     data/prospective_cards/). Registered here so the paper prose can
    #     reference the two values through claim tags; the value is treated
    #     as EXTERNAL by the traceability audit.
    add("prosp_registered_repos", 34, 0, "pre-registered prospective repos")
    add("prosp_name_gated_repos", 9, 0,
        "pre-registered repos rejected by parse_params_b name-gate")
    add("prosp_with_gated_inside", 143, 0,
        "with-gated-repos rows inside the interval (EXTERNAL, day-7 audit; "
        "not locally reproducible without the 9 gated cards)")
    add("prosp_with_gated_total", 156, 0,
        "with-gated-repos total rows (EXTERNAL, day-7 audit)")
    add("prosp_with_gated_cov_pct", 100 * 143 / 156, 0.02,
        "with-gated-repos coverage (EXTERNAL, day-7 audit)")

    # Tier 2.8 (day-7 audit): gemma-3-1b-it W4A16 has 6 TruthfulQA-and-
    # friends deltas; how many the shipped conformal bound catches was
    # hand-typed as "2 of 6" and now says 3 under the 1e-9 tolerance
    # (matches the bias_correction_empirical docstring). Registered live.
    _gem = [-3.03, -2.99, -2.90, -2.41, -1.34, 1.40]
    _EPS_gem = 1e-9
    _lo_gem, _hi_gem = -2.87, 1.40
    # Tier 2.5 v4: after inspecting EVERY GPQA card the plain 'gpqa'
    # label should be gone (all rows carry a card-verified specific
    # protocol label). Registered at 0 so a future re-introduction of
    # 'gpqa' would fail the audit rather than silently return.
    add("gpqa_other_rows",
        int((d.benchmark == "gpqa").sum()), 0,
        "rows still under the plain 'gpqa' label after Tier 2.5 v4 "
        "(expected 0 -- every row now carries a card-verified label)")
    _gpqa_labels = ("gpqa_main", "gpqa_main_norm", "gpqa_main_cot_5shot",
                    "gpqa_diamond", "gpqa_diamond_cot_5shot",
                    "gpqa_ambiguous_46")
    for _lbl in _gpqa_labels:
        add(f"n_rows_{_lbl}", int((d.benchmark == _lbl).sum()), 0,
            f"rows carrying the card-verified {_lbl} label")
    add("n_gpqa_labels", len(_gpqa_labels), 0,
        "number of card-verified GPQA protocol labels in the corpus (Step 4a)")
    add("n_raw_dataset_rows", len(pd.read_csv(
        os.path.join(HERE, "..", "data", "dataset.csv"))), 0,
        "raw rows in data/dataset.csv before the acc_before >= 20 modelling "
        "filter (Step 4a)")

    add("gemma_1b_wfour_caught",
        int(sum(1 for _x in _gem
                if _lo_gem - _EPS_gem <= _x <= _hi_gem + _EPS_gem)), 0,
        "of the 6 gemma-3-1b-it W4A16 deltas, how many the shipped "
        "conformal bound catches (with 1e-9 boundary tolerance, Tier 2.8)")
    add("gemma_1b_wfour_total", len(_gem), 0,
        "total gemma-3-1b-it W4A16 deltas evaluated in the correction note")

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
