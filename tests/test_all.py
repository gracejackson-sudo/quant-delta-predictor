"""Unit + regression tests. Run: ./.venv/bin/python -m pytest tests -q"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import harvest as H  # noqa: E402
from model import conformal_quantile, featurize, load, noise_scale  # noqa: E402
from predictor import Conformal, SchemeMean  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")


# ------------------------------------------------------------ cell parsing
@pytest.mark.parametrize("cell,expect", [
    ("73.5", 73.5), ("**73.5**", 73.5), ("105.4%", 105.4),
    ("25.8 (25.1 / 26.5)", 25.8), ("  81.60  ", 81.6), ("100", 100.0),
    ("-1.5", -1.5), ("**Recovery (%)**", None),
    ("MMLU (5-shot)", None),            # must NOT parse as 5
    ("GSM-8K (CoT, 8-shot, strict-match)", None),
    ("Average", None), ("", None), ("Llama-3.1-8B", None),
])
def test_num_cell(cell, expect):
    assert H.num(cell) == expect


def test_parse_row_takes_trailing_numeric_run():
    lab, nums = H.parse_row(["OpenLLM v1", "MMLU (5-shot)", "81.60", "81.31",
                             "99.6%"])
    assert lab == "MMLU (5-shot)"
    assert nums == [81.6, 81.31, 99.6]


def test_parse_row_handles_rowspan_missing_category_cell():
    lab, nums = H.parse_row(["Hellaswag (10-shot)", "86.49", "86.43", "99.9%"])
    assert lab == "Hellaswag (10-shot)" and len(nums) == 3


# ------------------------------------------------------------ benchmark names
@pytest.mark.parametrize("label,expect", [
    ("MMLU (5-shot)", "mmlu"),
    ("MMLU (CoT, 0-shot)", "mmlu_cot"),
    ("MMLU-Pro (5-shot)", "mmlu_pro"),
    ("ARC Challenge (0-shot)", "arc_challenge"),
    ("GSM-8K (CoT, 8-shot, strict-match)", "gsm8k"),
    ("HumanEval+ pass@1", "humaneval_plus"),
    ("HumanEval pass@1", "humaneval"),
    ("Math-Hard (Exact-Match, 4-shot)", "math_lvl5"),
    ("**Average**", None), ("**Recovery (%)**", None),
    ("Multilingual MMLU (Portuguese)", None),   # not a target benchmark
])
def test_canon_benchmark(label, expect):
    assert H.canon_benchmark(label) == expect


def test_mmlu_pro_wins_over_mmlu():
    """Ordering bug guard: 'MMLU-Pro' must not fall through to 'mmlu'."""
    assert H.canon_benchmark("MMLU-Pro (5-shot)") == "mmlu_pro"


# ------------------------------------------------------------ tolerance logic
def test_recovery_tolerance_tight_for_high_accuracy():
    # 81.60 -> 81.31 printed to 2dp: tolerance must be small enough to
    # discriminate orientation (the two readings differ by ~0.71)
    t = H.recovery_tolerance(81.60, 81.31, "81.60", "81.31", "99.6")
    assert t < 0.2
    assert abs(99.6 - 100 * 81.31 / 81.60) < t       # correct reading passes
    assert abs(99.6 - 100 * 81.60 / 81.31) > t       # swapped reading fails


def test_recovery_tolerance_loose_for_near_random():
    # GPQA at 3.7 -> 4.0 at 1dp: rounding moves the ratio a lot, so the
    # tolerance must widen or we would wrongly reject a valid row
    t = H.recovery_tolerance(3.7, 4.0, "3.7", "4.0", "109.8")
    assert t > 1.0
    assert abs(109.8 - 100 * 4.0 / 3.7) < t


def test_extract_rejects_internally_inconsistent_row():
    html = ("<table><tr><td>Category</td><td>Benchmark</td>"
            "<td>Base</td><td>Quant FP8 (this model)</td><td>Recovery</td></tr>"
            "<tr><td>MMLU (5-shot)</td><td>81.40</td><td>80.20</td>"
            "<td>60.0%</td></tr></table>")
    got = list(H.extract("## Evaluation\n" + html))
    assert got == [("mmlu", None, None, "reject")]


def test_extract_reads_correct_orientation():
    html = ("<table><tr><td>Benchmark</td><td>Base</td>"
            "<td>Quant (this model)</td><td>Recovery</td></tr>"
            "<tr><td>MMLU (5-shot)</td><td>81.60</td><td>81.31</td>"
            "<td>99.6%</td></tr></table>")
    (b, before, after, src), = list(H.extract("## Evaluation\n" + html))
    assert (b, before, after, src) == ("mmlu", 81.60, 81.31, "arith")


def test_extract_detects_swapped_columns():
    """Quantized column printed FIRST must still come out as 'after'."""
    html = ("<table><tr><td>Benchmark</td><td>Quant w4a16 (this model)</td>"
            "<td>Base</td><td>Recovery</td></tr>"
            "<tr><td>MMLU (5-shot)</td><td>81.31</td><td>81.60</td>"
            "<td>99.6%</td></tr></table>")
    (_, before, after, src), = list(H.extract("## Evaluation\n" + html))
    assert (before, after) == (81.60, 81.31) and src == "arith"


def test_recovery_first_column_order():
    """
    Regression: the Llama-4 cards order columns [Recovery, base, quant].
    Assuming [base, quant, Recovery] turned a recovery of 100.0 into a GPQA
    'accuracy before' of 100.0 and a fake -68pp delta.
    """
    md = ("## Evaluation\n\n"
          "|            | Recovery (%) | meta-llama/Llama-4-Scout "
          "| RedHatAI/Llama-4-Scout-quantized.w4a16<br>(this model) |\n"
          "|---|:---:|:---:|:---:|\n"
          "| ARC-Challenge<br>25-shot | 98.51 | 69.37 | 68.34 |\n"
          "| GPQA<br>0-shot | 100.0 | 31.88 | 31.88 |\n")
    got = {b: (before, after) for b, before, after, _ in H.extract(md)}
    assert got["arc_challenge"] == (69.37, 68.34)
    assert got["gpqa"] == (31.88, 31.88)
    assert all(v[0] <= 100 for v in got.values())


def test_no_accuracy_is_a_recovery_percentage():
    """A recovery column must never be emitted as an accuracy."""
    md = ("## Evaluation\n\n"
          "|  | Recovery (%) | base | quant-FP8 (this model) |\n"
          "|---|:---:|:---:|:---:|\n"
          "| MMLU<br>5-shot | 99.75 | 80.54 | 80.34 |\n")
    (_, before, after, src), = list(H.extract(md))
    assert (before, after) == (80.54, 80.34) and src == "arith"


def test_markdown_table_two_columns():
    md = ("## Evaluation\n\n"
          "| Metric | mistralai/Base | nm-testing/Base-FP8-dynamic |\n"
          "|---|---|---|\n"
          "| MMLU (Acc, 5-shot) | 80.69 | 80.55 |\n")
    (b, before, after, src), = list(H.extract(md))
    assert (b, before, after) == ("mmlu", 80.69, 80.55)
    assert src == "header_only"   # no recovery column to verify against


# ------------------------------------------------------------ config parsing
@pytest.mark.parametrize("mid,scheme,wb,ab", [
    ("RedHatAI/X-quantized.w4a16", "w4a16", 4, 16),
    ("RedHatAI/X-quantized.w8a8", "w8a8_int", 8, 8),
    ("RedHatAI/X-quantized.w8a16", "w8a16", 8, 16),
    ("RedHatAI/X-FP8-dynamic", "fp8_dynamic", 8, 8),
    ("RedHatAI/X-FP8", "fp8", 8, 8),
    ("RedHatAI/X-NVFP4", "nvfp4", 4, 4),
])
def test_parse_config(mid, scheme, wb, ab):
    s, w, a, _ = H.parse_config(mid)
    assert (s, w, a) == (scheme, wb, ab)


def test_fp8_dynamic_not_misread_as_plain_fp8():
    assert H.parse_config("RedHatAI/X-FP8-dynamic")[0] == "fp8_dynamic"


@pytest.mark.parametrize("mid,params", [
    ("RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16", 8.0),
    ("RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8", 405.0),
    ("RedHatAI/Qwen2.5-0.5B-quantized.w8a8", 0.5),
    ("RedHatAI/gemma-2-9b-it-quantized.w4a16", 9.0),
])
def test_parse_params(mid, params):
    assert H.parse_params_b(mid) == params


def test_family_boundary_anchored():
    """Regression: 'diffusiongemma-26B' must not be labelled gemma-2."""
    assert H.parse_family("RedHatAI/gemma-2-9b-it-FP8") == "gemma-2"
    assert H.parse_family("RedHatAI/diffusiongemma-26B-A4B-it-NVFP4") == "other"


def test_base_model_strips_quant_suffix():
    assert (H.base_model_name(
        "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16")
        == "Meta-Llama-3.1-8B-Instruct")
    assert (H.base_model_name("RedHatAI/Qwen3-8B-FP8-dynamic")
            == "Qwen3-8B")


def test_base_model_groups_configs_of_same_checkpoint():
    ids = ["RedHatAI/Qwen3-8B-quantized.w4a16", "RedHatAI/Qwen3-8B-FP8-dynamic"]
    assert len({H.base_model_name(i) for i in ids}) == 1


# ------------------------------------------------------------ conformal
def test_conformal_index():
    s = np.arange(1.0, 21.0)          # n=20
    assert conformal_quantile(s, 0.10) == 19.0   # ceil(21*0.9)=19


def test_conformal_infinite_when_too_few():
    assert not np.isfinite(conformal_quantile(np.arange(8.0), 0.10))
    assert np.isfinite(conformal_quantile(np.arange(9.0), 0.10))
    assert not np.isfinite(conformal_quantile(np.arange(18.0), 0.05))


def test_conformal_quantile_monotone_in_alpha():
    s = np.random.default_rng(0).normal(size=500)
    q = [conformal_quantile(np.abs(s), a) for a in (0.20, 0.10, 0.05)]
    assert q[0] < q[1] < q[2]


def test_conformal_coverage_on_exchangeable_data():
    """The guarantee must hold empirically when exchangeability truly holds."""
    rng = np.random.default_rng(11)
    n = 3000
    df = pd.DataFrame({"scheme": rng.choice(["a", "b"], n),
                       "benchmark": "mmlu", "n_items": 1000,
                       "acc_before": 60.0})
    df["delta"] = df.scheme.map({"a": -1.0, "b": 0.5}) + rng.normal(0, 1, n)
    df["acc_after"] = df.acc_before + df.delta
    covs = []
    for seed in range(30):
        i = np.random.default_rng(seed).permutation(n)
        m = SchemeMean().fit(df.iloc[i[:1500]])
        c = Conformal(alpha=0.10, mondrian_by="scheme").fit(m, df.iloc[i[1500:2200]])
        te = df.iloc[i[2200:]]
        _, lo, hi, _ = c.predict_interval(te)
        y = te.delta.to_numpy()
        covs.append(np.mean((y >= lo) & (y <= hi)))
    assert 0.88 <= np.mean(covs) <= 0.92


def test_interval_contains_point_prediction():
    d = load(DATA)
    m = SchemeMean().fit(d)
    c = Conformal(alpha=0.10, mondrian_by="scheme").fit(m, d)
    yhat, lo, hi, _ = c.predict_interval(d)
    assert np.all(lo <= yhat) and np.all(yhat <= hi)


# ------------------------------------------------------------ predictor
def test_scheme_mean_backs_off_for_unseen_scheme():
    d = load(DATA)
    tr = d[d.scheme != "nvfp4"]
    m = SchemeMean().fit(tr)
    te = d[d.scheme == "nvfp4"]
    assert np.allclose(m.predict(te), m.global_mean)


def test_mondrian_backs_off_when_group_absent_from_calibration():
    d = load(DATA)
    m = SchemeMean().fit(d)
    cal = d[d.scheme != "nvfp4"]
    c = Conformal(alpha=0.10, mondrian_by="scheme", backoff=True).fit(m, cal)
    te = d[d.scheme == "nvfp4"]
    _, lo, hi, fb = c.predict_interval(te)
    assert fb.all() and np.all(np.isfinite(hi - lo))


def test_mondrian_without_backoff_returns_infinite():
    d = load(DATA)
    m = SchemeMean().fit(d)
    c = Conformal(alpha=0.10, mondrian_by="scheme",
                  backoff=False).fit(m, d[d.scheme != "nvfp4"])
    _, lo, hi, _ = c.predict_interval(d[d.scheme == "nvfp4"])
    assert np.all(~np.isfinite(hi - lo))


# ------------------------------------------------------------ features
def test_features_have_no_target_leakage():
    d = load(DATA)
    X1, names = featurize(d)
    d2 = d.copy()
    d2["acc_after"] = 0.0
    d2["delta"] = 0.0
    X2, _ = featurize(d2)
    assert np.allclose(X1, X2)
    assert not any("after" in n for n in names)


def test_noise_scale_decreases_with_items():
    d = pd.DataFrame({"acc_before": [60.0, 60.0], "n_items": [164, 14042]})
    ns = noise_scale(d)
    assert ns[0] > ns[1]


# ------------------------------------------------------------ dataset invariants
def test_dataset_invariants():
    d = pd.read_csv(DATA)
    assert len(d) > 300
    assert d.acc_before.between(0.001, 100).all()
    assert d.acc_after.between(0, 100).all()
    assert np.allclose(d.delta, d.acc_after - d.acc_before)
    assert not d.duplicated(["model", "benchmark"]).any()
    assert (d.groupby("model").acc_before.max() > 1.0).all()  # no 0-1 scale
    assert "other" not in set(d.family)
    assert d.verified.mean() > 0.9


# ------------------------------------------------------------ strata / ranking
from strata import (  # noqa: E402
    ConservativeStratified, SchemeOnlyBaseline, StratifiedBaseline,
    annotate, is_moe, size_band,
)
import rank as R  # noqa: E402


@pytest.mark.parametrize("p,band", [
    (0.5, "<2B"), (1.9, "<2B"), (2.0, "2-10B"), (8.0, "2-10B"),
    (10.0, "2-10B"), (10.1, ">10B"), (405.0, ">10B"), (None, "unknown"),
])
def test_size_band(p, band):
    assert size_band(p) == band


@pytest.mark.parametrize("name,expect", [
    ("Mixtral-8x7B-Instruct-v0.1", True), ("Qwen3-30B-A3B", True),
    ("Llama-4-Scout-17B-16E-Instruct", True),
    ("Meta-Llama-3.1-8B-Instruct", False), ("gemma-2-9b-it", False),
])
def test_is_moe(name, expect):
    assert is_moe(name) is expect


def test_conservative_never_narrows_the_scheme_interval():
    """The whole point of the conservative variant: widen only."""
    d = load(DATA)
    full = StratifiedBaseline().fit(d)
    cons = ConservativeStratified().fit(d)
    _, lo_c, hi_c, _ = cons.predict_interval(d)
    scheme_hw = d.scheme.map(
        lambda s: full.by_scheme[s]["half_width"]).to_numpy()
    assert np.all((hi_c - lo_c) / 2 >= scheme_hw - 1e-9)


def test_conservative_centres_on_the_scheme_mean():
    d = load(DATA)
    m = ConservativeStratified().fit(d)
    yhat, _, _, _ = m.predict_interval(d)
    assert np.allclose(yhat, d.scheme.map(
        lambda s: m.by_scheme[s]["mean"]).to_numpy())


def test_thin_cells_are_rejected_not_used():
    """nvfp4/2-10B has 13 rows from ONE checkpoint and must not form a cell."""
    m = ConservativeStratified().fit(load(DATA))
    for (s, b), c in m.by_stratum.items():
        assert c["n"] >= 20 and c["n_checkpoints"] >= 3
    assert ("nvfp4", "2-10B") in m.rejected


def test_fit_calibrated_uses_held_out_widths():
    d = load(DATA)
    tr, ca = d[d.family != "mistral"], d[d.family == "mistral"]
    a = ConservativeStratified().fit(tr)
    b = ConservativeStratified().fit_calibrated(tr, ca)
    assert a.by_scheme["w4a16"]["mean"] == b.by_scheme["w4a16"]["mean"]
    assert a.by_scheme["w4a16"]["half_width"] != \
        b.by_scheme["w4a16"]["half_width"]


# ---- ranking
@pytest.fixture(scope="module")
def table():
    return R.build_table()


def test_alias_resolution():
    assert R.resolve("INT4") == "w4a16"
    assert R.resolve("fp8-dynamic") == "fp8_dynamic"
    assert R.resolve("not-a-scheme") is None


def test_rank_orders_tier_then_width(table):
    ranked, unknown = R.rank(["nvfp4", "fp8", "w4a16", "w8a8"], table)
    assert unknown == []
    tiers = [e["tier"] for e in ranked]
    assert tiers == sorted(tiers)            # tier A before B before C
    for a, b in zip(ranked, ranked[1:]):
        if a["tier"] == b["tier"]:
            assert a["half_width"] <= b["half_width"] + 1e-9


def test_unknown_schemes_are_reported_not_silently_dropped(table):
    ranked, unknown = R.rank(["fp8", "banana"], table)
    assert unknown == ["banana"] and len(ranked) == 1


def test_tail_risk_uses_rate_not_single_worst(table):
    """fp8_dynamic has a -8.72pp outlier but a 1% severe rate: not TAIL_RISK."""
    e = dict(table["fp8_dynamic"])
    flags, _ = R.assess(e, R.DEFAULT_RISK_PP)
    assert e["worst_observed"] < -8
    assert "TAIL_RISK" not in flags
    e2 = dict(table["nvfp4"])
    flags2, _ = R.assess(e2, R.DEFAULT_RISK_PP)
    assert "TAIL_RISK" in flags2


def test_worst_case_discrepancy_is_always_surfaced(table):
    """A tail far below the interval floor must produce a visible note."""
    for s in ("w4a16", "nvfp4", "fp8_dynamic"):
        e = dict(table[s])
        _, notes = R.assess(e, R.DEFAULT_RISK_PP)
        assert any("worst observed loss" in n for n in notes), s


def test_small_model_queries_state_measured_cell_coverage(table):
    """
    At a given size every scheme must either refuse, be held back for
    insufficient evidence, or state the MEASURED coverage for that cell. A
    blanket 'not validated at this size' claim was removed because it was
    itself unverified prose.
    """
    cc = R.load_cell_coverage()
    for s in table:
        if s == "_meta":
            continue
        e = dict(table[s])
        flags, notes = R.assess(e, R.DEFAULT_RISK_PP, band="<2B",
                                cell_cov=cc)
        if e.get("refused"):
            assert "INSUFFICIENT_CALIBRATION" in flags
            assert R.tier_of(flags) == "C"
        elif e.get("insufficient_evidence"):
            assert "INSUFFICIENT_EVIDENCE" in flags
            assert R.tier_of(flags) == "C"
        else:
            assert any("measured to contain the true result" in n
                       or "no coverage has been measured" in n
                       for n in notes), s


def test_moe_flag(table):
    e = dict(table["fp8"])
    flags, notes = R.assess(e, R.DEFAULT_RISK_PP, moe=True)
    assert "MOE_UNVALIDATED" in flags
    assert any("mixture-of-experts" in n for n in notes)


def test_every_scheme_reports_support_and_coverage(table):
    for s, e in table.items():
        if s == "_meta":
            continue
        assert e["n"] > 0 and e["n_checkpoints"] > 0 and e["n_families"] > 0
        assert 0.0 <= e["validated_coverage"] <= 1.0
        assert e["worst_observed"] <= e["mean"]


# ------------------------------------------------- standing rule + refusals
def test_every_scheme_states_its_worst_observed_loss(table):
    """Regardless of tier or rank, no scheme may be silent about its tail."""
    for s in table:
        if s == "_meta":
            continue
        e = dict(table[s])
        _, notes = R.assess(e, R.DEFAULT_RISK_PP)
        assert any("worst observed loss" in n for n in notes), s


def test_worst_case_line_is_tier_independent(table):
    ranked, _ = R.rank([k for k in table if k != "_meta"], table)
    for e in ranked:
        assert any("worst observed loss" in n for n in e["notes"]), e["scheme"]
    assert {e["tier"] for e in ranked} >= {"A", "C"}   # both tiers present


def test_undercovered_cells_are_refused_or_insufficient_not_estimated():
    """A cell that fails the coverage judgment -- whether cleanly refused or
    held back by the checkpoint/bootstrap evidence floor -- must not emit a
    confident-looking number. This does not assert which of the two states a
    given cell lands in; that is a property of the evidence, not a constant
    to pin down (see ONE_SIDED_COVERAGE.md: the checkpoint floor moved both
    previously-refused cells into 'insufficient evidence')."""
    cc = R.load_cell_coverage()
    table = R.build_table()
    flagged = [k for k, v in cc["cells"].items()
               if R.classify_cell(v) in ("refused", "insufficient_evidence")]
    assert flagged, "expected at least one refused or insufficient-evidence cell"
    for key in flagged:
        scheme, band = key.split("|")
        ranked, _ = R.rank([scheme], table, band=band, cell_cov=cc)
        e = ranked[0]
        assert e["refused"] is True or e["insufficient_evidence"] is True, key
        expected_flag = ("INSUFFICIENT_CALIBRATION" if e["refused"]
                          else "INSUFFICIENT_EVIDENCE")
        assert expected_flag in e["flags"], key
        assert e["tier"] == "C", key


def test_well_covered_cells_still_get_a_number():
    cc = R.load_cell_coverage()
    table = R.build_table()
    ranked, _ = R.rank(["fp8"], table, band="<2B", cell_cov=cc)
    assert ranked[0]["refused"] is False


def test_no_banned_prose_in_user_facing_output():
    """Claims that were wrong before must not reappear in printed text."""
    banned = ["safe to adopt", "ignores your model", "12 numbers",
              "26 of 29", "80%, not 90%"]
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    for fn in ("rank.py", "cli.py", "build_envelope.py"):
        p = os.path.join(src, fn)
        if not os.path.exists(p):
            continue
        text = open(p, encoding="utf-8").read().lower()
        for b in banned:
            assert b not in text, f"{fn} still contains {b!r}"


def test_ranking_doc_claims_all_verify():
    """The standing rule, enforced: every number in RANKING.md must compute."""
    import verify_claims
    assert verify_claims.main() == 0


def test_thin_cell_support_blocks_tier_a():
    """A cell with high coverage but one checkpoint must not reach Tier A."""
    cc = R.load_cell_coverage()
    table = R.build_table()
    blocked = [k for k, v in cc["support"].items()
               if v["train_checkpoints"] < R.MIN_CELL_CHECKPOINTS]
    assert blocked, "expected at least one under-supported cell"
    for key in blocked:
        scheme, band = key.split("|")
        if scheme not in table:
            continue
        e = dict(table[scheme])
        flags, notes = R.assess(e, R.DEFAULT_RISK_PP, band=band, cell_cov=cc)
        assert "THIN_CELL_SUPPORT" in flags, key
        assert R.tier_of(flags) != "A", key
        assert any("distinct checkpoint" in n for n in notes), key


def test_fp8_small_is_blocked_despite_perfect_coverage():
    """Regression for the specific case: fp8|<2B covers 100% on 1 checkpoint."""
    cc = R.load_cell_coverage()
    sup = cc["support"]["fp8|<2B"]
    assert sup["train_checkpoints"] < R.MIN_CELL_CHECKPOINTS
    assert cc["cells"]["fp8|<2B"]["coverage"] >= 0.99
    ranked, _ = R.rank(["fp8"], R.build_table(), band="<2B", cell_cov=cc)
    assert ranked[0]["tier"] == "B"
    assert "THIN_CELL_SUPPORT" in ranked[0]["flags"]


# ------------------------------------------------- track 2 / bias correction
def test_lowrank_does_not_beat_scheme_mean_on_deltas():
    """Track 2: low-rank completion must not be quietly claimed as better."""
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "track2_lowrank.json")
    if not os.path.exists(p):
        pytest.skip("run src/track2_lowrank.py first")
    t2 = _j.load(open(p))
    for rk, v in t2["by_rank"].items():
        assert v["mae_lowrank"] > v["mae_scheme_mean"], rk
        assert v["mae_lowrank"] > v["mae_zero"], rk


def test_bias_exact_formula_matches_simulation():
    """The one part of Track 3 that holds must keep holding."""
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "bias_correction.json")
    if not os.path.exists(p):
        pytest.skip("run src/bias_correction.py first")
    bc = _j.load(open(p))
    assert bc["exact_formula_validated"] is True
    for pi, v in bc["exact_degradation"].items():
        expect = float(pi) + (1 - float(pi)) * 0.05
        assert abs(v["lower_miss_rate"] - expect) < 1e-9


def test_bias_correction_is_not_claimed_to_work():
    """
    The GPD correction closes <25% of the bias. If someone later makes it
    look better, that must be a deliberate, re-audited change.
    """
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "bias_correction.json")
    if not os.path.exists(p):
        pytest.skip("run src/bias_correction.py first")
    bc = _j.load(open(p))
    for sc, v in bc["scenarios"].items():
        assert v["fraction_of_gap_closed"] < 0.25, sc


def test_corrected_bound_moves_down_as_censoring_rises():
    """Regression for the sign bug the synthetic check caught."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    import bias_correction as BC
    rng = np.random.default_rng(3)
    x = BC.draw_truth(rng, 4000, "mixture")
    x = x[x > np.quantile(x, 0.15)]
    bounds = [BC.corrected_lower(x, pi)[0] for pi in (0.05, 0.10, 0.20, 0.30)]
    assert all(b2 < b1 for b1, b2 in zip(bounds, bounds[1:])), bounds


# --------------------------------------------- GPU-result ingest + doc checks
def test_adversarial_validator_catches_format_errors(tmp_path):
    import adversarial_schema as A
    good = pd.DataFrame([{
        "model": "local/m1", "base_model": "SmolLM-135M-Instruct",
        "scheme": "w2a16", "recipe": "2-bit, no calibration",
        "params_b": 0.135, "benchmark": "arc_challenge",
        "acc_before": 37.2, "acc_after": 25.1,
    }])
    p = tmp_path / "good.csv"
    good.to_csv(p, index=False)
    ok, probs, warns, _ = A.validate(str(p))
    assert ok and not probs

    # 0-1 scale must be rejected (the day-1 bug)
    bad = good.copy()
    bad[["acc_before", "acc_after"]] /= 100
    p2 = tmp_path / "scale.csv"
    bad.to_csv(p2, index=False)
    ok2, probs2, _, _ = A.validate(str(p2))
    assert not ok2 and any("0-1 scale" in x for x in probs2)

    # unknown benchmark must be rejected (the MATH-500 bug)
    b3 = good.copy()
    b3["benchmark"] = "MATH-500"
    p3 = tmp_path / "bench.csv"
    b3.to_csv(p3, index=False)
    ok3, probs3, _, _ = A.validate(str(p3))
    assert not ok3 and any("unknown benchmark" in x for x in probs3)

    # missing required column
    p4 = tmp_path / "miss.csv"
    good.drop(columns=["recipe"]).to_csv(p4, index=False)
    ok4, probs4, _, _ = A.validate(str(p4))
    assert not ok4 and any("missing required" in x for x in probs4)


def test_adversarial_validator_warns_when_data_is_not_adversarial(tmp_path):
    import adversarial_schema as A
    mild = pd.DataFrame([{
        "model": "local/m1", "base_model": "X", "scheme": "w2a16",
        "recipe": "mild", "params_b": 0.135, "benchmark": "mmlu",
        "acc_before": 40.0, "acc_after": 39.9,
    }])
    p = tmp_path / "mild.csv"
    mild.to_csv(p, index=False)
    ok, _, warns, _ = A.validate(str(p))
    assert ok
    assert any("not damaging enough" in w for w in warns)


def test_docs_are_cross_consistent():
    import check_docs
    assert check_docs.main() == 0


# ------------------------------------------------- session 2: 5-shot protocol
def test_at_chance_models_excluded_from_mmlu_analysis():
    """
    SmolLM-135M sits at 25.57 on a 4-way MMLU (chance 25.0). Its deltas are
    noise and diluted the control estimate; it must stay excluded.
    """
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "session2_5shot.json")
    if not os.path.exists(p):
        pytest.skip("run the 5-shot analysis first")
    q = _j.load(open(p))
    assert "SmolLM-135M-Instruct" in q["at_chance_excluded"]
    base = q["base_mmlu"]["SmolLM-135M-Instruct"]
    se = 100 * np.sqrt(0.25 * 0.75 / 3000)
    assert (base - 25.0) / se < 3.0, "SmolLM is no longer at chance"


def test_five_shot_reproduces_published_base_accuracy():
    """The whole point of session 2: harness alignment."""
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "session2_5shot.json")
    if not os.path.exists(p):
        pytest.skip("run the 5-shot analysis first")
    q = _j.load(open(p))
    for m, pub in q["published_mmlu_targets"].items():
        ours = q["base_mmlu"][m]
        assert abs(ours - pub) < 2.0, f"{m}: {ours} vs published {pub}"


def test_control_arm_weakness_is_recorded():
    """
    Our 'correct' GPTQ is materially worse than production. If that stops
    being true the anchors change meaning, so it must fail loudly.
    """
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "session2_5shot.json")
    if not os.path.exists(p):
        pytest.skip("run the 5-shot analysis first")
    q = _j.load(open(p))
    assert q["implementation_gap_x"] > 2.0
    assert q["control_mean_5shot"] < q["published_w4a16_mmlu_mean"]


def test_correction_is_not_presented_as_a_single_number():
    """Anchor choice moves the bound materially; no point estimate allowed."""
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "correction_protocol_compare.json")
    if not os.path.exists(p):
        pytest.skip("run the protocol comparison first")
    c = _j.load(open(p))
    spread = abs(c["0.2"]["anchors_5shot"] - c["0.2"]["anchors_0shot"])
    assert spread > 1.0, "if anchors agree, revisit the no-point-estimate rule"


def test_control_gap_is_measured_against_matched_models():
    """
    The control-vs-production gap must be computed head-to-head on the same
    checkpoints, not against a published mean pooled over all model sizes.
    Pooling inflated it from 2.6x to 5.1x in an earlier draft.
    """
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "control_vs_published.json")
    if not os.path.exists(p):
        pytest.skip("run the matched comparison first")
    q = _j.load(open(p))
    assert q["matched_rows"] >= 2
    for r in q["per_model"]:
        # base accuracies must agree, or the comparison is not like-for-like
        assert r["base_acc_agreement_pp"] < 2.0, r
        # and the gap must be significant, not noise
        assert abs(r["z"]) > 2.0, r
    assert 1.5 < q["mean_ratio_x"] < 4.0, q["mean_ratio_x"]


def test_control_gap_has_an_explanation():
    """An unexplained degradation is as suspect as an unexplained improvement."""
    import json as _j
    p = os.path.join(os.path.dirname(__file__), "..", "out",
                     "control_vs_published.json")
    if not os.path.exists(p):
        pytest.skip("run the matched comparison first")
    q = _j.load(open(p))
    assert q["calib_ratio_x"] > 50, "calibration shortfall should be large"


# ------------------------------------------------------------- feedback loop
def test_feedback_never_renders_an_unconfigured_link(tmp_path, monkeypatch):
    """A dead link is worse than no link."""
    import feedback as F
    monkeypatch.setattr(F, "CONFIG", str(tmp_path / "nope.json"))
    dest, ok = F.destination()
    assert ok is False
    assert F.prompt_line() == [], "must print nothing when unconfigured"


def test_feedback_prompt_appears_once_configured(tmp_path, monkeypatch):
    import json as _j
    import feedback as F
    cfg = tmp_path / "c.json"
    cfg.write_text(_j.dumps({"form_url": "https://example.test/form"}))
    monkeypatch.setattr(F, "CONFIG", str(cfg))
    dest, ok = F.destination()
    assert ok and dest == "https://example.test/form"
    assert any("example.test" in ln for ln in F.prompt_line())


def test_refused_prompt_is_stronger_and_always_present(tmp_path, monkeypatch):
    """The ask on a refused cell must appear even with no destination."""
    import feedback as F
    monkeypatch.setattr(F, "CONFIG", str(tmp_path / "nope.json"))
    lines = F.refused_prompt(["w4a16"])
    assert lines, "refused cells must always carry the ask"
    assert any("help most" in ln for ln in lines)


def test_request_log_is_append_only_and_local(tmp_path, monkeypatch):
    import feedback as F
    log = tmp_path / "requests.log"
    monkeypatch.setattr(F, "REQUEST_LOG", str(log))
    F.log_request(["w4a16"], band="<2B", refused=["w4a16|<2B"])
    F.log_request(["fp8"], band=">10B")
    rows = log.read_text().strip().split("\n")
    assert len(rows) == 3          # header + 2
    assert "w4a16|<2B" in rows[1]
    assert rows[0].startswith("timestamp_utc")


def test_logging_failure_never_breaks_the_tool(monkeypatch):
    import feedback as F
    monkeypatch.setattr(F, "REQUEST_LOG", "/nonexistent-dir/x/y.log")
    F.log_request(["w4a16"])        # must not raise


def test_form_spec_covers_what_the_validator_requires():
    """Every required schema field must be collectable from the form."""
    import feedback as F
    import adversarial_schema as A
    names = {n for n, _, _ in F.form_spec()}
    assert "quantization_scheme" in names
    assert "accuracy_before" in names and "accuracy_after" in names
    assert "benchmark" in names and "model_size_params_b" in names
    # the form must ask for harness/shots, the confound that bit us
    assert "eval_harness_and_shots" in names
    assert len(A.REQUIRED) >= 5


def test_no_hand_typed_numbers_in_outward_facing_docs():
    """
    The standing rule, enforced at the strongest level: every figure a reader
    sees in a doc that could be sent out must trace to the registry, or sit in
    a named exempt category (axis inputs, external citations, superseded
    figures quoted as such, model-name digits, structural text).
    """
    import audit_traceability
    assert audit_traceability.main() == 0


# --------------------------------------------------------------- disk guard
def test_diskguard_refuses_impossible_download():
    """The guard must refuse, not warn, when space is short.

    Two downloads filled the disk on 2026-09-23 before this existed. The
    failure mode that matters is proceeding anyway, so this asserts the
    refusal is an exception rather than a printed message.
    """
    import diskguard
    with pytest.raises(SystemExit) as e:
        diskguard.require_free_gb(10_000_000, "an impossible download")
    assert "REFUSING TO START" in str(e.value)
    assert "Nothing has been downloaded" in str(e.value)


def test_diskguard_allows_when_space_is_ample():
    import diskguard
    assert diskguard.require_free_gb(0.0, "a zero-byte download") > 0


def test_download_scripts_are_guarded():
    """Every script that downloads must consult the guard first."""
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("src/publisher_census.py", "gpu/lora_forgetting.py"):
        p = os.path.join(here, rel)
        if not os.path.exists(p):
            continue
        src = open(p).read()
        assert "diskguard" in src, f"{rel} downloads but does not check disk"
        assert "require_free_gb" in src, f"{rel} imports guard but never calls it"


def test_no_unescaped_pipe_in_table_claim_tags():
    """Claim tags inside markdown TABLE rows must escape the pipe.

    GFM splits a table row on any unescaped pipe, including one inside an HTML
    comment, which breaks the table and prints the raw tag on github.com.
    python-markdown renders it correctly, so a local PDF check does not catch
    this -- hence a test.
    """
    import glob, os, re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad_key = re.compile(r"claim:\s*[^\s]*?(?<!\\)\|")
    offenders = []
    for f in glob.glob(os.path.join(root, "*.md")):
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            if line.startswith("|") and bad_key.search(line):
                offenders.append(f"{os.path.basename(f)}:{i}")
    assert not offenders, (
        "unescaped pipe in a claim tag inside a table row: "
        + ", ".join(offenders[:5]))


def test_cluster_bootstrap_bounds_do_not_depend_on_the_random_stream():
    """With few clusters the bootstrap is discrete and its 5th percentile can sit
    on an atom boundary; the bounds must be exact, not seed-dependent."""
    import numpy as np, cluster_boot
    ok, n = [50.0, 30.0, 45.0, 20.0], [50.0, 33.0, 48.0, 40.0]
    ref = cluster_boot.bounds(ok, n)
    for seed in range(20):
        assert cluster_boot.bounds(ok, n, rng=np.random.default_rng(seed)) == ref
    # exact value check on a case small enough to reason about: two clusters
    lo, hi = cluster_boot.bounds([0.0, 10.0], [10.0, 10.0])
    assert (lo, hi) == (0.0, 100.0)


def test_qwen35_published_rows_are_kept_separate_from_the_corpus():
    """Qwen3.5 is a different architecture and distillation recipe: its rows may be
    compared with Qwen2.5's but must never enter (or overlap) the main corpus."""
    import csv
    root = os.path.join(os.path.dirname(__file__), "..")
    q = list(csv.DictReader(open(os.path.join(root, "data", "qwen35", "published_rows.csv"))))
    d = list(csv.DictReader(open(os.path.join(root, "data", "dataset.csv"))))
    assert q and all(r["family"] == "qwen3.5" for r in q)
    assert not any("qwen3.5" in r["model"].lower() or r["family"] == "qwen3.5" for r in d)
    assert {r["model"] for r in q}.isdisjoint({r["model"] for r in d})
    assert all(r["verified"] == "1" for r in q)          # every row passed the recovery gate
    assert {r["moe"] for r in q} == {"0", "1"}           # MoE flagged, not silently mixed in


def test_qwen35_ingestion_only_reads_quantized_qwen35_repos():
    import qwen35_published as Q
    ids = Q.repos()
    assert ids and all("Qwen3.5-" in i and "speculator" not in i for i in ids)


def test_external_feedback_doc_cites_only_real_commits():
    """Every commit hash in EXTERNAL_FEEDBACK.md must resolve in git history,
    so the trail it claims can actually be followed. Skipped outside a git
    checkout."""
    import re, subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    if not os.path.isdir(os.path.join(root, ".git")):
        return
    text = open(os.path.join(root, "EXTERNAL_FEEDBACK.md")).read()
    hashes = set(re.findall(r"`([0-9a-f]{7})`", text))
    assert hashes, "the doc should cite commits"
    for h in hashes:
        r = subprocess.run(["git", "cat-file", "-t", h], cwd=root,
                           capture_output=True, text=True)
        assert r.stdout.strip() == "commit", f"{h} is not a commit"


def test_fair_variance_uses_only_fully_observed_cells():
    """The reviewer-suggested variance measure must fill nothing: a rank-1
    matrix with holes still reads as rank-1 on its fully observed part, and a
    noisy column outside the chosen submatrix must not leak in."""
    import numpy as np, pandas as pd
    import track2_lowrank as t
    rng = np.random.default_rng(0)
    u = rng.normal(size=(30, 1))
    X = u @ np.array([[1.0, 2.0, -1.0]]) + 5.0
    X = np.column_stack([X, rng.normal(size=30)])
    X[:10, 3] = np.nan
    M = pd.DataFrame(X, index=[f"BASE::m{i}" for i in range(30)],
                     columns=list("abcd"))
    n, cols, ve = t.fair_variance_explained(M, 3, rank=1)
    assert n == 30 and set(cols) == {"a", "b", "c"}
    assert ve > 0.999
