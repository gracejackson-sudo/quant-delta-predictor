"""Emit paper/numbers.tex from the claims registry.

Every figure in the paper is a macro defined here, so no number in main.tex is
hand-typed. Changing the data and re-running the pipeline updates the paper.
"""
from __future__ import annotations
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from verify_claims import registry  # noqa: E402

R = {k: v[0] for k, v in registry().items()}

# macro name -> (registry key, format)
M = {
    "NrowsRaw": ("n_rows_raw", "{:.0f}"), "NrowsDropped": ("n_rows_dropped_near_chance", "{:.0f}"),
    "MinAccBefore": ("min_acc_before_pct", "{:.0f}"),
    "Nrows": ("n_rows", "{:.0f}"), "Nckpt": ("n_checkpoints", "{:.0f}"),
    "Nfam": ("n_families", "{:.0f}"), "Nsch": ("n_schemes", "{:.0f}"),
    "MinCkpt": ("min_cell_checkpoints", "{:.0f}"),
    "NpredNums": ("n_numbers_in_predictor", "{:.0f}"),
    "AdvWorst": ("adv_worst_delta", "{:.1f}"),
    "AdvBadRows": ("adv_bad_rows", "{:.0f}"),
    "AdvBadOver": ("adv_bad_over_3pp", "{:.0f}"),
    "AdvCtrlRows": ("adv_control_rows", "{:.0f}"),
    "AdvCtrlOver": ("adv_control_over_3pp", "{:.0f}"),
    "AdvMaxB": ("adv_max_params_b", "{:.1f}"),
    "CtrlRatio": ("ctrl_ratio_x", "{:.1f}"),
    "CtrlGap": ("ctrl_mean_gap", "{:+.2f}"),
    "CalibRatio": ("calib_ratio_x", "{:.0f}"),
    "CtrlZmin": ("ctrl_min_z", "{:.1f}"), "CtrlZmax": ("ctrl_max_z", "{:.1f}"),
    "BaseQwenSmall": ("s2_base_qwen05", "{:.2f}"),
    "PubQwenSmall": ("s2_pub_qwen05", "{:.2f}"),
    "PubQwenSmallBase": ("s2_pub_qwen05_base", "{:.2f}"),
    "NvfpMarginal": ("nvfp_marginal_coverage_pct", "{:.0f}"),
    "AnomQuantMinPct": ("t1_anom_min_pct", "{:.0f}"),
    "AnomQuantMaxPct": ("t1_anom_max_pct", "{:.0f}"),
    # RedHatAI card recipe-documentation scan (date-suffixed).
    "NRhCardsListed": ("n_rh_cards_listed_2026_09_26", "{:.0f}"),
    "NRhCardsHarvested": ("n_rh_cards_harvested", "{:.0f}"),
    "NRhCardsWithLibVersion": (
        "n_rh_cards_with_lib_version_2026_09_26", "{:.0f}"),
    "NRhCardsWithActOrder": (
        "n_rh_cards_with_act_order_2026_09_26", "{:.0f}"),
    "NRhCardsWithDamp": ("n_rh_cards_with_damp_2026_09_26", "{:.0f}"),
    "NRhCardsNoRecipe": ("n_rh_cards_no_recipe_2026_09_26", "{:.0f}"),
    "NRhHarvestedWithLibVersion": (
        "n_rh_harvested_with_lib_version_2026_09_26", "{:.0f}"),
    "NRhHarvestedNoRecipe": (
        "n_rh_harvested_no_recipe_2026_09_26", "{:.0f}"),
    "BaseQwenMid": ("s2_base_qwen15", "{:.2f}"),
    "PubQwenMid": ("s2_pub_qwen15", "{:.2f}"),
    "ProtoSens": ("s2_protocol_sensitivity", "{:.2f}"),
    "ExcessN": ("s2_excess_n", "{:.0f}"),
    "IntervalsExclZero": ("intervals_excluding_zero", "{:.0f}"),
    "BigLossBelow": ("n_big_losses_below_floor", "{:.0f}"),
    "BigLoss": ("n_big_losses", "{:.0f}"),
    "CellsBlocked": ("n_cells_blocked_from_tier_a", "{:.0f}"),
    "PooledOneSided": ("pooled_one_sided_pct", "{:.1f}"),
    "PubOthersVerifiableCards": ("pub_others_verifiable_cards", "{:.0f}"),
    "ProspInside": ("prosp_inside", "{:.0f}"), "ProspN": ("prosp_n", "{:.0f}"),
    "ProspCov": ("prosp_cov_pct", "{:.1f}"),
    "ProspCILo": ("prosp_ci_lo", "{:.1f}"), "ProspCIHi": ("prosp_ci_hi", "{:.1f}"),
    "MaeGain": ("mae_gain_pp", "{:.2f}"),
    "TargetMean": ("target_mean_pp", "{:.2f}"), "TargetSd": ("target_sd_pp", "{:.2f}"),
    "NoiseNBench": ("noise_n_bench", "{:.0f}"), "NoiseNThird": ("noise_n_third", "{:.0f}"),
    "NoiseMinPct": ("noise_min_pct", "{:.0f}"), "GsmNoiseSd": ("noise_gsm8k_sd_pp", "{:.1f}"),
    "BandEmpCov": ("band_emp_coverage_pct", "{:.1f}"),
    "BandConfCov": ("band_conf_coverage_pct", "{:.1f}"),
    "InfShare": ("inf_share_pct", "{:.0f}"),
    "InfSharePct": ("inf_share_pct", "{:.1f}"),
    "FiniteConfCov": ("finite_conf_cov_pct", "{:.1f}"),
    "FiniteEmpCov": ("finite_emp_cov_pct", "{:.1f}"),
    "FiniteConfWidth": ("finite_conf_width", "{:.2f}"),
    "FiniteEmpWidth": ("finite_emp_width", "{:.2f}"),
    "FiniteN": ("finite_n", "{:.0f}"),
    "PubIntelPaired": ("pub_intel_paired_cards", "{:.0f}"),
    "NCellsInsuff": ("n_cells_insufficient_evidence", "{:.0f}"),
    "NCellsTotal": ("n_cells_total", "{:.0f}"),
    "NCellsRefused": ("n_cells_refused", "{:.0f}"),
    "PooledRows": ("pooled_distinct_rows", "{:.0f}"), "PairsPerRow": ("pooled_pairs_per_row", "{:.0f}"),
    "PooledPairs": ("pooled_scored_pairs", "{:.0f}"),
    "NSchemesInsuff": ("n_schemes_insufficient_evidence", "{:.0f}"),
    "NSchemesRefused": ("n_schemes_refused", "{:.0f}"),
    "ProspSchemes": ("prosp_n_schemes", "{:.0f}"), "ProspAllRows": ("prosp_all_rows", "{:.0f}"),
    "ProspAllCov": ("prosp_all_cov_pct", "{:.1f}"),
    "ProspClusLo": ("prosp_clus_lo", "{:.1f}"), "ProspClusHi": ("prosp_clus_hi", "{:.1f}"),
    "ProspClusN": ("prosp_clus_n", "{:.0f}"),
    "ProspRowsWfour": ("prosp_scheme_rows::w4a16", "{:.0f}"),
    "ProspCkptsWfour": ("prosp_scheme_ckpts::w4a16", "{:.0f}"),
    "ProspCovWfour": ("prosp_scheme_cov_pct::w4a16", "{:.1f}"),
    "ProspRowsWfourShare": ("prosp_wfour_share_pct", "{:.0f}"),
    # Family-split coverage on the strict prospective set (Sec 3 definition).
    "ProspInFamRows": ("prosp_in_family_rows", "{:.0f}"),
    "ProspInFamInside": ("prosp_in_family_inside", "{:.0f}"),
    "ProspInFamCov": ("prosp_in_family_cov_pct", "{:.1f}"),
    "ProspInFamCkpts": ("prosp_in_family_ckpts", "{:.0f}"),
    "ProspOutFamRows": ("prosp_out_family_rows", "{:.0f}"),
    "ProspOutFamInside": ("prosp_out_family_inside", "{:.0f}"),
    "ProspOutFamCov": ("prosp_out_family_cov_pct", "{:.1f}"),
    "ProspOutFamCkpts": ("prosp_out_family_ckpts", "{:.0f}"),
    "ProspGemThreeRows": ("prosp_gemma3_rows", "{:.0f}"),
    "ProspGemThreeInside": ("prosp_gemma3_inside", "{:.0f}"),
    "ProspGemThreeCov": ("prosp_gemma3_cov_pct", "{:.1f}"),
    "ProspGemThreeCkpts": ("prosp_gemma3_ckpts", "{:.0f}"),
    "ProspGemThreeGap": ("prosp_gemma3_gap_pp", "{:.1f}"),
    "ProspOutSmallRows": ("prosp_outfam_small_rows", "{:.0f}"),
    "ProspOutSmallCkpts": ("prosp_outfam_small_ckpts", "{:.0f}"),
    "ProspOutSmallPullup": ("prosp_outfam_small_pullup_pp", "{:.1f}"),
    # Round item 2: the checkpoint-level headline, the ICC explanation that
    # has to sit beside it, and the narrowing decomposition.
    "ProspCrLo": ("prosp_cr_lo", "{:.1f}"),
    "ProspCrHi": ("prosp_cr_hi", "{:.1f}"),
    "ProspCrSe": ("prosp_cr_se_pp", "{:.2f}"),
    "ProspNaiveSe": ("prosp_naive_se_pp", "{:.2f}"),
    "ProspSeRatio": ("prosp_se_ratio", "{:.3f}"),
    "ProspIcc": ("prosp_icc", "{:+.4f}"),
    "ProspDeff": ("prosp_design_effect", "{:.3f}"),
    "ProspRowsPerCkptMax": ("prosp_rows_per_ckpt_max", "{:.0f}"),
    "ProspCkptsWithMiss": ("prosp_ckpts_with_miss", "{:.0f}"),
    "ProspCkptsClean": ("prosp_ckpts_clean", "{:.0f}"),
    "ProspWorstCkptMisses": ("prosp_worst_ckpt_misses", "{:.0f}"),
    "ProspTotalMisses": ("prosp_total_misses", "{:.0f}"),
    "ProspWorstBenchMisses": ("prosp_worst_bench_misses", "{:.0f}"),
    "ProspBenchWithMiss": ("prosp_bench_with_miss", "{:.0f}"),
    "ProspWidthCp": ("prosp_width_cp", "{:.2f}"),
    "ProspWidthWald": ("prosp_width_wald", "{:.2f}"),
    "ProspWidthCr": ("prosp_width_cr", "{:.2f}"),
    "ProspNarrowEst": ("prosp_narrowing_estimator", "{:.2f}"),
    "ProspNarrowUnit": ("prosp_narrowing_unit", "{:.2f}"),
    "ProspDiffLo": ("prosp_infam_outfam_diff_lo", "{:+.0f}"),
    "ProspDiffHi": ("prosp_infam_outfam_diff_hi", "{:+.0f}"),
    # Tier 0.1 disclosure: per-subgroup coverage of the held-out
    # llama-3 fold, 3.1/3.2/3.3 x calibration family. §6 subgroup table.
    "LThreeSubOneGem": ("lofo_l3sub_cov::3.1::gemma-2", "{:.1f}"),
    "LThreeSubOneGra": ("lofo_l3sub_cov::3.1::granite", "{:.1f}"),
    "LThreeSubOneMis": ("lofo_l3sub_cov::3.1::mistral", "{:.1f}"),
    "LThreeSubOneQtf": ("lofo_l3sub_cov::3.1::qwen2.5", "{:.1f}"),
    "LThreeSubOneQth": ("lofo_l3sub_cov::3.1::qwen3", "{:.1f}"),
    "LThreeSubTwoGem": ("lofo_l3sub_cov::3.2::gemma-2", "{:.1f}"),
    "LThreeSubTwoGra": ("lofo_l3sub_cov::3.2::granite", "{:.1f}"),
    "LThreeSubTwoMis": ("lofo_l3sub_cov::3.2::mistral", "{:.1f}"),
    "LThreeSubTwoQtf": ("lofo_l3sub_cov::3.2::qwen2.5", "{:.1f}"),
    "LThreeSubTwoQth": ("lofo_l3sub_cov::3.2::qwen3", "{:.1f}"),
    "LThreeSubThreeGem": ("lofo_l3sub_cov::3.3::gemma-2", "{:.1f}"),
    "LThreeSubThreeGra": ("lofo_l3sub_cov::3.3::granite", "{:.1f}"),
    "LThreeSubThreeMis": ("lofo_l3sub_cov::3.3::mistral", "{:.1f}"),
    "LThreeSubThreeQtf": ("lofo_l3sub_cov::3.3::qwen2.5", "{:.1f}"),
    "LThreeSubThreeQth": ("lofo_l3sub_cov::3.3::qwen3", "{:.1f}"),
    "LThreeMergedGem": ("lofo_l3merged_cov::gemma-2", "{:.1f}"),
    "LThreeMergedGra": ("lofo_l3merged_cov::granite", "{:.1f}"),
    "LThreeMergedMis": ("lofo_l3merged_cov::mistral", "{:.1f}"),
    "LThreeMergedQtf": ("lofo_l3merged_cov::qwen2.5", "{:.1f}"),
    "LThreeMergedQth": ("lofo_l3merged_cov::qwen3", "{:.1f}"),
    # Tier 1.1 LOFO fallback disclosure (§9).
    "LofoFallbackPairs": ("lofo_fallback_pairs", "{:.0f}"),
    "LofoTotalPairs":    ("lofo_total_pairs",    "{:.0f}"),
    "LofoFallbackRate":  ("lofo_fallback_rate_pct", "{:.1f}"),
    "LofoFallRateFp":    ("lofo_fallback_rate_pct::fp8", "{:.0f}"),
    "LofoFallRateFpdyn": ("lofo_fallback_rate_pct::fp8_dynamic", "{:.0f}"),
    "LofoFallRateNvfp":  ("lofo_fallback_rate_pct::nvfp4", "{:.0f}"),
    "LofoFallRateWfour": ("lofo_fallback_rate_pct::w4a16", "{:.0f}"),
    "LofoFallRateWsixteen": ("lofo_fallback_rate_pct::w8a16", "{:.0f}"),
    "LofoFallRateWint":  ("lofo_fallback_rate_pct::w8a8_int", "{:.0f}"),
    "LofoHeldoutPairs":  ("lofo_heldout_pairs", "{:.0f}"),
    "LofoHeldoutCov":    ("lofo_heldout_cov_pct", "{:.1f}"),
    "LofoHeldoutOneSided": ("lofo_heldout_one_sided_pct", "{:.1f}"),
    "PooledCellCov":       ("pooled_cell_coverage_pct", "{:.1f}"),
    # Tier 1.3 gated-repos disclosure (§9).
    "ProspRegisteredRepos": ("prosp_registered_repos", "{:.0f}"),
    "ProspNameGatedRepos":  ("prosp_name_gated_repos", "{:.0f}"),
    "ProspWithGatedInside": ("prosp_with_gated_inside", "{:.0f}"),
    "ProspWithGatedTotal":  ("prosp_with_gated_total", "{:.0f}"),
    "ProspWithGatedCov":    ("prosp_with_gated_cov_pct", "{:.1f}"),
    "GpqaOtherRows":        ("gpqa_other_rows", "{:.0f}"),
    "MaxBenchmarksPerRun":  ("max_benchmarks_per_run", "{:.0f}"),
    # Tier 2.5 v4 §5 unified disclosure (per-fold spread).
    "LofoSchemeMeanWinsFolds": ("lofo_scheme_mean_wins_folds", "{:.0f}"),
    "LofoSchemeMeanLosesFolds":("lofo_scheme_mean_loses_folds","{:.0f}"),
    "LofoSchemeMeanMin":       ("lofo_scheme_mean_min",       "{:.2f}"),
    "LofoSchemeMeanMax":       ("lofo_scheme_mean_max",       "{:.2f}"),
    "LofoRidgeMin":            ("lofo_ridge_min",             "{:.2f}"),
    "LofoRidgeMax":            ("lofo_ridge_max",             "{:.2f}"),
    "LThreeFoldRows":          ("llama3_fold_rows",           "{:.0f}"),
    "LThreeFoldSharePct":      ("llama3_fold_share_pct",      "{:.1f}"),
    "LThreeHoldoutTrainPct":   ("llama3_holdout_train_share_pct", "{:.1f}"),
    "NonLThreeHoldoutMin":     ("nonllama3_holdout_train_min_pct","{:.0f}"),
    "NonLThreeHoldoutMax":     ("nonllama3_holdout_train_max_pct","{:.0f}"),
    "GemmaTwoFoldRows":        ("gemma2_fold_rows",           "{:.0f}"),
    "QwenThreeFoldRows":       ("qwen3_fold_rows",            "{:.0f}"),
    "NonLThreeFoldRowsMin":    ("nonllama3_fold_rows_min",    "{:.0f}"),
    "NonLThreeFoldRowsMax":    ("nonllama3_fold_rows_max",    "{:.0f}"),
    "NonLThreeFoldShareMin":   ("nonllama3_fold_share_min_pct","{:.1f}"),
    "NonLThreeFoldShareMax":   ("nonllama3_fold_share_max_pct","{:.1f}"),
    "ProspRowsWeightint": ("prosp_scheme_rows::w8a8_int", "{:.0f}"),
    "ProspCkptsWeightint": ("prosp_scheme_ckpts::w8a8_int", "{:.0f}"),
    "ProspCovWeightint": ("prosp_scheme_cov_pct::w8a8_int", "{:.1f}"),
    "ProspRowsFpdyn": ("prosp_scheme_rows::fp8_dynamic", "{:.0f}"),
    "ProspCkptsFpdyn": ("prosp_scheme_ckpts::fp8_dynamic", "{:.0f}"),
    "ProspCovFpdyn": ("prosp_scheme_cov_pct::fp8_dynamic", "{:.1f}"),
    "ProspRowsWeightsixteen": ("prosp_scheme_rows::w8a16", "{:.0f}"),
    "ProspCkptsWeightsixteen": ("prosp_scheme_ckpts::w8a16", "{:.0f}"),
    "ProspCovWeightsixteen": ("prosp_scheme_cov_pct::w8a16", "{:.1f}"),
    "MaeGainLo": ("mae_gain_ci_lo", "{:.2f}"), "MaeGainHi": ("mae_gain_ci_hi", "{:.2f}"),
    "MaeGainFamsPos": ("mae_gain_fams_pos", "{:.0f}"),
    "NoiseMaxPct": ("noise_max_pct", "{:.0f}"), "NoiseNOver": ("noise_n_ge100", "{:.0f}"),
    "NoiseNSmall": ("noise_n_small", "{:.0f}"),
    "RefuseBelow": ("refuse_below_pct", "{:.0f}"),
    "BootLoWsixteenBig": ("os_boot90_lo::w8a16|>10B", "{:.0f}"),
    "BootHiWsixteenBig": ("os_boot90_hi::w8a16|>10B", "{:.0f}"),
    "BootHiWfourSmall": ("cell_boot_hi::w4a16|<2B", "{:.1f}"),
    "PubChecked": ("pub_publishers_checked", "{:.0f}"),
    "PubCards": ("pub_cards_inspected", "{:.0f}"),
    "PubOthersCards": ("pub_others_cards", "{:.0f}"),
    "PubOthersCount": ("pub_others_count", "{:.0f}"),
    "PubOthersPaired": ("pub_others_paired_cards", "{:.0f}"),
    "PubOthersVerified": ("pub_others_verified_rows", "{:.0f}"),
    "PubCtrlVerified": ("pub_control_verified_rows", "{:.0f}"),
}
SCHEMES = ["fp8", "w8a8_int", "fp8_dynamic", "w8a16", "w4a16", "nvfp4"]
CAMEL = {"fp8": "Fpeight", "w8a8_int": "Weightint", "fp8_dynamic": "Fpdyn",
         "w8a16": "Weightsixteen", "w4a16": "Wfour", "nvfp4": "Nvfp"}
PRED = ["scheme_mean", "global_mean", "scheme_x_bench", "ridge", "bench_mean",
        "grad_boost"]
PCAMEL = {"scheme_mean": "Schememean", "global_mean": "Globalmean",
          "scheme_x_bench": "Schemebench", "ridge": "Ridge",
          "bench_mean": "Benchmean", "grad_boost": "Gradboost"}


def main():
    out = ["% GENERATED by paper/gen_numbers.py from the claims registry.",
           "% Do not hand-edit. Re-run the generator instead.", ""]
    miss = []
    for macro, (key, fmt) in M.items():
        if key not in R:
            miss.append(key); continue
        out.append(f"\\newcommand{{\\{macro}}}{{{fmt.format(R[key])}}}")
    for s in SCHEMES:
        c = CAMEL[s]
        for stat, fmt, pre in (("lo", "{:+.2f}", "Lo"), ("hi", "{:+.2f}", "Hi"),
                               ("worst", "{:+.2f}", "Worst"),
                               ("severe_pct", "{:.1f}", "Sev"),
                               ("n", "{:.0f}", "N")):
            k = f"{stat}::{s}"
            if k in R:
                out.append(f"\\newcommand{{\\{pre}{c}}}{{{fmt.format(R[k])}}}")
            else:
                miss.append(k)
    for p in PRED:
        k = f"pred_mae::{p}"
        if k in R:
            out.append(f"\\newcommand{{\\Mae{PCAMEL[p]}}}{{{R[k]:.4f}}}")
        else:
            miss.append(k)
    for cell, mac in (("w4a16|<2B", "CellWfourSmall"),
                      ("w8a16|>10B", "CellWsixteenBig")):
        for pre, key, fmt in (("Cov", "cell_coverage_pct", "{:.1f}"),
                              ("OneSided", "cell_one_sided_pct", "{:.1f}"),
                              ("Ckpts", "cell_ckpts", "{:.0f}"),
                              ("Rows", "cell_rows", "{:.0f}")):
            k = f"{key}::{cell}"
            if k in R:
                out.append(f"\\newcommand{{\\{pre}{mac}}}{{{fmt.format(R[k])}}}")
            else:
                miss.append(k)
    if miss:
        raise SystemExit("missing registry keys: " + ", ".join(sorted(set(miss))))
    names = re.findall(r"\\newcommand\{\\(\w+)\}", "\n".join(out))
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise SystemExit("duplicate macro definitions (LaTeX will refuse): "
                         + ", ".join(dupes))
    p = os.path.join(HERE, "numbers.tex")
    open(p, "w").write("\n".join(out) + "\n")
    print(f"wrote {p}: {len([l for l in out if l.startswith(chr(92)+'newcommand')])} macros")
    return 0


if __name__ == "__main__":
    sys.exit(main())
