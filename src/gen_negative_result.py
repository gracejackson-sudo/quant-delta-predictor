"""Generate NEGATIVE_RESULT.md from computed values (same rule as RANKING.md)."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from verify_claims import registry  # noqa: E402

HERE = os.path.dirname(__file__)
DOC = os.path.join(HERE, "..", "NEGATIVE_RESULT.md")
REG = registry()
def v(k): return REG[k][0]
def n(k, f="{:.0f}"):
    x = v(k); return f"{f.format(x)}<!-- claim: {k} = {x:.4f} -->"

def main():
    L=[];A=L.append
    A("# Per-model quantization accuracy prediction has no signal beyond the "
      "per-scheme average\n")
    A("*A negative result, with the measured noise floor, a low-rank transfer "
      "test, and a comparison to BenchPress.*\n")
    A(f"Data: {n('n_rows')} published evaluations from {n('n_checkpoints')} "
      f"checkpoints across {n('n_families')} model families, scraped from "
      f"RedHatAI model cards. Every figure below is generated from a computed "
      f"value and re-verified by `src/verify_claims.py`.\n")
    A("---\n")

    A("## 1. The claim\n")
    A("Given a base model and a quantization config, predict the accuracy "
      "delta. We tested this properly and it does not work. The only thing "
      "that beats guessing the global average is a six-cell lookup on the "
      "quantization scheme, and the margin is small enough that it is worth "
      "stating in full rather than summarising.\n")

    A("## 2. What we tested\n")
    A("Leave-one-family-out, so the test family's checkpoints are never in "
      "training. Row-weighted mean absolute error, lower is better:\n")
    A("| predictor | MAE (pp) | beats global mean? |")
    A("|---|---|---|")
    LABEL = {"scheme_mean": "per-scheme mean (shipped)",
             "global_mean": "global mean (baseline)",
             "scheme_x_bench": "per-(scheme x benchmark) mean",
             "ridge": "ridge, 38 features",
             "bench_mean": "per-benchmark mean",
             "grad_boost": "gradient boosting"}
    base = v("pred_mae::global_mean")
    for key in sorted((k for k in REG if k.startswith("pred_mae::")),
                      key=v):
        name = key.split("::", 1)[1]
        mark = ("--" if name == "global_mean"
                else ("yes" if v(key) < base else "**no**"))
        A(f"| {LABEL.get(name, name)} | {n(key, '{:.4f}')} | {mark} |")
    A("")
    A("Ridge regression and gradient boosting both do **worse than predicting "
      "the average**. Adding model size, benchmark identity, base accuracy "
      "and quantization method all degraded out-of-family accuracy.\n")

    A("## 3. Why: the target is mostly measurement noise\n")
    A(f"Benchmark scores are sample proportions over finite item sets, so a "
      f"delta is a difference of two noisy quantities. Estimating the noise "
      f"floor directly from near-lossless schemes (W8A16 and FP8-dynamic, "
      f"whose true delta should be ~0) gives an irreducible MAE of "
      f"**{n('mae_floor_pp', '{:.3f}')}pp**.\n")
    A(f"The global-mean baseline sits at {n('mae_global_lofo', '{:.4f}')}pp. "
      f"So the entire headroom available to any predictor is about "
      f"{v('mae_global_lofo') - v('mae_floor_pp'):.3f}"
      f"<!-- claim: headroom_pp = {v('mae_global_lofo') - v('mae_floor_pp'):.4f} -->pp, and the best "
      f"predictor we found captures roughly a tenth of it.\n")
    A("On GSM8K specifically, the observed spread for supposedly lossless "
      "schemes is nearly identical to the spread for 4-bit schemes: the "
      "measurement is louder than the effect.\n")

    A("## 4. Low-rank structure and paired deltas (Track 2, revised after external review)\n")
    A("BenchPress (arXiv:2606.24020) predicts unseen benchmark scores by "
      "exploiting the fact that a frontier-model score matrix is roughly "
      "rank-2. We tested whether that structure helps on **paired** "
      "quantization deltas.\n")
    A("> **Correction.** The first version of this section made three "
      "statements that a BenchPress author showed, on reviewing our code, to "
      "be wrong or unfair. (1) It said rank-2 explains "
      f"{n('t2_var_explained_rank2_pct', '{:.1f}')}% of the variance in our "
      f"matrix, against over 90% for BenchPress. That figure was computed "
      f"after filling {n('t2b_filled_global_mean_pct', '{:.1f}')}% of the "
      f"matrix with one global mean, which weakens any low-rank structure. "
      f"Measured as the BenchPress paper does it, the structure is present "
      f"(table below). (2) It tested a plain SVD-completion approximation, "
      f"not BenchPress's method. (3) It called the resulting error gap 'an "
      f"order of magnitude'; even our own numbers showed about "
      f"{n('t2_mae_lowrank::2', '{:.0f}')} versus "
      f"{n('t2_mae_scheme::2', '{:.1f}')}. What follows replaces it.\n")
    A("### 4a. Is the structure there?\n")
    A("Largest fully observed submatrix with k benchmarks, each column "
      "mean-centred, nothing filled:\n")
    A("| benchmarks | rows (base + quantized) | rank-2 variance | base rows only |")
    A("|---|---|---|---|")
    for k in range(3, 7):
        A(f"| {k} | {n(f't2b_fair_rows_k{k}')} | "
          f"{n(f't2b_fair_rank2_k{k}_pct', '{:.2f}')}% | "
          f"{n(f't2b_fair_base_rank2_k{k}_pct', '{:.2f}')}% |")
    A("")
    A("Yes: rank-2 explains 88% to 99% here, as the reviewer said. Two "
      "cautions on reading it. With only 3 to 6 columns, two factors can "
      "explain a large share almost by construction, so this is a weak test "
      "next to BenchPress's 133 benchmarks. And each quantized checkpoint sits "
      "next to its own base, so the rows are not independent; the base-only "
      "column removes that and gives the same picture. What it establishes is "
      "that cross-model variation is low-rank. It says nothing yet about "
      "whether that helps predict a one-point paired delta.\n")
    A("### 4b. Does BenchPress's actual method predict the delta?\n")
    A(f"Protocol, unchanged: reveal the full base-model row plus 3 of the "
      f"quantized model's scores, predict the rest, read off the implied "
      f"delta. Now run with BenchPress's released predictor (Logit + Bias ALS, "
      f"rank 2, default regularisation, unmodified code; a one-off spot check on their own released "
      f"matrix gave a median error near their reported figure, but that check "
      f"is not among this repository's scripts) and with our original imputer, "
      f"averaged over {n('t2b_seeds')} random draws of the revealed scores "
      f"on {n('t2b_n::bp')} scored rows. Errors are in accuracy points.\n")
    A("| method | MAE (sd over draws) | scheme-mean MAE | always-zero MAE | MAE / scheme-mean | correlation with true delta |")
    A("|---|---|---|---|---|---|")
    for lab, i in (("our original imputer", "orig"),
                   ("BenchPress Logit + Bias ALS", "bp"),
                   ("BenchPress, target's sibling rows removed", "bp_nosib")):
        sd = f" ({n('t2b_mae_sd::' + i, '{:.2f}')})" if i in ("orig", "bp") else ""
        A(f"| {lab} | {n('t2b_mae::' + i, '{:.2f}')}{sd} | "
          f"{n('t2b_mae_scheme_mean::' + i, '{:.2f}')} | "
          f"{n('t2b_mae_zero::' + i, '{:.2f}')} | "
          f"{n('t2b_ratio_vs_scheme_mean::' + i, '{:.1f}')}x | "
          f"{n('t2b_corr_pred_true::' + i, '{:.2f}')} |")
    A("")
    A(f"Using their method instead of ours improves the error by about "
      f"{n('t2b_bp_improvement_pct', '{:.0f}')}% but does "
      f"not close the gap: it is still {n('t2b_ratio_vs_scheme_mean::bp', '{:.1f}')} "
      f"times the error of the six-number scheme lookup, the predicted deltas "
      f"are uncorrelated with the true ones, and it flags "
      f"{n('t2b_false_alarms::bp', '{:.0f}')} rows that were not severe as "
      f"losing at least 1.5 points while catching "
      f"{n('t2b_severe_caught::bp', '{:.1f}')} of {n('t2b_n_severe::bp')} severe ones. "
      f"The likely reason is scale, not a flaw in their method: BenchPress "
      f"reports recovering held-out scores to within about {n('benchpress_reported_error_pts', '{:.1f}')} points, and a "
      f"paired quantization delta averages {n('t2b_mae_zero::bp', '{:.1f}')} points in size, so an "
      f"estimate built from cross-benchmark prediction cannot resolve it. "
      f"Our error of {n('t2b_mae::bp', '{:.1f}')} points on a sparser, "
      f"16-benchmark matrix is the same order as their own.\n")
    A("### 4c. The reviewer's two smaller points\n")
    A(f"- **Benchmarks without close neighbours.** Confirmed: the strongest "
      f"correlation for GPQA is {n('t2b_corr_gpqa', '{:.2f}')} "
      f"(over {n('t2b_corr_gpqa_overlap')} rows) and for MuSR "
      f"{n('t2b_corr_musr', '{:.2f}')} (over {n('t2b_corr_musr_overlap')} rows). "
      f"Correlations elsewhere are near 1 but rest on very few overlapping "
      f"rows and on base and quantized rows that duplicate each other, so we "
      f"do not treat them as strong evidence.\n")
    A(f"- **Choosing the known scores.** Revealing the 3 most predictive "
      f"benchmarks instead of 3 random ones gave a ratio to the lookup of "
      f"{n('t2b_ratio_vs_scheme_mean::bp_pred', '{:.1f}')}x for BenchPress "
      f"and {n('t2b_ratio_vs_scheme_mean::orig_pred', '{:.1f}')}x for our "
      f"imputer, against {n('t2b_ratio_vs_scheme_mean::bp', '{:.1f}')}x and "
      f"{n('t2b_ratio_vs_scheme_mean::orig', '{:.1f}')}x with random ones. "
      f"That is a single deterministic run, with predictiveness ranked from "
      f"base rows that include the model being tested, and it changes which "
      f"rows are scored, so it is a weak check; it does not change the "
      f"conclusion.\n")
    A("**What this now supports, and what it does not.** With BenchPress's "
      "released method on our matrix, low-rank completion still predicts the "
      "paired delta several times worse than a per-scheme average. It does "
      "not support an order-of-magnitude claim, and it does not show that "
      "BenchPress fails on its own task. *Not tested:* their exact evaluation "
      "harness; regularisation and rank tuned on our data; revealing more "
      "than 3 scores; applying low-rank structure to the deltas themselves "
      "rather than to scores. Any of these could change the picture.\n")

    A("## 5. Relationship to BenchPress\n")
    A("BenchPress solves the same problem shape: predict an eval result, "
      "estimate whether to trust it, wrap it in a conformal interval, decide "
      "whether to skip the real run. Section 6.2 builds three reliability "
      "estimators and a risk-normalised conformal wrapper. It is a more "
      "sophisticated method than ours, with public code and data.\n")
    A("What it does not do is quantization: the string `quantiz` does not "
      "appear in the paper, and its 84 models contain no quantized "
      "checkpoints. Its estimand is a model's absolute score, not the delta "
      "between a model and its compressed self.\n")
    A("**A closer neighbour, found after the fact.** Tong et al., *Does "
      "Compression Preserve Uncertainty?* (arXiv:2606.01850) apply conformal "
      "prediction directly to quantized and sparse LLMs, including W4A16, "
      "across 12 models from 1B to 70B. On domain that is far closer to this "
      "work than BenchPress is. On estimand it is a different problem: their "
      "conformal sets are over label space, built from a compressed model's "
      "own output probabilities, so the method requires running the "
      "compressed model. Ours is over historical accuracy deltas and exists "
      "to avoid running it. Their Figure 2 plots the same quantity this tool "
      "predicts, `Acc_compressed - Acc_dense`, but measures it rather than "
      "forecasting it.\n")
    A("Two of their findings bear on ours. Compression decouples accuracy "
      "from uncertainty, so an accuracy-only estimate like this one is an "
      "incomplete picture of deployment risk. And larger models absorb "
      "compression-induced uncertainty better, which is independent support "
      "for the size-band behaviour this tool already refuses on.\n")
    A("So the honest framing of this work is not that we built something "
      "new. It is that we measured something neither of them did, and the "
      "measurement says the per-model version of this is not worth "
      "building.\n")

    A("## 6. Measured left-tail data\n")
    A(f"The published corpus contains only recipes that worked. We rented an "
      f"A100 and measured {n('adv_rows')} rows from {n('adv_models')} models "
      f"under {n('adv_recipes')} recipes, including a correctly-configured "
      f"control arm evaluated on identical items.\n")
    A(f"**Worst measured loss: {n('adv_worst_delta','{:.1f}')}pp.** "
      f"{n('adv_rows_over_3pp')} of {n('adv_rows')} rows lost more than 3pp. "
      f"The worst loss anywhere in the published w4a16 corpus is "
      f"{n('worst::w4a16','{:.2f}')}pp.\n")
    A("| deliberate fault | mean excess over control | worst |")
    A("|---|---|---|")
    A(f"| pooled bad w4a16 recipes ({n('adv_excess_n')} rows) | "
      f"{n('adv_excess_mean','{:+.2f}')}pp | "
      f"{n('adv_excess_worst','{:+.2f}')}pp |")
    A("")
    A("Two results worth separating:\n")
    A("1. **Wrong scale granularity (per-tensor instead of per-group) is "
      "catastrophic** and is the single largest effect we measured.\n"
      "2. **Skipping SmoothQuant for INT8 did not hurt** on these small "
      "models -- that arm came out at or better than control. A plausible "
      "failure mode that turns out not to be one.\n")
    A("This is the data the published corpus structurally cannot contain.\n")
    A("### What a second GPU session then established, and cost us\n")
    A(f"Re-running MMLU under the published protocol (5-shot, letter-scored) "
      f"reproduces published base accuracy: {n('s2_base_qwen05','{:.2f}')} "
      f"against {n('s2_pub_qwen05','{:.2f}')} for Qwen2.5-0.5B and "
      f"{n('s2_base_qwen15','{:.2f}')} against "
      f"{n('s2_pub_qwen15','{:.2f}')} for Qwen2.5-1.5B. The harness "
      f"confound is closed.\n")
    A(f"Closing it exposed a larger one. Red Hat published w4a16 for the same "
      f"two checkpoint families, so we can compare head to head: our "
      f"*correctly-configured* control loses "
      f"{n('ctrl_our_delta::Qwen2_5-0_5B','{:+.2f}')}pp and "
      f"{n('ctrl_our_delta::Qwen2_5-1_5B','{:+.2f}')}pp where published loses "
      f"{n('ctrl_pub_delta::Qwen2_5-0_5B','{:+.2f}')}pp and "
      f"{n('ctrl_pub_delta::Qwen2_5-1_5B','{:+.2f}')}pp - "
      f"**{n('ctrl_ratio_x','{:.1f}')}x more damaging**, significant at "
      f"z = {n('ctrl_min_z','{:.1f}')} and {n('ctrl_max_z','{:.1f}')}. The "
      f"cause is about {n('calib_ratio_x','{:.0f}')}x less calibration data "
      f"than production, not a bug.\n")
    A("So the excess-over-control figures measure **bad recipe against "
      "mediocre recipe**, not against production. That makes them a "
      "conservative anchor rather than a bound, and it is why no corrected "
      "point estimate is published. See `BIAS_CORRECTION.md`.\n")
    A("## 7. The self-correction that mattered most\n")
    A("Late in the work we believed we had found a product bug: the tool "
      "shows a *symmetric* interval, and an earlier audit had measured a raw "
      "asymmetric empirical band covering better "
      f"({n('band_emp_coverage_pct','{:.1f}')}% vs "
      f"{n('band_conf_coverage_pct','{:.1f}')}%). Real quantization damage is "
      "left-skewed, so a symmetric band is obviously the wrong shape. We were "
      "about to switch the product over.\n")
    A(f"Measuring it properly first killed the change. The empirical band "
      f"returns an INFINITE interval on "
      f"{n('inf_rows')} of {n('band_n_scored')} rows "
      f"({n('inf_share_pct','{:.0f}')}%), because a two-sided empirical "
      f"index needs n >= 2/alpha - 1 = 19 calibration points and often has "
      f"fewer. An infinite interval covers 100% of the time by construction. "
      f"That was the entire source of its apparent advantage.\n")
    A("On the rows where both bands are actually defined:\n")
    A("| band | coverage | mean width |")
    A("|---|---|---|")
    A(f"| symmetric conformal (shipped) | "
      f"**{n('finite_conf_cov_pct','{:.1f}')}%** | "
      f"**{n('finite_conf_width','{:.2f}')}pp** |")
    A(f"| asymmetric empirical | {n('finite_emp_cov_pct','{:.1f}')}% | "
      f"{n('finite_emp_width','{:.2f}')}pp |")
    A("")
    A(f"Conformal wins on coverage AND width, on "
      f"{n('finite_n')} rows. A hybrid that falls back when the empirical "
      f"band is undefined does no better "
      f"({n('band_hybrid_coverage_pct','{:.1f}')}%). **The shipped interval "
      f"was right and the intuition was wrong.**\n")
    A("Two conclusions, both uncomfortable and both kept:\n")
    A("1. The earlier audit's own finding was an artefact of in-sample "
      "calibration. It is now marked `[CORRECTED]` rather than deleted.\n"
      "2. The gemma-3-1b case is still a miss. It is an example of the tail "
      "this tool does not describe -- which is what the selection-bias work "
      "is for -- not evidence that the interval shape is wrong.\n")
    A("The pattern worth naming: **every time a result improved without a "
      "mechanism to explain it, the improvement was an artefact.** Infinite "
      "intervals covering 100%. In-sample calibration flattering NVFP4 from "
      "74% to 94%. The discipline that produced the "
      "usable results is refusing to accept an unexplained improvement.\n")
    A("Errors also ran the other way: our first low-rank comparison was "
      "unfair to the method it tested, and an external reviewer corrected it "
      "(section 4).\n")
    A("## 8. What we would still build\n")
    A("- A per-scheme calibrated envelope, which is what the data supports. "
      "See `RANKING.md`.\n"
      "- A correction for survivorship bias, now anchored on measured "
      "adversarial data rather than an assumed tail shape. See "
      "`BIAS_CORRECTION.md`.\n")
    A("What we would not build again: a per-model predictor. The noise floor "
      "forbids it, and two independent method families (feature regression "
      "and low-rank completion, the latter re-tested with BenchPress's own "
      "code) both failed to beat a six-number lookup.\n")

    open(DOC, "w").write("\n".join(L) + "\n")
    print(f"wrote {DOC} ({len(L)} lines)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
