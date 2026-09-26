# Per-model quantization accuracy prediction has no signal beyond the per-scheme average

*A negative result, with the measured noise floor, a low-rank transfer test, and a comparison to BenchPress.*

Data: 817<!-- claim: n_rows = 817.0000 --> published evaluations from 38<!-- claim: n_checkpoints = 38.0000 --> checkpoints across 8<!-- claim: n_families = 8.0000 --> model families, scraped from RedHatAI model cards. Every figure below is generated from a computed value and re-verified by `src/verify_claims.py`.

---

## 1. The claim

Given a base model and a quantization config, predict the accuracy delta. We tested this properly and it does not work. The only thing that beats guessing the global average is a six-cell lookup on the quantization scheme, and the margin is small enough that it is worth stating in full rather than summarising.

## 2. What we tested

Leave-one-family-out, so the test family's checkpoints are never in training. Row-weighted mean absolute error, lower is better:

| predictor | MAE (pp) | beats global mean? |
|---|---|---|
| per-scheme mean (shipped) | 0.7227<!-- claim: pred_mae::scheme_mean = 0.7227 --> | yes |
| global mean (baseline) | 0.7545<!-- claim: pred_mae::global_mean = 0.7545 --> | -- |
| per-(scheme x benchmark) mean | 0.7607<!-- claim: pred_mae::scheme_x_bench = 0.7607 --> | **no** |
| ridge, 38 features | 0.7639<!-- claim: pred_mae::ridge = 0.7639 --> | **no** |
| per-benchmark mean | 0.7647<!-- claim: pred_mae::bench_mean = 0.7647 --> | **no** |
| gradient boosting | 0.7878<!-- claim: pred_mae::grad_boost = 0.7878 --> | **no** |

Ridge regression and gradient boosting both do **worse than predicting the average**. Adding model size, benchmark identity, base accuracy and quantization method all degraded out-of-family accuracy.

## 3. Why: the target is mostly measurement noise

Benchmark scores are sample proportions over finite item sets, so a delta is a difference of two noisy quantities. Estimating the noise floor directly from near-lossless schemes (W8A16 and FP8-dynamic, whose true delta should be ~0) gives an irreducible MAE of **0.529<!-- claim: mae_floor_pp = 0.5294 -->pp**.

The global-mean baseline sits at 0.7545<!-- claim: mae_global_lofo = 0.7545 -->pp. So the entire headroom available to any predictor is about 0.225<!-- claim: headroom_pp = 0.2251 -->pp, and the best predictor we found captures roughly a tenth of it.

On GSM8K specifically, the observed spread for supposedly lossless schemes is nearly identical to the spread for 4-bit schemes: the measurement is louder than the effect.

## 4. Low-rank structure and paired deltas (Track 2, revised after external review)

BenchPress (arXiv:2606.24020) predicts unseen benchmark scores by exploiting the fact that a frontier-model score matrix is roughly rank-2. We tested whether that structure helps on **paired** quantization deltas.

> **Correction.** The first version of this section made three statements that a BenchPress author showed, on reviewing our code, to be wrong or unfair. (1) It said rank-2 explains 55.9<!-- claim: t2_var_explained_rank2_pct = 55.9315 -->% of the variance in our matrix, against over 90% for BenchPress. That figure was computed after filling 49.5<!-- claim: t2b_filled_global_mean_pct = 49.5089 -->% of the matrix with one global mean, which weakens any low-rank structure. Measured as the BenchPress paper does it, the structure is present (table below). (2) It tested a plain SVD-completion approximation, not BenchPress's method. (3) It called the resulting error gap 'an order of magnitude'; even our own numbers showed about 9<!-- claim: t2_mae_lowrank::2 = 8.7468 --> versus 1.4<!-- claim: t2_mae_scheme::2 = 1.3853 -->. What follows replaces it.

### 4a. Is the structure there?

Largest fully observed submatrix with k benchmarks, each column mean-centred, nothing filled:

| benchmarks | rows (base + quantized) | rank-2 variance | base rows only |
|---|---|---|---|
| 3 | 135<!-- claim: t2b_fair_rows_k3 = 135.0000 --> | 98.76<!-- claim: t2b_fair_rank2_k3_pct = 98.7612 -->% | 94.93<!-- claim: t2b_fair_base_rank2_k3_pct = 94.9316 -->% |
| 4 | 131<!-- claim: t2b_fair_rows_k4 = 131.0000 --> | 92.25<!-- claim: t2b_fair_rank2_k4_pct = 92.2523 -->% | 89.46<!-- claim: t2b_fair_base_rank2_k4_pct = 89.4636 -->% |
| 5 | 128<!-- claim: t2b_fair_rows_k5 = 128.0000 --> | 90.55<!-- claim: t2b_fair_rank2_k5_pct = 90.5500 -->% | 89.35<!-- claim: t2b_fair_base_rank2_k5_pct = 89.3481 -->% |
| 6 | 125<!-- claim: t2b_fair_rows_k6 = 125.0000 --> | 88.32<!-- claim: t2b_fair_rank2_k6_pct = 88.3211 -->% | 88.55<!-- claim: t2b_fair_base_rank2_k6_pct = 88.5533 -->% |

Yes: rank-2 explains 88% to 99% here, as the reviewer said. Two cautions on reading it. With only 3 to 6 columns, two factors can explain a large share almost by construction, so this is a weak test next to BenchPress's 133 benchmarks. And each quantized checkpoint sits next to its own base, so the rows are not independent; the base-only column removes that and gives the same picture. What it establishes is that cross-model variation is low-rank. It says nothing yet about whether that helps predict a one-point paired delta.

### 4b. Does BenchPress's actual method predict the delta?

Protocol, unchanged: reveal the full base-model row plus 3 of the quantized model's scores, predict the rest, read off the implied delta. Now run with BenchPress's released predictor (Logit + Bias ALS, rank 2, default regularisation, unmodified code; a one-off spot check on their own released matrix gave a median error near their reported figure, but that check is not among this repository's scripts) and with our original imputer, averaged over 5<!-- claim: t2b_seeds = 5.0000 --> random draws of the revealed scores on 511<!-- claim: t2b_n::bp = 511.0000 --> scored rows. Errors are in accuracy points.

| method | MAE (sd over draws) | scheme-mean MAE | always-zero MAE | MAE / scheme-mean | correlation with true delta |
|---|---|---|---|---|---|
| our original imputer | 8.86<!-- claim: t2b_mae::orig = 8.8574 --> (0.50<!-- claim: t2b_mae_sd::orig = 0.4997 -->) | 1.45<!-- claim: t2b_mae_scheme_mean::orig = 1.4537 --> | 1.51<!-- claim: t2b_mae_zero::orig = 1.5059 --> | 6.1<!-- claim: t2b_ratio_vs_scheme_mean::orig = 6.0930 -->x | 0.02<!-- claim: t2b_corr_pred_true::orig = 0.0186 --> |
| BenchPress Logit + Bias ALS | 6.81<!-- claim: t2b_mae::bp = 6.8149 --> (0.19<!-- claim: t2b_mae_sd::bp = 0.1935 -->) | 1.45<!-- claim: t2b_mae_scheme_mean::bp = 1.4537 --> | 1.51<!-- claim: t2b_mae_zero::bp = 1.5059 --> | 4.7<!-- claim: t2b_ratio_vs_scheme_mean::bp = 4.6880 -->x | -0.01<!-- claim: t2b_corr_pred_true::bp = -0.0115 --> |
| BenchPress, target's sibling rows removed | 7.20<!-- claim: t2b_mae::bp_nosib = 7.1951 --> | 1.45<!-- claim: t2b_mae_scheme_mean::bp_nosib = 1.4537 --> | 1.51<!-- claim: t2b_mae_zero::bp_nosib = 1.5059 --> | 4.9<!-- claim: t2b_ratio_vs_scheme_mean::bp_nosib = 4.9495 -->x | 0.02<!-- claim: t2b_corr_pred_true::bp_nosib = 0.0208 --> |

Using their method instead of ours improves the error by about 23<!-- claim: t2b_bp_improvement_pct = 23.0592 -->% but does not close the gap: it is still 4.7<!-- claim: t2b_ratio_vs_scheme_mean::bp = 4.6880 --> times the error of the six-number scheme lookup, the predicted deltas are uncorrelated with the true ones, and it flags 201<!-- claim: t2b_false_alarms::bp = 200.8000 --> rows that were not severe as losing at least 1.5 points while catching 11.6<!-- claim: t2b_severe_caught::bp = 11.6000 --> of 29<!-- claim: t2b_n_severe::bp = 29.0000 --> severe ones. The likely reason is scale, not a flaw in their method: BenchPress reports recovering held-out scores to within about 4.6<!-- claim: benchpress_reported_error_pts = 4.6000 --> points, and a paired quantization delta averages 1.5<!-- claim: t2b_mae_zero::bp = 1.5059 --> points in size, so an estimate built from cross-benchmark prediction cannot resolve it. Our error of 6.8<!-- claim: t2b_mae::bp = 6.8149 --> points on a sparser, 16-benchmark matrix is the same order as their own.

### 4c. The reviewer's two smaller points

- **Benchmarks without close neighbours.** Confirmed: the strongest correlation for GPQA is 0.58<!-- claim: t2b_corr_gpqa = 0.5801 --> (over 31<!-- claim: t2b_corr_gpqa_overlap = 31.0000 --> rows) and for MuSR 0.74<!-- claim: t2b_corr_musr = 0.7381 --> (over 22<!-- claim: t2b_corr_musr_overlap = 22.0000 --> rows). Correlations elsewhere are near 1 but rest on very few overlapping rows and on base and quantized rows that duplicate each other, so we do not treat them as strong evidence.

- **Choosing the known scores.** Revealing the 3 most predictive benchmarks instead of 3 random ones gave a ratio to the lookup of 4.5<!-- claim: t2b_ratio_vs_scheme_mean::bp_pred = 4.4686 -->x for BenchPress and 4.3<!-- claim: t2b_ratio_vs_scheme_mean::orig_pred = 4.2941 -->x for our imputer, against 4.7<!-- claim: t2b_ratio_vs_scheme_mean::bp = 4.6880 -->x and 6.1<!-- claim: t2b_ratio_vs_scheme_mean::orig = 6.0930 -->x with random ones. That is a single deterministic run, with predictiveness ranked from base rows that include the model being tested, and it changes which rows are scored, so it is a weak check; it does not change the conclusion.

**What this now supports, and what it does not.** With BenchPress's released method on our matrix, low-rank completion still predicts the paired delta several times worse than a per-scheme average. It does not support an order-of-magnitude claim, and it does not show that BenchPress fails on its own task. *Not tested:* their exact evaluation harness; regularisation and rank tuned on our data; revealing more than 3 scores; applying low-rank structure to the deltas themselves rather than to scores. Any of these could change the picture.

## 5. Relationship to BenchPress

BenchPress solves the same problem shape: predict an eval result, estimate whether to trust it, wrap it in a conformal interval, decide whether to skip the real run. Section 6.2 builds three reliability estimators and a risk-normalised conformal wrapper. It is a more sophisticated method than ours, with public code and data.

What it does not do is quantization: the string `quantiz` does not appear in the paper, and its 84 models contain no quantized checkpoints. Its estimand is a model's absolute score, not the delta between a model and its compressed self.

**A closer neighbour, found after the fact.** Tong et al., *Does Compression Preserve Uncertainty?* (arXiv:2606.01850) apply conformal prediction directly to quantized and sparse LLMs, including W4A16, across 12 models from 1B to 70B. On domain that is far closer to this work than BenchPress is. On estimand it is a different problem: their conformal sets are over label space, built from a compressed model's own output probabilities, so the method requires running the compressed model. Ours is over historical accuracy deltas and exists to avoid running it. Their Figure 2 plots the same quantity this tool predicts, `Acc_compressed - Acc_dense`, but measures it rather than forecasting it.

Two of their findings bear on ours. Compression decouples accuracy from uncertainty, so an accuracy-only estimate like this one is an incomplete picture of deployment risk. And larger models absorb compression-induced uncertainty better, which is independent support for the size-band behaviour this tool already refuses on.

So the honest framing of this work is not that we built something new. It is that we measured something neither of them did, and the measurement says the per-model version of this is not worth building.

## 6. Measured left-tail data

The published corpus contains only recipes that worked. We rented an A100 and measured 45<!-- claim: adv_rows = 45.0000 --> rows from 3<!-- claim: adv_models = 3.0000 --> models under 5<!-- claim: adv_recipes = 5.0000 --> recipes, including a correctly-configured control arm evaluated on identical items.

**Worst measured loss: -39.5<!-- claim: adv_worst_delta = -39.5000 -->pp.** 18<!-- claim: adv_rows_over_3pp = 18.0000 --> of 45<!-- claim: adv_rows = 45.0000 --> rows lost more than 3pp. The worst loss anywhere in the published w4a16 corpus is -8.86<!-- claim: worst::w4a16 = -8.8600 -->pp.

| deliberate fault | mean excess over control | worst |
|---|---|---|
| pooled bad w4a16 recipes (27<!-- claim: adv_excess_n = 27.0000 --> rows) | -5.00<!-- claim: adv_excess_mean = -4.9998 -->pp | -35.77<!-- claim: adv_excess_worst = -35.7667 -->pp |

Two results worth separating:

1. **Wrong scale granularity (per-tensor instead of per-group) is catastrophic** and is the single largest effect we measured.
2. **Skipping SmoothQuant for INT8 did not hurt** on these small models -- that arm came out at or better than control. A plausible failure mode that turns out not to be one.

This is the data the published corpus structurally cannot contain.

### What a second GPU session then established, and cost us

Re-running MMLU under the published protocol (5-shot, letter-scored) recovers accuracies in the range on the RedHatAI cards: our Qwen2.5-0.5B-Instruct scores 47.20<!-- claim: s2_base_qwen05 = 47.2000 --> against the Instruct card's 46.83<!-- claim: s2_pub_qwen05 = 46.8300 --> (the base-model card gives 47.57<!-- claim: s2_pub_qwen05_base = 47.5700 -->), and our Qwen2.5-1.5B-Instruct scores 59.57<!-- claim: s2_base_qwen15 = 59.5667 -->. The harness confound is closed.

Closing it exposed a larger one. Our control ran the *Instruct* checkpoints, but Red Hat has not published W4A16 for those; the nearest available comparators are the *base-model* W4A16 cards. So the comparison is Instruct-vs-base, not like-for-like: our *correctly-configured* Instruct control loses -7.73<!-- claim: ctrl_our_delta::Qwen2_5-0_5B = -7.7333 -->pp and -4.40<!-- claim: ctrl_our_delta::Qwen2_5-1_5B = -4.4000 -->pp against the base-model W4A16 losses of -2.53<!-- claim: ctrl_pub_delta::Qwen2_5-0_5B = -2.5300 -->pp and -2.05<!-- claim: ctrl_pub_delta::Qwen2_5-1_5B = -2.0500 -->pp - **2.6<!-- claim: ctrl_ratio_x = 2.6015 -->x more damaging** (z = 2.9<!-- claim: ctrl_min_z = 2.9276 --> and 5.3<!-- claim: ctrl_max_z = 5.2859 -->). Instruct models may be more fragile than their base counterparts under 4-bit, so the ratio is indicative rather than a clean estimate; a matched Instruct-vs-Instruct control is deferred future work. The pattern is consistent with roughly 171<!-- claim: calib_ratio_x = 170.6667 -->x less calibration data than production defaults, though we have not independently tested the attribution.

So the excess-over-control figures measure **bad recipe against mediocre recipe**, not against production. That makes them an indicative anchor rather than a bound, and it is why no corrected point estimate is published. See `BIAS_CORRECTION.md`.

## 7. The self-correction that mattered most

Late in the work we believed we had found a product bug: the tool shows a *symmetric* interval, and an earlier audit had measured a raw asymmetric empirical band covering better (93.4<!-- claim: band_emp_coverage_pct = 93.3628 -->% vs 89.6<!-- claim: band_conf_coverage_pct = 89.5526 -->%). Real quantization damage is left-skewed, so a symmetric band is obviously the wrong shape. We were about to switch the product over.

Measuring it properly first killed the change. The empirical band returns an INFINITE interval on 43<!-- claim: inf_share_pct = 43.3137 -->% of evaluations (1762<!-- claim: inf_rows = 1762.0000 --> of 4068<!-- claim: band_n_pairs = 4068.0000 -->; the 817<!-- claim: band_n_rows = 817.0000 --> distinct rows are each scored under several calibration families), because a two-sided empirical index needs n >= 2/alpha - 1 = 19 calibration points and often has fewer. An infinite interval covers 100% of the time by construction. That was the entire source of its apparent advantage.

On the rows where both bands are actually defined:

| band | coverage | mean width |
|---|---|---|
| symmetric conformal (shipped) | **89.0<!-- claim: finite_conf_cov_pct = 89.0286 -->%** | **3.93<!-- claim: finite_conf_width = 3.9283 -->pp** |
| asymmetric empirical | 88.3<!-- claim: finite_emp_cov_pct = 88.2914 -->% | 5.48<!-- claim: finite_emp_width = 5.4808 -->pp |

Conformal wins on coverage AND width, on 2306<!-- claim: finite_n = 2306.0000 --> rows. A hybrid that falls back when the empirical band is undefined does no better (89.1<!-- claim: band_hybrid_coverage_pct = 89.1347 -->%). **The shipped interval was right and the intuition was wrong.**

Two conclusions, both uncomfortable and both kept:

1. The earlier audit's own finding was an artefact of in-sample calibration. It is now marked `[CORRECTED]` rather than deleted.
2. The gemma-3-1b case is still a miss. It is an example of the tail this tool does not describe -- which is what the selection-bias work is for -- not evidence that the interval shape is wrong.

The pattern worth naming: **every time a result improved without a mechanism to explain it, the improvement was an artefact.** Infinite intervals covering 100%. In-sample calibration flattering NVFP4 from 74% to 94%. The discipline that produced the usable results is refusing to accept an unexplained improvement.

Errors also ran the other way: our first low-rank comparison was unfair to the method it tested, and an external reviewer corrected it (section 4).

## 8. What we would still build

- A per-scheme calibrated envelope, which is what the data supports. See `RANKING.md`.
- A correction for survivorship bias, now anchored on measured adversarial data rather than an assumed tail shape. See `BIAS_CORRECTION.md`.

What we would not build again: a per-model predictor. The noise floor forbids it, and two independent method families (feature regression and low-rank completion, the latter re-tested with BenchPress's own code) both failed to beat a six-number lookup.

