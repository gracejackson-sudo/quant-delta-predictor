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

## 4. Low-rank structure does not transfer (Track 2)

BenchPress (arXiv:2606.24020) predicts unseen benchmark scores by exploiting the fact that a frontier-model score matrix is roughly rank-2. We tested whether that structure helps on **paired** quantization deltas.

Our matrix is 140<!-- claim: t2_matrix_rows = 140.0000 --> rows (base and quantized checkpoints) x 16<!-- claim: t2_matrix_cols = 16.0000 --> benchmarks, 50.5<!-- claim: t2_fill_pct = 50.4911 -->% filled. Rank-2 explains 55.9<!-- claim: t2_var_explained_rank2_pct = 55.9315 -->% of standardised variance here, against the >90% BenchPress reports on frontier models.

Protocol: reveal the full base-model row plus 3 of the quantized model's scores, predict the rest, and read off the implied delta.

| rank | rows scored | low-rank MAE | scheme-mean MAE | always-zero MAE |
|---|---|---|---|---|
| 2 | 511<!-- claim: t2_n::2 = 511.0000 --> | **8.747<!-- claim: t2_mae_lowrank::2 = 8.7468 -->** | 1.385<!-- claim: t2_mae_scheme::2 = 1.3853 --> | 1.436<!-- claim: t2_mae_zero::2 = 1.4361 --> |
| 3 | 511<!-- claim: t2_n::3 = 511.0000 --> | **9.062<!-- claim: t2_mae_lowrank::3 = 9.0622 -->** | 1.385<!-- claim: t2_mae_scheme::3 = 1.3853 --> | 1.436<!-- claim: t2_mae_zero::3 = 1.4361 --> |
| 5 | 511<!-- claim: t2_n::5 = 511.0000 --> | **9.487<!-- claim: t2_mae_lowrank::5 = 9.4867 -->** | 1.385<!-- claim: t2_mae_scheme::5 = 1.3853 --> | 1.436<!-- claim: t2_mae_zero::5 = 1.4361 --> |

We predicted low-rank would *collapse to zero* and miss the damaging cases. It does something worse: at rank 2 the mean absolute predicted delta is 8.66<!-- claim: t2_mean_abs_pred::2 = 8.6558 -->pp, against true deltas whose mean absolute size is 1.44<!-- claim: t2_mae_zero::2 = 1.4361 -->pp. The reconstruction noise floor is an order of magnitude larger than the quantity being estimated.

**That is the transferable lesson:** low-rank score-matrix completion is built for cross-model variation of tens of points. A paired quantization delta lives at ~1pp. The method is not wrong; it is operating below its own resolution.

*Caveat, stated plainly:* this is a fast approximation using plain iterative-SVD completion, not a reimplementation of BenchPress. Their link functions, regularisation search and bias terms would sharpen point accuracy. They could not close an order-of-magnitude resolution gap, and their own reported 90% conformal interval width is 27.01 score points, but we did not test their exact method.

## 5. Relationship to BenchPress

BenchPress solves the same problem shape: predict an eval result, estimate whether to trust it, wrap it in a conformal interval, decide whether to skip the real run. Section 6.2 builds three reliability estimators and a risk-normalised conformal wrapper. It is a more sophisticated method than ours, with public code and data.

What it does not do is quantization: the string `quantiz` does not appear in the paper, and its 84 models contain no quantized checkpoints. Its estimand is a model's absolute score, not the delta between a model and its compressed self.

So the honest framing of this work is not that we built something new. It is that we measured something they did not, in a regime where their approach does not reach, and the measurement says the per-model version of this is not worth building.

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

Re-running MMLU under the published protocol (5-shot, letter-scored) reproduces published base accuracy: 47.20<!-- claim: s2_base_qwen05 = 47.2000 --> against 47.42<!-- claim: s2_pub_qwen05 = 47.4200 --> for Qwen2.5-0.5B and 59.57<!-- claim: s2_base_qwen15 = 59.5667 --> against 60.98<!-- claim: s2_pub_qwen15 = 60.9800 --> for Qwen2.5-1.5B. The harness confound is closed.

Closing it exposed a larger one. Red Hat published w4a16 for the same two checkpoint families, so we can compare head to head: our *correctly-configured* control loses -7.73<!-- claim: ctrl_our_delta::Qwen2_5-0_5B = -7.7333 -->pp and -4.40<!-- claim: ctrl_our_delta::Qwen2_5-1_5B = -4.4000 -->pp where published loses -2.53<!-- claim: ctrl_pub_delta::Qwen2_5-0_5B = -2.5300 -->pp and -2.05<!-- claim: ctrl_pub_delta::Qwen2_5-1_5B = -2.0500 -->pp - **2.6<!-- claim: ctrl_ratio_x = 2.6015 -->x more damaging**, significant at z = 2.9<!-- claim: ctrl_min_z = 2.9276 --> and 5.3<!-- claim: ctrl_max_z = 5.2859 -->. The cause is about 171<!-- claim: calib_ratio_x = 170.6667 -->x less calibration data than production, not a bug.

So the excess-over-control figures measure **bad recipe against mediocre recipe**, not against production. That makes them a conservative anchor rather than a bound, and it is why no corrected point estimate is published. See `BIAS_CORRECTION.md`.

## 7. The self-correction that mattered most

Late in the work we believed we had found a product bug: the tool shows a *symmetric* interval, and an earlier audit had measured a raw asymmetric empirical band covering better (93.4<!-- claim: band_emp_coverage_pct = 93.3628 -->% vs 89.6<!-- claim: band_conf_coverage_pct = 89.5526 -->%). Real quantization damage is left-skewed, so a symmetric band is obviously the wrong shape. We were about to switch the product over.

Measuring it properly first killed the change. The empirical band returns an INFINITE interval on 1762<!-- claim: inf_rows = 1762.0000 --> of 4068<!-- claim: band_n_scored = 4068.0000 --> rows (43<!-- claim: inf_share_pct = 43.3137 -->%), because a two-sided empirical index needs n >= 2/alpha - 1 = 19 calibration points and often has fewer. An infinite interval covers 100% of the time by construction. That was the entire source of its apparent advantage.

On the rows where both bands are actually defined:

| band | coverage | mean width |
|---|---|---|
| symmetric conformal (shipped) | **89.0<!-- claim: finite_conf_cov_pct = 89.0286 -->%** | **3.93<!-- claim: finite_conf_width = 3.9283 -->pp** |
| asymmetric empirical | 88.3<!-- claim: finite_emp_cov_pct = 88.2914 -->% | 5.48<!-- claim: finite_emp_width = 5.4808 -->pp |

Conformal wins on coverage AND width, on 2306<!-- claim: finite_n = 2306.0000 --> rows. A hybrid that falls back when the empirical band is undefined does no better (89.1<!-- claim: band_hybrid_coverage_pct = 89.1347 -->%). **The shipped interval was right and the intuition was wrong.**

Two conclusions, both uncomfortable and both kept:

1. The earlier audit's own finding was an artefact of in-sample calibration. It is now marked `[CORRECTED]` rather than deleted.
2. The gemma-3-1b case is still a miss. It is an example of the tail this tool does not describe -- which is what the selection-bias work is for -- not evidence that the interval shape is wrong.

The pattern worth naming: **every time a result improved without a mechanism to explain it, the improvement was an artefact.** Infinite intervals covering 100%. In-sample calibration flattering NVFP4 from 74% to 94%. Low-rank completion looking plausible until its noise floor was compared to the signal. The discipline that produced the usable results is refusing to accept an unexplained improvement.

## 8. What we would still build

- A per-scheme calibrated envelope, which is what the data supports. See `RANKING.md`.
- A correction for survivorship bias, now anchored on measured adversarial data rather than an assumed tail shape. See `BIAS_CORRECTION.md`.

What we would not build again: a per-model predictor. The noise floor forbids it and two independent method families (feature regression, low-rank completion) both failed to beat a six-number lookup.

