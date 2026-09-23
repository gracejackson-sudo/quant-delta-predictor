# Talking points - day 3

*Every figure comes from the claims registry. Regenerate with `src/gen_talking_points.py`.*

---

## The spine: the self-correction

> "Late today I thought I'd found a bug in my own product. The tool shows a symmetric interval, and quantization damage is obviously left-skewed, so a symmetric range is the wrong shape. An earlier audit of mine had even measured an asymmetric version covering better, 93.4<!-- claim: band_emp_coverage_pct = 93.3628 -->% against 89.6<!-- claim: band_conf_coverage_pct = 89.5526 -->%. I was about to switch it."

> "I measured it properly first. The asymmetric band returns an *infinite* interval on 1762<!-- claim: inf_rows = 1762.0000 --> of 4068<!-- claim: band_n_scored = 4068.0000 --> rows, 43<!-- claim: inf_share_pct = 43.3137 -->%, because a two-sided empirical index needs 19 calibration points and often has fewer. An infinite interval covers 100% of the time by construction. That was the entire advantage."

> "On the 2306<!-- claim: finite_n = 2306.0000 --> rows where both bands are actually defined, the original wins on coverage *and* width: 89.0<!-- claim: finite_conf_cov_pct = 89.0286 -->% at 3.93<!-- claim: finite_conf_width = 3.9283 -->pp versus 88.3<!-- claim: finite_emp_cov_pct = 88.2914 -->% at 5.48<!-- claim: finite_emp_width = 5.4808 -->pp. What I shipped was right and my intuition was wrong."

**Why this is the story and not a footnote:** it is the fourth time this week that auditing a result before believing it caught something real. The discipline is the product.

## The one-liner

> "Every time a result improved without a mechanism I could explain, the improvement turned out to be an artefact. That happened four times this week. Refusing unexplained improvements is the only reason any of these numbers are worth anything."

## The artefacts, if pushed for specifics

1. **Infinite intervals** covering 100% by construction, on 43<!-- claim: inf_share_pct = 43.3137 -->% of rows.
2. **In-sample calibration** flattering NVFP4 from a true 74% to 94%.
3. **Low-rank completion** looking plausible until its reconstruction noise (8.66<!-- claim: t2_mean_abs_pred::2 = 8.6558 -->pp) was set against the signal it was meant to estimate (1.44<!-- claim: t2_mae_zero::2 = 1.4361 -->pp).

## Today's new evidence

- Rented an A100 and measured 45<!-- claim: adv_rows = 45.0000 --> rows across 3<!-- claim: adv_models = 3.0000 --> models and 5<!-- claim: adv_recipes = 5.0000 --> recipes, including a correctly-configured control arm.
- **Worst measured loss -39.5<!-- claim: adv_worst_delta = -39.5000 -->pp**, against -8.86<!-- claim: worst::w4a16 = -8.8600 -->pp as the worst anywhere in the published corpus the tool is calibrated on.
- 16<!-- claim: adv_bad_over_3pp = 16.0000 --> of 36<!-- claim: adv_bad_rows = 36.0000 --> deliberately faulted rows lost 3pp or more; separately 2<!-- claim: adv_control_over_3pp = 2.0000 --> of 9<!-- claim: adv_control_rows = 9.0000 --> control-arm rows did too.
- Independently corroborated: public llm-compressor issues report -14 to -16pp real-world failures, the same region as our deliberate ones.

## The second GPU session: fixed one confound, found a bigger one

> "We re-ran MMLU under the published protocol and reproduced their base accuracy to within a point - 47.20<!-- claim: s2_base_qwen05 = 47.2000 --> against their 47.42<!-- claim: s2_pub_qwen05 = 47.4200 -->. That closed the harness question. And then it showed us something worse: our own *correctly-configured* baseline is 2.6<!-- claim: ctrl_ratio_x = 2.6015 -->x more damaging than Red Hat's published one on closely comparable checkpoints - base versus -Instruct variants, not the identical artifact - because we calibrate on about 171<!-- claim: calib_ratio_x = 170.6667 -->x less data than production does."

> "So what we can prove is that bad configs are catastrophic. What we *cannot* yet say is how much worse they are than a professionally tuned recipe - our own 'good' recipe isn't good enough to be the yardstick. That's a narrower claim than I had this morning, and it's the true one."

## What I will not claim

- **Not a predictor.** It reads two inputs and has no per-model signal.
- **Not novel against BenchPress** (arXiv:2606.24020) on the core mechanism. They published predict-eval-then-conformal-interval first, with a better method, public code and data.
- **No corrected point estimate.** The selection-bias correction moves 5.70<!-- claim: s2_protocol_sensitivity = 5.7027 -->pp depending on which anchor set is used, so we publish a range, not a number.
- **pi is still unidentified.** How often a real user botches a config is not something any public data answers. I looked; nothing credible exists.

