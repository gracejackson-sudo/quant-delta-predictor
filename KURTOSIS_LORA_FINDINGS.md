# Does weight-outlier kurtosis predict LoRA-merge forgetting?

*Working note, 25 September 2026. GPU run on a rented A100-SXM4-40GB. Every number below is reproduced by
`gpu/analyze_lora_kurtosis.py`, which reads the two raw artifacts directly --
none is hand-typed. Not published.*

## The question

A Reddit thread hypothesized that models with more per-channel weight
outliers (measured as excess kurtosis on output-channel rows of q/k/v/o/gate/
up/down_proj) should be more fragile under post-training perturbation --
LoRA-merge forgetting being one instance. Early data (2 models, informal)
hinted that Qwen2.5-0.5B had roughly 2x the outlier-channel share of
Qwen2.5-1.5B and also worse observed quantization damage, directionally
consistent with the hypothesis.

The standing instruction was explicit: **test whether this holds once the
full LoRA/kurtosis data is analyzed together, don't assume it does.** It does
not hold. The result is a genuine negative finding, reported as such.

## What was actually measured

**Kurtosis: 4 models** -- Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct,
granite-3.1-2b-instruct, Qwen3-8B.

**LoRA forgetting: 3 base models, 8 adapters, 3000-item 5-shot MMLU each**
-- Qwen2.5-0.5B-Instruct (2 adapters), Qwen2.5-1.5B-Instruct (4 adapters),
Qwen2.5-3B-Instruct (2 adapters). All 8 pairs completed cleanly; results
written incrementally (verified: the CSV had partial rows on disk mid-run,
so a crash would not have lost completed pairs).

**The overlap between the two datasets is 2 models: Qwen2.5-0.5B-Instruct
and Qwen2.5-1.5B-Instruct.** Qwen2.5-3B-Instruct, the third LoRA-tested
model, was never measured for kurtosis. granite-3.1-2b-instruct and Qwen3-8B,
two of the four kurtosis-measured models, were never LoRA-tested. This
scoping gap was not visible from the earlier informal 2-model comparison
because that comparison never had a third model to expose it.

## The result

| base model | LoRA pairs | mean forgetting | std | range | pct channels kurt>3 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 2 | -0.83pp | 1.46 | [-1.87, +0.20] | 4.245% |
| Qwen2.5-1.5B-Instruct | 4 | -0.68pp | 1.05 | [-2.20, +0.13] | 2.234% |
| Qwen2.5-3B-Instruct | 2 | -0.98pp | 0.02 | [-1.00, -0.97] | not measured |

**The direction from the earlier 2-model hint technically survives**: 0.5B
still has both the higher outlier-channel share and (now) the more negative
mean forgetting. **But this is not a supportable correlation.** Two problems:

1. **It is still an n=2 model comparison.** Having 3000-item evals and 8
   adapters did not add a third data point to the model-level comparison --
   Qwen2.5-3B, the one model that could have made this an n=3 test, has no
   kurtosis measurement. The "fuller data" is fuller only in per-pair
   precision, not in the number of models being compared.
2. **Within-model (adapter-to-adapter) variance is larger than the
   between-model signal.** The gap between the two models' mean forgetting
   is 0.16pp; the standard deviation *within* each model's own adapter set is
   1.46pp and 1.05pp respectively. Qwen2.5-0.5B's two adapters alone range
   from +0.20pp to -1.87pp -- opposite signs. Whatever is driving forgetting
   varies far more by *which adapter* than by *which base model*.

**Pooling all 6 rows where kurtosis is available (pseudo-replicated -- only 2
distinct kurtosis values across those 6 rows) gives essentially no
correlation: Pearson r = -0.078 (p = 0.88), Spearman r = 0.207 (p = 0.69).**
Neither is close to significant, and with only 2 distinct model-level
kurtosis values the test is underpowered by construction regardless of the
p-value -- it could not have found a real effect even if one existed at this
sample size. The honest reading is not "no effect was found" but "this
dataset cannot test for one."

## What does show a signal

**LoRA rank correlates with forgetting across all 8 pairs: Spearman r =
-0.700 (p = 0.053).** Higher-rank adapters tend toward more negative
forgetting. This is close to, but does not clear, conventional significance
at n=8, and it is confounded with everything else about each adapter
(what it was fine-tuned for, training data quality, training duration) that
rank was not designed to isolate. It is reported as a candidate explanation
worth a properly designed follow-up, not as a finding on its own.

## What this run supports and does not support

**Supports:** LoRA-merge forgetting is real, measurable, and adapter-
dependent -- ranging from +0.20pp (slight improvement) to -2.20pp (real
damage) across otherwise-similar adapters on the same base model. That range
is worth having on record regardless of what predicts it.

**Does not support:** any claim that per-channel weight kurtosis predicts
which models or channels are more fragile under LoRA merging. The data this
run produced cannot distinguish that hypothesis from noise, because the
models it can compare (0.5B vs 1.5B) is too few, and the one model that
could have extended the test (3B) was not measured for kurtosis in this run.

## What would actually test this

Kurtosis needs to be measured for Qwen2.5-3B-Instruct (cheap, CPU-bound,
~5-10s per model based on this run's timing -- no GPU rental required) to
turn this into a real n=3 model comparison. Even at n=3 this remains
underpowered for a correlation claim; a properly powered version would need
kurtosis and LoRA-forgetting measured on the same 6-8+ models, not
opportunistically overlapping sets chosen for other reasons. Not started
here -- flagged as a scoping decision for whoever picks this up next, not
decided unilaterally.

## Standing note on GPU timing (carried from the run approval)

This run was on an A100-SXM4-40GB. The prior run in this same investigation
was on an A10 and ran ~4.5x slower per eval than an A100-based estimate
predicted. This run's per-eval times (111-226s for a 3000-item 5-shot MMLU
eval) came in faster than the original ~110 min combined-job estimate (actual
wall clock: kurtosis ~2 min + LoRA ~36 min = ~38 min), consistent with being
back on the card class the original estimate assumed. Future GPU-time
estimates should ask which card class before quoting a number, not carry over
a prior session's timing -- this is the second time in this project that
card class alone explained a >2x timing discrepancy.
