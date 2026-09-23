# Feasibility result: predicting quantization accuracy delta with calibrated intervals

Built and validated 2026-09-21. Data: 102 RedHatAI model cards → 850 rows.
Research + pre-registered predictions: [RESEARCH.md](RESEARCH.md).

> **Corrections applied 2026-09-21 after two audit passes**
> ([ADVERSARIAL_AUDIT.md](ADVERSARIAL_AUDIT.md), [ACCOUNTING.md](ACCOUNTING.md),
> [PROVENANCE.md](PROVENANCE.md)). The headline survived both, but these claims were
> overstated and are fixed in place below:
> 1. **"7 families never used to build it" was wrong.** 32 prospective rows come from families
>    that *are* in training (`Llama-3.1-Nemotron-70B` → llama-3.1, `Qwen3-30B-A3B` → qwen3).
>    They are new *checkpoints*, not new families. Strict result: **118/131 = 90.1%**,
>    95% CI [83.6%, 94.6%].
> 2. **phi-4 was never actually tested**, nor was the entire **Phi-3 family** in training —
>    18 models dropped because their names contain no parameter count, via a gate the shipped
>    artifact doesn't even use.
> 3. **Coverage is a range, not a point.** The integrity gate removed 6 test rows, so the
>    honest figure on the full set is **87.5%–90.6%**.
> 4. **"Predictor" overstates it.** The fitted artifact was 12 numbers at the time of
>    that audit and is **18** now (6 scheme means, 6 half-widths, 6 size-dependent
>    widths), and a raw historical
>    quantile band covers better. It is a *calibrated historical baseline*.
> 5. **Two parser bugs found, both fixed:** cards that print `Math-|v|-5` instead of
>    `Math-lvl-5` lost 11 training rows; and `MATH-500` was being matched as `Math-Lvl-5`,
>    contaminating 7 test rows with ~95-point scores from a different benchmark.
> 6. **66% of the available corpus was never considered** — a hand-written family filter
>    excluded 286 of 431 quantized RedHatAI models before anything was downloaded.
>
> Current dataset: **850 rows / 102 cards / 38 checkpoints / 8 families.**

> **Update 2026-09-22 — read alongside this document.** The result above still stands
> unchanged, but two things measured since materially affect how it should be read:
> - **The corpus understates the left tail.** We quantized real models on an A100 with
>   deliberately-bad configs and measured losses to **−39.50pp** (18 of 45 rows worse than 3pp).
>   The worst loss anywhere in the published corpus below is −8.86pp. Every interval in this
>   document is therefore a floor on risk, not a ceiling. See
>   [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md).
> - **Our own control arm is not production-grade.** A correctly-configured control we ran
>   ourselves is 2.6× more damaging than published w4a16 on closely comparable checkpoints
>   (base versus -Instruct variants, not the identical artifact), because we
>   calibrate on ~171× less data. So "excess over control" is a conservative anchor, not a
>   bound, and the selection-bias correction is published as a range rather than a number.
>   See [BIAS_CORRECTION.md](BIAS_CORRECTION.md).
>
> An attempt on 2026-09-22 to close the control gap using production llm-compressor GPTQ was
> abandoned on dependency conflicts before producing any measurement; nothing here depends on it.

---

## The one-line answer

**The calibration works. The prediction doesn't — and "predictor" is the wrong word for what
survived.**

A 90% interval built by split conformal prediction contains the true measured delta
**118/131 = 90.1%** of the time on checkpoints the system had never seen (95% CI
[83.6%, 94.6%], nominal 90%) — a figure reproduced exactly by a from-scratch,
stdlib-only reimplementation that shares no code with the pipeline
(`verify/independent_check.py`: 0 delta disagreements, 0 verdict disagreements
across all 186 shared rows). But the point estimate carries almost no information beyond
"which quantization scheme did you pick" — it beats guessing the global average by
**0.026pp of MAE**, ~12% of the distance to the noise floor.

The adversarial audit went further: the entire fitted artifact is **18 numbers** (6 scheme
means, 6 half-widths, 6 size-dependent widths; it was 12 before size-widening was added), and a raw per-scheme historical quantile band with no model and no conformal
machinery achieves *better* coverage (92.2% vs 90.2%). What conformal buys is a finite-sample
guarantee and a principled rule for small-n schemes, not accuracy.

**What that guarantee is conditional on, stated precisely.** Split conformal gives exact
finite-sample coverage *under exchangeability* between the calibration set and the test point.
Our calibration set is the set of configs Red Hat chose to publish, which is a selection event,
and selection is known to break exchangeability: conditional on having been selected, calibration
points are no longer exchangeable with an arbitrary test point, and the coverage guarantee does
not transfer (Barber, Candes, Ramdas & Tibshirani, *Conformal Prediction Beyond Exchangeability*,
Ann. Statist. 51(2), 2023; Jin & Candes, arXiv:2403.03868). So the guarantee holds for a new
checkpoint drawn from the same publication process, and is **void** for a recipe you tuned
yourself and that no one would have published. That is the formal statement of the same warning
the tool prints in plain language.

So the honest product is **not** "predicted accuracy delta for your model", and not really a
predictor at all. It is **a calibrated historical baseline: the observed distribution of
quantization damage per scheme, with a coverage guarantee that holds against the published
population and not against your own untuned recipe.** That is a much smaller
claim than the one I set out to test, and it is defensible.

---

## 1. Did a held-out real result land inside the predicted interval? Show the numbers.

Yes. Two levels of rigor, all with real measured values.

*(An earlier draft opened with a single flagship model scored across nine
benchmarks, 9/9 inside. It was cut: all nine rows carry the same prediction and
the same interval, so it is one prediction scored nine times on correlated
benchmarks, not nine independent successes. §1b and §1c below are the real
evidence.)*

### 1b. Five held-out models, five unseen families

Same frozen train/calibration split:

| model | family | scheme | inside / n |
|---|---|---|---|
| Mistral-Small-24B-Instruct-2501-FP8-dynamic | mistral | fp8_dynamic | 9/9 |
| Llama-3.3-70B-Instruct-quantized.w4a16 | llama-3.3 | w4a16 | 9/9 |
| gemma-2-9b-it-quantized.w4a16 | gemma-2 | w4a16 | 5/6 |
| Qwen3-8B-quantized.w4a16 | qwen3 | w4a16 | 8/11 |
| Llama-3.2-3B-Instruct-FP8-dynamic | llama-3.2 | fp8_dynamic | 7/7 |

**Pooled: 38/42 = 90.5%** (95% CI [77.4%, 97.3%]), nominal 90%. Mean half-width 2.60pp.

The single worst miss is real and instructive: **Qwen3-8B W4A16 loses 8.86pp on MMLU-Pro**
(34.57 → 25.71). The card itself states 74.4% recovery, so this is not a parsing artifact —
it is a genuine failure mode far outside anything Llama-3.1/Qwen2.5 cards contain.

### 1c. The prospective test (the one that counts)

After the predictor was frozen, I downloaded model cards for checkpoints not used during
development: gemma-3, DeepSeek-R1-Distill, SmolLM/SmolLM3, Nemotron, Llama-4 and Qwen3-30B-A3B
(MoE). 24 (model, config) pairs, 186 rows:

```
inside the 90% interval  : 168/186 = 90.3%
95% CI on that coverage  : [85.1%, 94.2%]   -> contains nominal 90%
mean half-width          : 1.83pp
MAE                      : 0.929pp
MAE (global-mean baseline): 0.958pp
intervals excluding zero : 0%
```

**Audit-corrected version of the same number.** Three contaminating effects, each removed:

| test set | rows | coverage | 95% CI | contains 90%? |
|---|---|---|---|---|
| all prospective rows | 186 | 90.3% | [85.1%, 94.2%] | yes |
| − rows whose family is in training | 154 | 91.6% | [86.0%, 95.4%] | yes |
| − those **and** Llama-4 (**strict**) | 131 | **90.1%** | **[83.6%, 94.6%]** | **yes** |

Llama-4 is excluded from the strict figure because the recovery-first column-order fix in
`harvest.py` was written *after* seeing a bad Llama-4 row — so the parser was adapted to that
card family and those 23 rows are not strictly prospective.

Accounting for the 6 test rows the integrity gate rejected, the honest bound on the full set is
**87.5% (if every rejected row would have missed) to 90.6% (if every one would have hit)**.
Manual inspection shows those 6 are card errors, not extreme-but-valid data — e.g.
`DeepSeek-R1-Distill-Qwen-32B-W4A16` prints IFEval as `42.87 → 72.48` with `99.1%` recovery,
which is arithmetically impossible in either direction.

The strict subset (**118/131 = 90.1%**) is the number I would publish.

Per unfamiliar group:

| group | n | coverage | MAE | baseline MAE | worst true delta |
|---|---|---|---|---|---|
| nemotron | 11 | 81.8% | 1.441 | 1.281 | -2.28 |
| qwen3-30b-a3b | 22 | 86.4% | 1.330 | 1.458 | -6.69 |
| gemma-3 | 45 | 86.7% | 0.862 | 0.947 | -6.37 |
| deepseek-r1-distill | 75 | 90.7% | 1.008 | 1.009 | -8.15 |
| llama-4 | 23 | 100.0% | 0.427 | 0.408 | -1.03 |
| smollm | 9 | 100.0% | 0.231 | 0.358 | -0.71 |
| smollm3 | 1 | 100.0% | 1.375 | 1.029 | +0.64 |

### 1d. Calibration across levels and split regimes

Coverage tracks the nominal level, which is the real evidence that the conformal layer is doing
its job rather than getting lucky at one setting (leave-one-family-out):

| nominal | observed | gap | mean half-width |
|---|---|---|---|
| 80% | 79.2% | −0.8pp | 1.26pp |
| 90% | 88.2% | −1.8pp | 2.00pp |
| 95% | 94.8% | −0.2pp | 3.47pp |

And across split regimes, with scheme-conditional (Mondrian) calibration:

| regime | coverage | half-width | MAE | worst fold |
|---|---|---|---|---|
| A: random row split (**leaky**) | 91.2% | 1.93pp | 0.719 | 82.3% |
| B: leave-one-base-model-out | 89.8% | 1.88pp | 0.717 | 40.0% |
| C: leave-one-family-out | 89.2% | 1.96pp | 0.731 | 54.5% |

The A-vs-C gap I expected to be the headline (RESEARCH.md §1, thread E) turned out to be small
for *marginal* coverage — 91.2% vs 89.3%. The leakage showed up somewhere else instead: in
**per-scheme** coverage. With one global interval, leave-family-out coverage collapses on the
aggressive schemes:

| scheme | marginal calibration | scheme-conditional (Mondrian) |
|---|---|---|
| nvfp4 | 68% | 74% |
| w4a16 | 78% | 89% |
| fp8_dynamic | 88% | 90% |
| w8a8_int | 89% | 92% |
| w8a16 | 93% | 90% |
| fp8 | 96% | 93% |

So a "90% interval" advertised marginally was really a 68% interval for the people using NVFP4 —
exactly the users who most need the number. Mondrian calibration fixes most of that and is the
shipped default. **NVFP4 is still undercovered at 76% and should carry a warning.**

---

## 2. How many model families / configs does this currently work for?

**Configs: 6, and this is the axis where it genuinely works.** The learned table, with its
scheme-conditional 90% half-widths:

| scheme | mean delta | 90% half-width | n (train) |
|---|---|---|---|
| fp8_dynamic | −0.06pp | ±1.39 | 213 |
| w8a16 | −0.08pp | ±0.90 | 86 |
| w8a8_int | −0.31pp | ±1.33 | 203 |
| fp8 | −0.32pp | ±0.79 | 80 |
| w4a16 | −0.73pp | ±2.13 | 202 |
| nvfp4 | −0.91pp | ±2.78 | 63 |

That ordering independently reproduces the Neural Magic result (FP8 ≈ lossless, INT8 small,
4-bit worse) from a completely separate scrape, which is a good sign the data pipeline is sound.

**Families: 15 have been evaluated — 8 in development, 7 prospectively — covering 62
(model, config) pairs and 1,032 rows.** But this number is misleading and I don't want to quote
it as a capability. The predictor **does not use the model family at all.** It returns the same
number for Llama-3.1-8B-W4A16 and for DeepSeek-R1-Distill-Qwen-14B-W4A16. So it doesn't "work
for 15 families" — it works for 6 schemes, and 15 families' worth of data says the *interval*
transfers across families reasonably well.

The reason it ignores the family is that nothing else survived honest evaluation. Under
leave-one-family-out (row-weighted MAE, lower is better):

| predictor | MAE | beats global mean? |
|---|---|---|
| **per-scheme mean** | **0.7065** | **yes, by 0.026** |
| global mean (baseline) | 0.7323 | — |
| per-(scheme × benchmark) mean | 0.7537 | no |
| per-benchmark mean | 0.7668 | no |
| ridge, 38 features | 0.7678 | no |
| gradient boosting | 0.7834 | no |

Ridge *and* gradient boosting are both **worse than guessing the average**. Adding model size,
bit-width, method, benchmark identity and headroom all made it worse out-of-family. This refutes
my pre-registered H2 (I expected benchmark identity to be the strongest feature — per-benchmark
means don't transfer across families at all).

And the skill that does exist is tiny but real. Bootstrapping over families (5,000 resamples,
clustered to respect the grouped structure):

```
skill = +0.0258pp MAE   95% CI [+0.0071, +0.0428]   P(skill > 0) = 100%
```

Statistically positive, practically negligible: the irreducible MAE floor set by evaluation
noise is **0.529pp**, the baseline is 0.755pp, so there was 0.226pp of headroom available and
the predictor captured ~0.024pp of it — about **11%**.

### Why there's so little to capture

This was the thing I most wanted to know from the research phase, and it held up. Estimating
per-benchmark evaluation noise directly from near-lossless schemes (W8A16, FP8-dynamic, whose
true delta should be ≈0 — observed mean −0.065pp, so the probe is sound):

| benchmark | E\|noise\| | sd (lossless) | sd (all) | noise share of variance |
|---|---|---|---|---|
| GSM8K | 1.31pp | 2.10 | 2.12 | 97% |
| MMLU (CoT) | 0.51pp | 0.84 | 0.75 | 100% |
| GPQA | 1.26pp | 1.54 | 1.64 | 88% |
| IFEval | 0.53pp | 0.69 | 0.77 | 81% |
| ARC-Challenge | 0.45pp | 0.69 | 0.77 | 80% |
| MMLU | 0.25pp | 0.48 | 0.64 | 56% |
| Winogrande | 0.40pp | 0.52 | 0.77 | 46% |
| TruthfulQA | 0.31pp | 0.43 | 0.67 | 42% |
| HumanEval | 0.45pp | 0.69 | 1.13 | 37% |
| MMLU-Pro | 0.93pp | 1.31 | 2.29 | 33% |
| BBH | 0.50pp | 0.74 | 1.31 | 32% |
| HellaSwag | 0.18pp | 0.32 | 0.84 | 15% |

The target is mean **−0.36pp with sd 1.21pp**. On GSM8K the noise alone is ±2pp — the observed
spread for 4-bit models (sd 3.02) is barely separable from the spread for supposedly lossless
ones (sd 2.10). **There is very little signal to find on most benchmarks, and a predictor that
claimed high R² here would be leaking.** Where there is real headroom it's on
HellaSwag / MMLU-Pro / BBH (noise share 15–33%), which is where a v2 should concentrate.

---

## 3. What's the biggest thing that would break if a stranger tried an unfamiliar model family right now?

Four things, in order of how badly they'd burn someone.

**1. Selection bias in the training data — the most serious and least fixable.** Red Hat
publishes checkpoints they were happy with. A quantization config that wrecked a model never got
a model card. My training distribution is therefore **truncated on the outcome**, which means the
predictor systematically underpredicts damage for a *badly chosen* config — precisely the case a
stranger most needs a warning about. Nothing in the current design detects this. Someone
quantizing an unusual architecture with an untuned recipe would get a reassuring ±2pp envelope
that was estimated entirely from recipes that worked.

**2. Small models and reasoning-distilled / MoE models fall outside the envelope.**
`gemma-3-1b-it-W4A16` missed on **4 of 6** benchmarks (GSM8K −3.03, MMLU −2.99, ARC −2.90,
TruthfulQA +1.40 against an interval of [−2.85, +1.40]). The worst single prospective misses are
all in this class:

| model | benchmark | true delta | interval |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-14B W4A16 | IFEval | **−8.15** | [−2.85, +1.40] |
| Qwen3-30B-A3B W4A16 | BBH | **−6.69** | [−2.85, +1.40] |
| gemma-3-27b-it W8A8 | GSM8K | **−6.37** | [−1.64, +1.02] |
| Qwen3-30B-A3B FP8-dynamic | Math-lvl-5 | **+7.25** | [−1.44, +1.33] |

The interval is built from a population dominated by 7B–70B dense instruct models. A 1B model, a
sparse MoE, or a reasoning distill is a different animal, and the envelope doesn't know that.

**3. It has no model-specific signal whatsoever.** A stranger will type in their model name and
receive an answer that ignores it completely. That's honest given §2, but it means the tool
cannot answer "is *my* model unusually fragile?" — which is the actual question people have. If
the framing implies otherwise, it's misleading.

**4. The intervals never exclude zero — 0% of them, in both hold-out tests.** The tool can say
"probably fine, within ±2pp." It can never say "don't do this, it will cost you." That caps the
commercial value: the answer is almost always "go ahead," which is also what you'd have guessed
without the tool.

A fifth, smaller one: **data-pipeline fragility on unfamiliar cards.** The prospective run
caught a real instance — the Llama-4 cards order their columns `[Recovery, base, quant]` instead
of `[base, quant, Recovery]`, which turned a recovery of 100.0 into a GPQA "accuracy" of 100.0
and a fabricated −68pp delta. The recovery-arithmetic gate rejected 10 of that card's 11 rows and
only let through the one where `base == quant` made the check degenerate. It is now fixed and
regression-tested, but it's a reminder that a stranger's card format can silently corrupt input.

---

## 4. Rough estimate: hours from "works on 2-3 families" to "post it publicly and let strangers try it"

**For what it actually does today — a calibrated per-scheme risk envelope: 12–18 hours.**

| work | hours |
|---|---|
| Harvest all ~480 quantized RedHatAI cards (pipeline already generalizes; needs per-card format handling + a rejection dashboard) | 3–4 |
| Add held-out-family calibration as the shipped path, and fix NVFP4's 76% undercoverage (needs more NVFP4 families — only 3 in the data) | 2–3 |
| Stratify the envelope by model-size band and dense-vs-MoE, since §3.2 is where it breaks | 3–4 |
| CLI/web front end, honest copy about what it can't do, and a "your model looks out-of-distribution" flag | 3–4 |
| Write-up, reproducibility, licensing/attribution for scraped cards | 1–2 |

**For the thing I originally wanted — a real per-model predictor: I don't think more hours fix
it, and that's the important finding.** The blocker isn't engineering, it's that (a) 33–100% of
the variance on most benchmarks is evaluation noise that no model can predict, and (b) published
model cards are outcome-truncated. Getting past that needs a different input, not a better
regressor: raw `preds.json` to compute flip rates and KL-divergence per item, plus deliberately
*bad* quantization runs to break the selection bias. That's a 40–80 hour project with real GPU
cost, and the literature (*Displacement Is Not Direction*, *Accuracy is Not All You Need*)
suggests KLD correlates poorly with accuracy delta exactly in the near-baseline regime where all
this data lives. I would not bet the week on it.

---

## Recommendation for the week

Shift to the **narrower claim**, which the numbers actually support:

> For the six quantization schemes Red Hat publishes, here is a conformal-calibrated envelope on
> the OpenLLM accuracy delta, validated at **118/131 = 90.1%** empirical coverage (95% CI
> [83.6%, 94.6%]) on checkpoints never used to build it, under the strict protocol of §1c. It does
> not predict per-model damage, and it should not be trusted for sub-2B models, MoE models, or
> untuned recipes. It is calibrated only on recipes that were published, i.e. that worked: for
> measured behaviour of deliberately-bad recipes see [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md).

That is publishable, honest, and genuinely useful as a deployment sanity check. The version that
takes a model family and tells you what *your* model will lose is not supported by this evidence,
and tonight's result is that the bottleneck is measurement noise plus data selection, not model
capacity.

---

## Verification performed

- **113 unit/regression tests** pass (`pytest tests -q`), covering cell parsing, benchmark
  canonicalization, the recovery-tolerance derivation, column-orientation detection (including
  swapped and recovery-first layouts), config/params/family parsing, conformal quantile indexing,
  and Mondrian back-off.
- **Three audits** (`src/audit.py`): 0 hard failures, 1 known warning. Including an end-to-end
  check that the conformal implementation hits **90.14%** coverage on synthetic exchangeable data,
  a proof that features are unchanged when the target is destroyed, and row-/family-disjointness
  assertions on every fold.
- **Audits found four real bugs**, all fixed: `diffusiongemma-26B` matched into the `gemma-2`
  family by substring; that same card reporting accuracy on a 0–1 scale (deltas ~100× too small);
  the Llama-4 recovery-first column order; and an arithmetic slip in RESEARCH.md (the minimum
  conformal calibration size for 90% is 9, not 19 — the over-conservative `n≥19` guards are
  retained deliberately and documented).
- **Data integrity**: 786 of 786 three-column rows reproduce their card's own printed Recovery
  figure to within a tolerance derived from the card's printed precision. 18 rows were **rejected**
  because the card is internally inconsistent (e.g. Llama-3.1-8B-W4A16 prints ARC 81.4 → 80.2 but
  claims 98.0% recovery, when the ratio is 98.5%). 53 rows come from two-column tables with no
  recovery figure and are flagged as sign-unverifiable.
- **Robustness**: headline coverage moves by <0.5pp when the `acc_before ≥ 20` filter is dropped
  (88.1%) or when restricted to recovery-verified rows only (88.6%).

## Files

| path | what |
|---|---|
| [RESEARCH.md](RESEARCH.md) | research synthesis + pre-registered predictions, written before building |
| `src/harvest.py` | model-card → dataset, with the recovery self-verification gate |
| `src/model.py` | features, ridge, conformal quantile with finite-sample correction |
| `src/predictor.py` | the shipped predictor: scheme-mean + Mondrian conformal |
| `src/diagnose.py` | the diagnostics that chose the model class |
| `src/run_experiments.py` | ridge across the three split regimes |
| `src/run_final.py` | final evaluation + MAE floor |
| `src/demo_holdout.py` | the 2-family → held-out-model demo (§1b) |
| `src/real_use_case.py` | prospective test on never-seen families (§1c) |
| `src/audit.py` | the three audits |
| `tests/` | 113 tests |
| `data/dataset.csv` | 850 rows |
| `out/` | all results as JSON + per-row predictions |
