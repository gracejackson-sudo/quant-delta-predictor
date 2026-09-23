# Can you predict how much accuracy a quantized LLM loses, before you run the eval?

**Research + design synthesis, written before building anything.**
Grace Jackson · 2026-09-21

---

## 0. What I'm actually asking

The product idea is: you tell me `(base model family, quantization config)` and I tell you
`predicted accuracy delta ± a confidence interval`, without you having to burn GPU hours running
lm-eval-harness yourself.

Before writing code I wanted to know two things:

1. **Is there a real signal to learn?** i.e. is quantization damage systematic enough that a
   regression on cheap features (bit width, model size, method, benchmark) can beat "just guess the
   average"?
2. **Is the target even measurable?** If the thing I'm predicting is mostly evaluation noise, then
   no model — not ridge, not XGBoost, not a transformer — can predict it, and the honest deliverable
   is a wide-but-correctly-calibrated interval rather than a sharp point estimate.

Question 2 turns out to be the whole ballgame, and I only realized that from the literature. I'm
writing it down here so that when the numbers come out modest, I don't retroactively pretend I
expected something else.

---

## 1. Prior art, five threads

### Thread A — somebody already ran the experiment I'd want to run (500,000 times)

Neural Magic (now Red Hat AI) published *"Give Me BF16 or Give Me Death"? Accuracy-Performance
Trade-Offs in LLM Quantization* (arXiv 2411.02355, Kurtic, Marques, Pandit, Kurtz, Alistarh). They
ran **over 500,000 evaluations** across the whole Llama-3.1 family (8B / 70B / 405B) on three
schemes. Headline results:

| Scheme | Compression | What they found |
|---|---|---|
| W8A8-FP (FP8) | ~2× | "effectively lossless across all model scales" |
| W8A8-INT | ~2× | 1–3% accuracy degradation when well-tuned |
| W4A16-INT | ~3.5× | "more competitive than expected", rivals 8-bit |

Their companion Red Hat Developer article adds the numbers I care about most: on **OpenLLM v1, all
schemes regardless of model size recover >99% of the average score**. On **OpenLLM v2, average
recovery is near 99% with a floor of 96%**. HumanEval: 99.9% recovery for 8-bit, 98.9% for 4-bit.
Arena-Hard deltas had *overlapping 95% confidence intervals* across every scheme.

Two implications I can't ignore:

- **The dynamic range of my target variable is tiny.** If recovery is 96–100%, then on a benchmark
  scoring ~75, the delta is roughly −3 to 0 points. That's the entire span I'm trying to predict.
- **They explicitly say the harder v2 tasks have "higher variance", especially for smaller models.**
  That's the authors of the dataset telling me part of my signal is noise.

### Thread B — there *is* a known functional form for quantization damage

*Scaling Laws for Precision* (Kumar, Ankner et al., ICLR 2025 — 465 pretraining runs, models to
1.7B, up to 26B tokens) fits precision-aware scaling laws. The part relevant to me is their
**post-training quantization** term: degradation from PTQ

- **decays roughly exponentially in bit width** (each extra bit buys you a multiplicative reduction
  in damage), and
- **grows with the data-to-parameter ratio** — models trained on more tokens per parameter are
  *more* fragile to PTQ, to the point where extra pretraining data becomes actively harmful if you
  intend to quantize.

This is genuinely useful as a prior, and it tells me my feature set should not be linear in
bit-width. `exp(-bits)` or `1/bits` or a log-compression-ratio term should carry more of the signal
than raw `bits`. It also tells me **model size alone is the wrong size feature** — what matters is
tokens-per-parameter, which I mostly can't observe from a model card. That's a known, named gap in
my feature set rather than a surprise.

### Thread C — the proxy-metric people have already failed at the easy version of this

I expected to find that perplexity or KL-divergence is a cheap oracle for downstream accuracy
delta. It isn't, and the failure mode is instructive:

- *Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deployment*
  finds KLD is predictive of quality mainly for **badly degraded low-bit variants**, and that "in
  the near-baseline range the correlation collapses across model families and metric variants."
  Their framing is sharp: **KLD measures displacement, not direction** — it counts how often the
  quantized model disagrees with the original, but not whether the disagreement helped or hurt.
- *Accuracy is Not All You Need* (NeurIPS 2024) shows that even when quantized and baseline
  accuracy match almost exactly, there are large numbers of **"flips"** — answers going
  correct→incorrect and incorrect→correct in roughly offsetting quantities.

The flips result is the one that reframed the project for me. **A near-zero accuracy delta is not
evidence of a near-zero change in behavior; it's evidence of two large error flows cancelling.**
Which means the accuracy delta I'm predicting is a *difference of two noisy quantities that
partially cancel* — the numerically smallest, and therefore hardest-to-predict, summary of
quantization damage available.

Also relevant: I found no existing tool that does what I'm proposing (input a config, get a
calibrated interval on the delta). The literature does measurement and explanation; nobody ships a
predictor. That's an opening, but Thread C suggests it's an opening partly because the thing is
hard.

### Thread D — the noise floor, computed rather than assumed

This is the calculation I should have done first. Every lm-evaluation-harness task reports a
standard error (`acc_stderr`, `exact_match_stderr`) because a benchmark score is a sample
proportion over a finite item set. Using the binomial standard error `sqrt(p(1-p)/n)` at the
accuracy levels Llama-3.1-8B-Instruct actually achieves:

| Benchmark | items n | typical p | stderr (pp) |
|---|---|---|---|
| MMLU (5-shot) | 14,042 | 0.68 | **0.39** |
| MMLU-Pro | 12,032 | 0.31 | **0.42** |
| HellaSwag | 10,042 | 0.81 | **0.40** |
| BBH | ~6,500 | 0.30 | **0.57** |
| GSM8K | 1,319 | 0.83 | **1.03** |
| Winogrande | 1,267 | 0.78 | **1.16** |
| ARC-Challenge | 1,172 | 0.81 | **1.15** |
| Math-lvl-5 | 1,324 | 0.16 | **1.00** |
| MuSR | 756 | 0.08 | **0.96** |
| TruthfulQA (mc2) | 817 | 0.55 | **1.74** |
| IFEval | 541 | 0.78 | **1.78** |
| GPQA (0-shot) | 448 | 0.04 | **0.89** |
| HumanEval | 164 | 0.67 | **3.67** |

Now: my target is a **difference** of two such scores. If the two evals were independent, the
standard error on the delta would be `sqrt(2)×` the above — so ±1.5pp on GSM8K and ±5pp on
HumanEval. They're *not* independent (same items, same prompts, same harness), so the paired
standard error is smaller: `sqrt((p01+p10)/n)` where `p01`,`p10` are the flip rates. But Thread C
tells me flip rates are *substantial* even at matched accuracy. If 10% of GSM8K items flip in one
direction or the other, the paired SE on the delta is `sqrt(0.10/1319)` ≈ **0.87pp**.

**Put that next to Thread A: typical true deltas are 0 to −3pp, and the measurement noise on a
single delta is ~0.4pp on the big benchmarks and ~1–4pp on the small ones.** On GSM8K, HumanEval,
IFEval, TruthfulQA, GPQA and MuSR, the noise is the same order as the signal or larger. That is a
hard ceiling on achievable R². It is not a modeling problem and no amount of XGBoost fixes it.

**This is the single most important thing this review surfaced, and it generates a concrete prediction:
a good predictor will have low R² and wide intervals, and that will be the *correct* answer rather
than a failure.** It also tells me *where* the achievable signal lives: the high-n benchmarks
(MMLU, MMLU-Pro, HellaSwag, BBH), where noise is ~0.4pp.

### Thread E — conformal prediction, and the assumption I'm about to break

Split conformal regression (the method the task specifies) is the right tool: it's distribution-free
and gives **exact finite-sample coverage under exchangeability**. Mechanics: fit on a training
split, compute nonconformity scores `s_i = |y_i - ŷ_i|` on a held-out calibration split of size
`n`, take `q̂` = the `ceil((n+1)(1-α))`-th smallest score, and emit `ŷ ± q̂`.

Two facts from the literature that matter for a small dataset:

1. **The finite-sample correction is not optional at my sample sizes.** You need
   `ceil((n+1)(1-α)) ≤ n`, i.e. `n ≥ 1/α − 1` calibration points, which is **9** for a 90%
   interval and 19 for a 95% one. Below that the method cannot produce a finite interval at all
   and must return ±∞. With n in the tens, using the plain empirical 90th percentile instead of
   the corrected index measurably undercovers.
   *(Correction added after audit 2: an earlier draft of this document said 9 was 19 — the
   arithmetic slip is recorded here rather than quietly fixed, because the code's `n_cal >= 19`
   guards were originally sized from the wrong number. They are retained as a deliberately
   conservative floor, not because 90% coverage requires them.)*
2. **Coverage is guaranteed only *marginally, in expectation over calibration draws*.** A recent
   small-data paper (*Probabilistic Conformal Coverage Guarantees in Small-Data Settings*, 2025)
   makes the point that realized coverage for one particular calibration set can be far from
   nominal. So with n≈40 I should expect coverage to wobble by several points around 90% purely
   from the split, and I should measure that wobble over many random seeds rather than trusting one
   split.

And the assumption I will break: **my rows are not exchangeable.** One model card gives me ~13 rows
(one per benchmark) that share a model, a quantization run, a calibration dataset, and a single
random seed. A random row-level split puts MMLU-from-Llama-8B-w4a16 in training and
GSM8K-from-Llama-8B-w4a16 in calibration. The model then "knows" that specific quantization run.
Coverage will look great and mean nothing, because a real user is asking about a model that has
**zero** rows in my training set.

*Conformal Prediction Beyond Exchangeability* (Barber, Candès, Ramdas, Tibshirani, Ann. Statist.
2023) is the formal statement: under departures from exchangeability the coverage guarantee degrades
by a bounded gap measured by the total-variation distance between the observed and exchanged data
distributions. There's no free lunch — the fix is either weighting, or being honest about which
question you're answering.

**Design consequence, and the main methodological decision of this project: I will evaluate two
splits and report both.**

- **Split A (row-level, random):** the leaky one. This is what a naive implementation reports and
  it is the number I'd be tempted to put in a README.
- **Split B (grouped by model, leave-model-out):** the honest one. Every row from a held-out model
  is removed from both training *and* calibration. This answers the actual user question: "a model
  I have never evaluated."

I predict Split A shows excellent coverage and Split B shows undercoverage. **The gap between them
is the real headline of this experiment**, and reporting only Split A would be the way this
project quietly becomes fraudulent.

---

## 2. Pre-registered predictions

Writing these down now so the audits later have something to check me against.

- **H1.** Point-prediction R² under the honest grouped split will be **low, 0.0–0.35**, and may be
  negative on some folds. Skill over the mean-baseline will come mostly from *benchmark identity*
  and *base accuracy* (headroom), not from the quantization config.
- **H2.** Benchmark identity will be the strongest single feature block, because per-benchmark noise
  and per-benchmark sensitivity vary by an order of magnitude (Thread D).
- **H3.** Among config features, **weight bit-width** will dominate; W4A16 will show larger negative
  deltas than FP8/W8A8, consistent with Thread A. Model size will have a *weak* effect because the
  right variable is tokens-per-parameter, which I can't see (Thread B).
- **H4.** Split A coverage ≈ nominal (88–93% at α=0.10). Split B coverage **below** nominal, I'd
  guess 78–88%.
- **H5.** Intervals will be wide relative to the signal — half-width of **±2–4pp** at 90%. Given
  typical deltas of 0 to −3pp, an honest interval will usually contain zero. **A predictor whose
  interval always contains zero is only commercially interesting if it's sometimes narrow enough to
  rule zero out** — i.e. the useful output may be "this config is safe" rather than "this config
  costs you 1.7 points."
- **H6.** A normalized (heteroscedastic) nonconformity score — dividing residuals by the analytic
  per-benchmark noise scale from Thread D — will produce *better-shaped* intervals (narrow on MMLU,
  wide on HumanEval) at similar marginal coverage. This is my one planned "reasonable tuning
  attempt" before declaring calibration broken.

---

## 3. Design that falls out of the research

| Decision | Because |
|---|---|
| Target = `acc_after − acc_before` in percentage points, not recovery % | Recovery % explodes when `acc_before` is near zero (GPQA at 3.7 → 109.8% recovery is meaningless). Cards themselves say this. |
| Drop rows with `acc_before < 20` | Near-random baselines are pure noise (Mistral's own card excludes GPQA/MuSR/Math-Hard from recovery for exactly this reason). Audit the effect of this choice. |
| Features: `weight_bits`, `act_bits`, `log2(16/weight_bits)` compression, `is_int` vs `is_fp`, method (GPTQ / SmoothQuant / RTN / AWQ), `log(params)`, `is_instruct`, benchmark one-hot, `acc_before` | Threads A, B, D. Nonlinear in bits per Thread B; `acc_before` captures headroom; benchmark one-hot per H2. |
| Model = ridge regression first | Task says simplest thing that could work. With n≈50–500 rows and a noise-dominated target, regularized linear is the *right* capacity, not a compromise. Only escalate if it clearly underfits. |
| Baseline to beat = global mean delta | Task's own definition of "no real signal." Also report per-benchmark mean as a *stronger* baseline, because if benchmark-mean wins, the config features add nothing. |
| Conformal with the `ceil((n+1)(1-α))` index, and ±∞ when `n < 19` | Thread E fact 1. |
| Coverage measured over many random seeds, not one split | Thread E fact 2. |
| Both Split A and Split B reported | Thread E, the main decision. |

---

## 4. Threats to validity I already know about

1. **Non-exchangeability / clustered rows.** Addressed by Split B, not solved.
2. **`acc_before` is itself a noisy measurement**, so I have errors-in-variables on a key feature.
   This biases its coefficient toward zero (regression dilution). Unaddressed in this pass.
3. **Selection bias in the data source.** Red Hat publishes quantized checkpoints they consider
   *good*. Configs that destroyed a model never got a model card. So my training distribution is
   **truncated on the outcome** — I will systematically underpredict damage for a badly-chosen
   config, which is exactly the case a user most needs warning about. This is a fundamental,
   limitation of using published cards that this design cannot fix, and it's the honest answer to "what
   breaks for a stranger."
4. **Harness drift.** Cards span different lm-eval forks, prompt styles, and chat templates
   (Llama-3.1 cards note results changed after Meta modified the chat template). Base numbers are
   not comparable across cards, though the *within-card delta* mostly cancels this. Deltas are the
   right target partly for this reason.
5. **Single seed per cell.** No card gives me repeat runs, so I cannot separate measurement noise
   from true effect empirically — I can only bound it analytically as in Thread D.
6. **Parsing risk.** I'm scraping HTML tables with rowspans out of markdown READMEs. A silent
   column-misalignment would corrupt everything downstream. Mitigation: the cards print a
   `Recovery %` column, so I can **verify every parsed row against its own stated recovery** and
   reject mismatches. That's a free, strong integrity check and I'll make it a hard gate.

---

## 5. What "it worked" has to mean, given all of the above

Not "high R²". Given Thread D, high R² would actually be *suspicious* — it would suggest leakage.
What working means:

- Coverage under the **honest grouped split** lands near nominal (say 85–95% at α=0.10), on real
  held-out models, with the numbers shown.
- Point predictions beat the global-mean baseline by *some* margin under the honest split.
- Interval widths are small enough to be actionable on at least the low-noise benchmarks.

And the failure modes worth reporting are exactly the two the task names: coverage far off after
reasonable tuning, or no skill over the mean. Either would mean the claim needs narrowing to
something like *"for Llama-family W4A16/FP8 on high-n multiple-choice benchmarks, degradation is
within X points with 90% confidence"* — a much smaller but defensible product.

---

## Sources

- [Kurtic et al., "Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization (arXiv 2411.02355)](https://arxiv.org/abs/2411.02355)
- [Red Hat Developer — We ran over half a million evaluations on quantized LLMs](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms)
- [Kumar, Ankner et al., Scaling Laws for Precision (ICLR 2025, arXiv 2411.04330)](https://arxiv.org/html/2411.04330)
- [Accuracy is Not All You Need (NeurIPS 2024, arXiv 2407.09141)](https://arxiv.org/pdf/2407.09141)
- [Displacement Is Not Direction: Evaluating Fidelity Metrics for Quantized LLM Deployment](https://arxiv.org/pdf/2606.19558)
- [Barber, Candès, Ramdas, Tibshirani — Conformal Prediction Beyond Exchangeability (Ann. Statist. 2023)](https://www.stat.berkeley.edu/~ryantibs/papers/nexcp.pdf)
- [Split Conformal Prediction and Non-Exchangeable Data (JMLR 25)](https://www.jmlr.org/papers/volume25/23-1553/23-1553.pdf)
- [Probabilistic Conformal Coverage Guarantees in Small-Data Settings (arXiv 2509.15349)](https://arxiv.org/html/2509.15349v1)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (Frantar, Ashkboos, Hoefler & Alistarh, arXiv 2210.17323)](https://arxiv.org/abs/2210.17323) — the method behind every w4a16 checkpoint in this dataset
- [SmoothQuant (Xiao et al., arXiv 2211.10438)](https://arxiv.org/abs/2211.10438) — the activation-smoothing step used for the w8a8 checkpoints
- [vllm-project/llm-compressor — quantization recipes (GPTQ, SmoothQuant, RTN, AWQ)](https://github.com/vllm-project/llm-compressor)
- [Choosing the right compression algorithm — LLM Compressor docs](https://docs.vllm.ai/projects/llm-compressor/en/0.10.0.1/steps/choosing-algo/)
- [RedHatAI model cards on Hugging Face (data source)](https://huggingface.co/RedHatAI)
