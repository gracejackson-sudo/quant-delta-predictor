# Adversarial audit of the 90.2% coverage result

Run: `./.venv/bin/python src/adversarial_audit.py` · machine output in `out/adversarial_audit.json`
and `out/adversarial_passfail_193.csv`.

**Verdict: the result holds, but four of my claims were overstated and one framing was wrong.**
The corrected headline is **124/138 = 89.9%** (95% CI [83.6%, 94.3%]) on a strictly clean test
set, and the thing being validated should be called a **calibrated historical baseline**, not a
predictor.

---

## 1. Leakage and contamination

### No hard leakage
| check | result |
|---|---|
| shared model ids between train and test | **0** |
| shared base checkpoints | **0** |
| shared (model, benchmark) rows | **0** |

### But "unseen families" was overstated — finding
`parse_family()` assigns **`Llama-3.1-Nemotron-70B-Instruct-HF` → `llama-3.1`** and
**`Qwen3-30B-A3B` → `qwen3`**, and both of those families are in training. I had defined a
`TRAINED_FAMILIES` set in `real_use_case.py` and then **never applied it**.

| group | family | n | coverage | family in training? |
|---|---|---|---|---|
| deepseek-r1-distill | other | 81 | 91.4% | no |
| gemma-3 | other | 45 | 84.4% | no |
| llama-4 | llama-4 | 23 | 100.0% | no |
| nemotron | **llama-3.1** | 10 | 80.0% | **yes** |
| nemotron | other | 1 | 100.0% | no |
| qwen3-30b-a3b | **qwen3** | 22 | 86.4% | **yes** |
| smollm | other | 9 | 100.0% | no |
| smollm3 | other | 2 | 100.0% | no |

32/193 rows (17%) are new *checkpoints* in *known* families, not new families.

### A development decision that used the test data — finding
The recovery-first column-order fix in `harvest.py` was written **after** seeing the fabricated
−68pp Llama-4 GPQA row in the prospective run. The parser was therefore adapted to that card
family, and Llama-4's 23 rows are not strictly prospective. They also happen to be the group with
100% coverage, so leaving them in flatters the result.

### Corrected coverage
| test set | rows | coverage | 95% CI | contains 90%? |
|---|---|---|---|---|
| all prospective rows | 193 | 90.2% | [85.1%, 94.0%] | yes |
| − family-overlap rows | 161 | 91.3% | [85.8%, 95.2%] | yes |
| − those **and** Llama-4 (**strict**) | 138 | **89.9%** | **[83.6%, 94.3%]** | **yes** |

Strict groups: gemma-3, DeepSeek-R1-Distill, SmolLM, SmolLM3, NVIDIA-Nemotron-Nano.

### Shared preprocessing: not leakage, but it moves the number
Train and test share the benchmark whitelist, the `n_items` constants, the `acc_before ≥ 20`
filter and the recovery gate. None of these see the test target, but two of them **remove test
rows**, so they can change measured coverage:

- The `acc_before ≥ 20` filter dropped 4 test rows. Those 4 had **100%** coverage, so the filter
  slightly *lowered* the reported number; without it, coverage is 178/197 = 90.4%. Harmless.
- The recovery gate rejected **7** test rows. This is the one that matters, because a gate that
  throws away extreme rows would silently inflate coverage. Bounding it:

  | assumption | coverage |
  |---|---|
  | every rejected row would have **missed** | 174/200 = **87.0%** |
  | every rejected row would have **hit** | 181/200 = **90.5%** |

  So the honest statement for the full set is a **range, 87.0%–90.5%**, not a point estimate.
  I inspected all 7 by hand; they are card errors rather than valid extremes. The clearest:
  `DeepSeek-R1-Distill-Qwen-32B-W4A16` prints IFEval as `42.87 → 72.48` with `99.1%` recovery —
  arithmetically impossible in either direction (the ratios are 169% and 59%). Another,
  `Llama-4-Scout` GSM8K `90.45 → 90.90` claiming `100.4%` when the ratio is 100.497%, is a pure
  rounding inconsistency whose true delta (+0.45pp) would have *passed*.

### Silent data loss — finding
**10 of the 34 requested models contributed zero rows**, and I did not notice:

| model | reason |
|---|---|
| phi-4-quantized.w4a16 / .w8a8 / -FP8-dynamic | no parameter count in name |
| Phi-4-mini-instruct-FP8-dynamic | no parameter count in name |
| Phi-4-mini-instruct-quantized.w8a8 | card not found (15 bytes) |
| Mistral-Nemo-Instruct-2407-FP8 / -w4a16 | no parameter count in name |
| Devstral-Small-2507-w4a16 / -w8a8 | no parameter count in name |
| Llama-4-Maverick-17B-128E-Instruct-FP8 | no benchmark rows matched |

`parse_params_b()` requires a `<n>B` token in the model name, and `phi-4`, `Mistral-Nemo` and
`Devstral-Small` have none. **FINDINGS.md listed phi-4 among the tested families — it was never
tested.** Worse, `params_b` is not used by the shipped artifact at all, so this gate discards
valid data for zero benefit. It should be demoted to an optional field.

---

## 2. Independent recomputation of the coverage number

I reimplemented the interval arithmetic from first principles inside the audit script, without
calling `predictor.py`, so a bug there could not hide:

```
scheme            mean(pp)  qhat(pp)     n
fp8                -0.3167    0.7933    80
fp8_dynamic        -0.0568    1.3868   198
nvfp4              -0.9134    2.7834    59
w4a16              -0.7264    2.1264   191
w8a16              -0.0841    0.8959    86
w8a8_int           -0.3105    1.3305   193

independent lo matches pipeline : True
independent hi matches pipeline : True
independent pass/fail matches   : True
recomputed coverage             : 174/193 = 90.16%
delta == acc_after - acc_before : all 193 rows
```

### Hand-traceable sample of 18 of the 193
Each row re-derived from the source card text: both accuracies located verbatim in the card,
delta recomputed, interval reconstructed from the scheme mean and scheme qhat, pass/fail
re-evaluated. Full list in `out/adversarial_passfail_193.csv`.

| model | benchmark | before | after | delta | pred | lo | hi | inside | card |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B-w4a16 | humaneval | 81.10 | 80.20 | −0.90 | −0.73 | −2.85 | +1.40 | yes | ok |
| DeepSeek-R1-Distill-Qwen-1.5B-w4a16 | mmlu | 37.38 | 36.98 | −0.40 | −0.73 | −2.85 | +1.40 | yes | ok |
| gemma-3-4b-it-FP8-dynamic | gsm8k | 76.12 | 75.51 | −0.61 | −0.06 | −1.44 | +1.33 | yes | ok |
| gemma-3-4b-it-w4a16 | hellaswag | 74.96 | 73.35 | −1.61 | −0.73 | −2.85 | +1.40 | yes | ok |
| DeepSeek-R1-Distill-Qwen-32B-w4a16 | math_lvl5 | 95.09 | 95.01 | −0.08 | −0.73 | −2.85 | +1.40 | yes | ok |
| DeepSeek-R1-Distill-Llama-8B-w4a16 | winogrande | 68.51 | 68.43 | −0.08 | −0.73 | −2.85 | +1.40 | yes | ok |
| DeepSeek-R1-Distill-Qwen-7B-w8a8 | humaneval_plus | 38.50 | 37.20 | −1.30 | −0.31 | −1.64 | +1.02 | yes | ok |
| Llama-4-Scout-w4a16 | mmlu | 80.54 | 80.34 | −0.20 | −0.73 | −2.85 | +1.40 | yes | ok |
| SmolLM-1.7B-Instruct-w8a16 | mmlu | 28.10 | 28.42 | +0.32 | −0.08 | −0.98 | +0.81 | yes | ok |
| **Llama-3.1-Nemotron-70B-w4a16** | **gsm8k** | 82.94 | 86.88 | **+3.94** | −0.73 | −2.85 | +1.40 | **no** | ok |
| DeepSeek-R1-Distill-Qwen-32B-w4a16 | truthfulqa | 58.41 | 58.54 | +0.13 | −0.73 | −2.85 | +1.40 | yes | ok |
| Llama-4-Scout-w4a16 | hellaswag | 85.23 | 84.95 | −0.28 | −0.73 | −2.85 | +1.40 | yes | ok |
| gemma-3-27b-it-w4a16 | hellaswag | 85.78 | 84.97 | −0.81 | −0.73 | −2.85 | +1.40 | yes | ok |
| gemma-3-12b-it-w8a8 | arc_challenge | 68.43 | 68.43 | +0.00 | −0.31 | −1.64 | +1.02 | yes | ok |
| Llama-3.1-Nemotron-70B-w4a16 | ifeval | 74.30 | 72.02 | −2.28 | −0.73 | −2.85 | +1.40 | yes | ok |
| Llama-4-Scout-FP8-dynamic | mmlu_pro | 55.70 | 55.60 | −0.10 | −0.06 | −1.44 | +1.33 | yes | ok |
| DeepSeek-R1-Distill-Qwen-14B-w4a16 | arc_challenge | 58.79 | 58.28 | −0.51 | −0.73 | −2.85 | +1.40 | yes | ok |
| Qwen3-30B-A3B-w4a16 | truthfulqa | 56.27 | 54.76 | −1.51 | −0.73 | −2.85 | +1.40 | yes | ok |

All 18 re-derive correctly. The scoring is right; the number is real.

One thing the sample makes visible: the miss is a row where quantization **improved** GSM8K by
+3.94pp. A meaningful share of the misses are upside, not damage — which matters for how the
interval should be presented.

---

## 3. Is the "predictor" more than a historical mean and spread?

**No. It is exactly that, and I should say so plainly.**

The complete fitted artifact is **12 numbers** at the time of this audit (it is **18** in the shipped tool, after 6 size-dependent widths were added):

| scheme | mean | half-width | ⇒ 90% interval |
|---|---|---|---|
| fp8 | −0.317 | 0.793 | [−1.11, +0.48] |
| fp8_dynamic | −0.057 | 1.387 | [−1.44, +1.33] |
| nvfp4 | −0.913 | 2.783 | [−3.70, +1.87] |
| w4a16 | −0.726 | 2.126 | [−2.85, +1.40] |
| w8a16 | −0.084 | 0.896 | [−0.98, +0.81] |
| w8a8_int | −0.311 | 1.331 | [−1.64, +1.02] |

`predict(row)` is `mean(delta)` over training rows sharing `row.scheme`. The half-width is the
`ceil((n+1)·0.9)`-th smallest `|delta − that mean|`. It uses **no other feature** — not size, not
family, not benchmark, not base accuracy.

### It does not beat a raw historical quantile band
A pure historical baseline needs no model and no conformal machinery: just take the 5th and 95th
percentile of past deltas for that scheme.

| scheme | conformal interval | raw empirical band |
|---|---|---|
| fp8 | [−1.11, +0.48] | [−1.21, +0.20] |
| fp8_dynamic | [−1.44, +1.33] | [−1.20, +1.73] |
| nvfp4 | [−3.70, +1.87] | [−3.92, +1.87] |
| w4a16 | [−2.85, +1.40] | [−3.22, +1.27] |
| w8a16 | [−0.98, +0.81] | [−1.26, +0.71] |
| w8a8_int | [−1.64, +1.02] | [−1.71, +1.13] |

```
coverage, conformal (symmetric about the mean): 174/193 = 90.2%,  mean width 3.63pp
coverage, raw empirical quantile band         : 178/193 = 92.2%,  mean width 3.84pp
```

The raw band is *better covered* and only 6% wider. **So the conformal layer is not what makes
the result work.** What it genuinely adds is worth keeping, but it's narrower than I implied:

1. a distribution-free **finite-sample guarantee** under exchangeability, rather than an
   empirical observation;
2. a defined answer when a scheme has too few rows — it returns ±∞ instead of a fake narrow band;
3. the **Mondrian (per-scheme) split**, which is the change that actually mattered: it lifted
   nvfp4 from 68% → 76% and w4a16 from 78% → 89% coverage.

**Framing consequence:** "predictor" is not defensible. "Calibrated historical baseline" is.
The asymmetry of the empirical band is also more informative than a symmetric conformal interval
(fp8's real distribution is [−1.21, +0.20] — mostly downside — which the symmetric
[−1.11, +0.48] hides), so the public version should show the empirical band too.

---

## 4. A real bad-quantization case the tool fails to flag

### First, the structural fact
**0 of 6 scheme intervals exclude zero.** For every scheme, "no change" is inside the 90%
interval. The tool is structurally incapable of saying *this config will cost you accuracy*.

### Case study: 4-bit weights on a 1B model
`gemma-3-1b-it-quantized.w4a16` — the most aggressive scheme applied to the smallest model, a
realistically bad choice someone would actually make:

| benchmark | before | after | true delta | interval | inside? |
|---|---|---|---|---|---|
| GSM8K | 25.17 | 22.14 | **−3.03** | [−2.85, +1.40] | **no** |
| MMLU | 39.99 | 37.00 | **−2.99** | [−2.85, +1.40] | **no** |
| ARC-Challenge | 36.86 | 33.96 | **−2.90** | [−2.85, +1.40] | **no** |
| HellaSwag | 56.03 | 53.62 | −2.41 | [−2.85, +1.40] | yes |
| Winogrande | 58.88 | 57.54 | −1.34 | [−2.85, +1.40] | yes |
| TruthfulQA | 38.54 | 39.94 | **+1.40** | [−2.85, +1.40] | boundary |

Mean −1.88pp, **4 of 6 benchmarks outside the interval**, and the tool reports
`[−2.85, +1.40]` with **no warning** because zero is inside it. A user would be told "expected
−0.73pp, probably no worse than −2.85pp" and would lose ~3pp across the board.

### The single worst case: neither flagged nor covered
`Qwen3-8B-quantized.w4a16` on MMLU-Pro:

```
measured   : 34.57 -> 25.71  =  -8.86pp   (the card itself states 74.4% recovery)
tool says  : -0.73pp, 90% interval [-2.85, +1.40]
flagged as harmful?     NO  (interval contains zero)
truth inside interval?  NO  (-8.86 is 6pp below the floor)
```

### How general is this failure?
| | |
|---|---|
| rows with delta < −3pp, across all 1,000 rows | **29** |
| of those, above the interval's lower bound | **3 (10%)** |

**26 of 29 losses worse than 3pp fall below the interval floor.** Large degradations are exactly
where the envelope fails, and at 2.9% of all rows they are not rare. A 90% interval is by
construction allowed to miss 10% of the time — but the misses are not random, they are
concentrated in the cases that matter most. That is the honest thing to publish, and it is an
argument for reporting the *worst observed delta* per scheme alongside the interval, which the
prototype CLI now does.

---

## What changed as a result

- `FINDINGS.md` carries a correction block and the strict 89.9% figure.
- The word "predictor" is replaced by "calibrated historical baseline".
- `src/build_envelope.py` emits the artifact as **plain JSON** (12 numbers then, 18 now + the validation
  record + 6 named limitations) so a public version has nothing hidden in it.
- `src/cli.py` refuses to accept a model name, reports the worst observed delta, states whether
  the interval can rule out "no change" (it never can), and warns on NVFP4's 76% coverage.
- `parse_params_b()` should stop being a hard gate — tracked in [SCOPE.md](SCOPE.md).
