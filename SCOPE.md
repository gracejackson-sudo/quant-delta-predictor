# Scope: minimal public-facing per-scheme tool

Scoped after the adversarial audit ([ADVERSARIAL_AUDIT.md](ADVERSARIAL_AUDIT.md)). This is the
**narrow** tool only — not the per-model predictor, which the evidence does not support.

## What it is

> **Input:** a quantization scheme name.
> **Output:** the calibrated 90% interval on the accuracy delta, the worst delta ever observed,
> the measured coverage rate for that scheme, and the limitations that apply.

That's it. It deliberately does **not** take a model name, because the artifact has no per-model
signal and accepting one would imply a capability that doesn't exist.

A working prototype already exists — `src/build_envelope.py` (emits the artifact) and
`src/cli.py` (consumes it):

```
$ python src/cli.py w4a16

  W4A16 (4-bit weights, 16-bit activations, GPTQ [arXiv:2210.17323])
  expected accuracy change   -0.73 pp
  90% interval               [-2.85, +1.40] pp
  worst ever observed        -8.86 pp
  based on                   191 published evaluations
                             23 checkpoints, 7 model families
  measured coverage          89% (target 90%), leave-one-family-out
  can this rule out 'no change'?  NO -- zero is inside the interval
```

The entire shipped artifact is `out/scheme_envelope.json`: a small number table, a validation
record, and named limitations (exact counts in [RANKING.md](RANKING.md), which is generated). Deliberately plain JSON so nothing is hidden in a pickle.

## Hours to ship publicly: **9–14**

| # | work | hours | why it's needed |
|---|---|---|---|
| 1 | Fix the silent data-loss bugs: demote `parse_params_b` from a hard gate to an optional field, handle cards with no `<n>B` token, and add a visible rejection report | 1.5–2 | The audit found 10 of 34 models silently dropped. A public tool cannot quietly discard a third of its input. Recovers phi-4, Mistral-Nemo, Devstral immediately. |
| 2 | Re-harvest the full ~480-card RedHatAI corpus and rebuild the envelope | 1.5–2 | Current data is 102 cards. More rows per scheme is the *only* thing that improves NVFP4, which rests on 3 families. Pipeline already generalizes; budget is for per-card format handling. |
| 3 | Add the asymmetric empirical band alongside the symmetric conformal interval | 1 | Audit §3: the raw band covers better (92.2%) and is more informative — fp8's real spread is [−1.21, +0.20], mostly downside, which the symmetric interval hides. |
| 4 | ~~Stratify by model-size band and dense-vs-MoE~~ **DONE — see [RANKING.md](RANKING.md).** Full stratification was tested and **rejected** (prospective coverage fell 90.3% → 84.4%). A one-sided widen-only variant shipped instead (LOFO 90.0% → 91.1%). The sub-2B case cannot be fixed with current data — 11 rows from 2 checkpoints — so it is disclosed via a `SIZE_UNVALIDATED` flag rather than modelled. | ~~2–3~~ done | |
| 5 | ~~Publish the worst-observed delta and the tail statistic as first-class output~~ **DONE** — both print on every scheme at every tier, recomputed each run | done | The misses are concentrated in the cases that matter. Hiding that would make the tool misleading. |
| 6 | Single-page web UI (dropdown + result card + limitations panel), or ship the CLI as-is | 2–3 | Optional. The CLI is already functional; a page mostly buys reach. |
| 7 | Write-up, reproducibility instructions, attribution/licensing for the scraped cards | 1–1.5 | Data is other people's published evaluations; provenance must be explicit. |

**Remaining after item 4: items 1, 2, 3, 5, 7 ≈ 5.5–7.5 hours** (ships the ranking CLI + JSON
artifact, no web UI). Item 2 is now the highest-value item: NVFP4 rests on 3 families and
sub-2B on 11 rows, and only more data fixes either.

## What must be on the page, non-negotiably

These came directly out of the audit and are the difference between honest and misleading:

1. **"This is a historical baseline, not a prediction about your model."** It reads exactly two
   inputs: quantization scheme and size band. It reads no family, benchmark or base accuracy.
2. **"No interval can rule out 'no change'."** 0 of 6 intervals exclude zero.
3. **"Large losses are where this fails."** The count of >3pp losses falling below the interval
   floor is recomputed every run and printed in the tool footer.
4. **"Trained only on checkpoints Red Hat chose to publish."** Outcome-truncated data, so it
   underpredicts damage from an untuned recipe.
5. **The validation provenance**: 90.1% coverage (95% CI [83.6%, 94.6%]) on 131 held-out rows
   from 5 unseen checkpoint groups, with the 87.5%–90.6% bound from gate-rejected rows stated. Those rows are mostly W4A16, whose coverage is below 90%, and FP8 and NVFP4 have no prospective rows, so the figure says nothing about them (see the paper).
6. **NVFP4 carries a warning**: 64 evaluations from 3 families, 15.6% of them losing >3pp, and
   loss-side coverage that falls to 79% on the worst held-out family (70% inside the full interval).
7. **Undercovered or thinly supported cells are flagged, not trusted**: a cell is marked
   `INSUFFICIENT_EVIDENCE` when it rests on too few checkpoints or its coverage estimate is too
   uncertain to judge (the range is still printed, demoted to Tier C), and is *refused* with
   `INSUFFICIENT CALIBRATION` and no interval only when coverage is measurably poor on adequate
   evidence (see [RANKING.md](RANKING.md) for the per-cell table).

## Explicitly out of scope

- Accepting a model name or model family as input.
- Any claim that a specific model will lose a specific amount.
- Any claim of validated accuracy for sub-2B or MoE models. Item 4 is done, and its finding is
  that these cannot be modelled with current data — they are flagged instead.
- Anything requiring raw `preds.json`, flip rates or KL-divergence — that's the 40–80 hour
  research project, and the literature suggests it may not work either.
