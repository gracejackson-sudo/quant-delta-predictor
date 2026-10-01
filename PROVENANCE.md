# Provenance of every parser and gate change

**Question being answered:** is there any other case like the Llama-4 one, where a bug fix was
informed by a row that was already in the test set?

**Answer: yes — one more, found during this audit. There are now two, both disclosed below.**

> **Note on `data/cards/` and `data/prospective_cards/`.** The raw Hugging Face
> model cards referenced below are **not shipped in this repository** - they are
> RedHatAI's content under several different licences, so only the extracted
> numbers in `data/dataset.csv` are published. Commands below that read those
> directories will not run against a fresh clone until you refetch the cards.
> [data/README.md](data/README.md) explains why and gives a script that does it.


---

## An important limitation of this record

The project directory is **not under version control** (`git ls-files` returns nothing; it sits
untracked inside the `~/Downloads` repo). So there is **no commit history to audit**, and what
follows is reconstructed from the session transcript, then verified against the code and data
where verification is possible.

Anything marked *"verified"* below was re-checked mechanically (e.g. by grepping the card corpus
to confirm which set contains the motivating row). Anything marked *"session record"* rests on
the transcript alone.

**Recommendation before any further work: `git init` the project and commit, so subsequent
changes have real provenance rather than a reconstruction.**

---

## Every change to `src/harvest.py`, in order

| # | change | what motivated it | motivating row is in | test-informed? |
|---|---|---|---|---|
| 1 | Section-narrowing made start-aware (`## Evaluation` first, then cut) | `Meta-Llama-3.1-8B-Instruct-quantized.w4a16` yielded 0 tables, because `## Deployment` appears *before* the eval section | **training** (`data/cards`) | no |
| 2 | `recovery_tolerance()` replaced a fixed ±1.0pp tolerance | 340 rows rejected as "ambiguous" — `Llama-3.3-70B-Instruct-FP8-dynamic` and others, where near-lossless rows satisfied both orientations | **training** | no |
| 3 | Family patterns boundary-anchored (`(?:^|[-_/])…(?![\d.])`) | `diffusiongemma-26B-A4B-it` matched `gemma-2` by substring | **training** | no |
| 4 | 0–1 accuracy-scale guard added | the same `diffusiongemma` cards report accuracy in [0,1] | **training** | no |
| 5 | `harvest_card()` extracted as a reusable function | refactor to share logic with the prospective test; **made before** any prospective card was downloaded | n/a | no |
| 6 | `table_layout()` + `_candidate_tiers()` — handle recovery-first column order | `Llama-4-Scout-17B-16E-Instruct-quantized.w4a16` GPQA parsed as `acc_before=100.00`, a fabricated −68pp delta | **TEST** (`data/prospective_cards`) | **YES** |
| 7 | `canon_benchmark()` strips `|`; math pattern accepts `lv/vl/v` | 11 rows lost because some cards literally print `Math-\|v\|-5` where others print `Math-lvl-5` | **training** | no |
| 8 | Previously-silent row drops now logged (`diag`) | census found 221 non-whitelisted + 12 duplicate rows dropped with no record | **training** | no |
| 9 | `math_lvl5` pattern given `(?!\d)` so it stops matching `MATH-500` | `verify/independent_check.py` found 7 rows where MATH-500 (pass@1, ~95) was stored as Math-Lvl-5 (exact-match, ~6–59) | **TEST** | **YES** |

### Verification of the "training" classifications

Each claim that a motivating row is in the training set was checked mechanically:

```
change 3,4  grep -l "diffusiongemma"  data/cards/ -> 2 files
                                      data/prospective_cards/ -> 0 files
change 7    grep -l "Math-|v|-5"      data/cards/ -> 11 files
                                      data/prospective_cards/ -> 0 files
change 9    grep -il "math-500"       data/cards/ -> 0 files
                                      data/prospective_cards/ -> 7 files
```

Changes 1 and 2 predate the existence of `data/prospective_cards/` entirely (the directory is
created by `src/real_use_case.py`, which was written afterwards) — session record, and consistent
with file creation order.

---

## The two test-informed changes, in detail

### Change 6 — Llama-4 recovery-first column order (already disclosed)
The Llama-4 cards order columns `[Recovery, base, quant]`. The fix was written after seeing a bad
Llama-4 row. **Consequence:** the 23 Llama-4 rows are excluded from the strict headline number.
They also happened to score 100% coverage, so including them flattered the result.

### Change 9 — MATH-500 vs Math-Lvl-5 (new, found in this audit)
`verify/independent_check.py`, written from scratch with no project imports, found 7 rows the
pipeline had and it did not. All 7 were `math_lvl5` rows in DeepSeek-R1-Distill and SmolLM3 cards.
Investigation showed the pipeline's regex `^math[\s\-_]*(…)?[\s\-_]*5` was matching the `5` in
**MATH-500**, a different benchmark scoring ~95 rather than the ~6–59 of Math-Lvl-5.

**Is this a problem for the result?** Less than change 6, for three reasons, but it must still be
declared:
- The fix **removes** mislabeled test rows rather than tuning anything toward the test set. It
  makes the test set smaller and cleaner, not friendlier.
- It **cannot** have changed the fitted envelope: no training card mentions MATH-500 (verified by
  grep), and the training row count is unchanged at 850 with all 27 `math_lvl5` rows ≤ 58.99.
- It was surfaced by an independent reimplementation rather than by looking at whether the result
  improved.

**Effect on the headline**: the strict set drops from 138 rows to 131, and coverage moves from
89.9% to **90.1%** at the time of that audit; the 2026-09-27 A4 float-boundary
fix subsequently moved it again to **119/131 = 90.8%** (see the "2026-09-27
A4 fix" section below). All three figures contain nominal 90%.

---

## One further disclosure the audit surfaced

The prospective model list (`UNSEEN` in `src/real_use_case.py`) was **hand-picked by me**, not
sampled at random. I chose families I expected to be interesting (small models, MoE, reasoning
distills). That is a selection choice made *before* fetching any card or seeing any number, so it
cannot have been tuned to the outcome — but it is not a random sample of the corpus, and the
coverage estimate should not be read as representative of all RedHatAI models.

---

## Net position

- Two test-informed parser changes exist; both are declared, and the affected rows (Llama-4) are
  excluded from the headline.
- The strict number now rests on 131 rows from 5 checkpoint groups — gemma-3, DeepSeek-R1-Distill,
  SmolLM, SmolLM3, NVIDIA-Nemotron-Nano — none of which motivated any parser change.
- Confirmed by an independent stdlib-only reimplementation: **118/131 = 90.1%**, with **0 delta
  disagreements and 0 verdict disagreements** against the pipeline across all 186 shared rows.

---

## 2026-09-27, A4 fix: independent_check.csv was patched, not regenerated

On 2026-09-27 the day-7 external audit flagged a float-boundary miss: one strict
prospective row (`RedHatAI/gemma-3-1b-it-quantized.w4a16` on TruthfulQA) had
`delta == +1.40` exactly and a scheme upper bound of `+1.40` exactly, but the
floating-point computation of the upper bound produced `+1.3999999999999997`,
so the closed-interval comparison returned `False`. Every comparison site in
the pipeline used the same strict comparison, so three "independent"
reimplementations agreed on the same 118/131 wrong answer.

The fix patched twelve comparison sites to use a `1e-9` tolerance on each
bound (see the shared-assumption bullet in the paper's `\section{Audit}`).

---

## 2026-09-30, ridge MAE decomposition (day-7 audit follow-up)

An external auditor asked why `pred_mae::ridge` moved from a Sep-26
stale-artifact value of 0.763916 to a Sep-30 live value of 0.776423
(+0.012507) while every other predictor moved in the fourth decimal.
Attribution by reconstructing each intermediate corpus state and
re-measuring ridge under identical BENCH_LEVELS / featurize logic:

| step (Sep 26 → live)                       | ridge move  |
|---|---:|
| family merge (Tier 0.1, 8 → 6 families)    | **+0.013460** |
| gpqa 5-way split in dataset.csv (Tier 2.5 v4) | +0.009690 |
| BENCH_LEVELS featurizer fix (add gpqa* splits) | -0.010137 |
| Mixtral MoE params_b 7 → 56 (Tier 1.6)     |  -0.000505 |
| residual (other data changes since Sep 26) |  0.000000 |
| **net**                                    | **+0.012507** |

The residual is exactly zero: the four factors above account for
every part of the move. The dominant contribution is the family
merge; the gpqa split and the featurizer fix nearly cancel (the
featurizer fix restores the benchmark identity the split removed);
Mixtral is negligible for ridge.

The BENCH_LEVELS featurizer bug is a real class-of-bug that could
have been silent-permanent: when a data label ceases to match a
hard-coded enumeration list, downstream one-hot columns become all
zero for rows carrying the new label. Test coverage was added
(`tests/test_all.py::test_hardcoded_enumeration_lists_cover_all_data_values`)
so any future data-label change that isn't reflected in
`BENCH_LEVELS` / `METHOD_LEVELS` / `KNOWN_SCHEMES` / `BANDS` fails
loudly instead of silently reducing feature expressiveness.
A subsequent audit on the same day found six more sites that had also kept
the strict comparison (`src/model.py::evaluate`, three sites in `src/audit.py`,
three in `src/interval_shape.py`, two in `src/adversarial_audit.py`, the
`one_sided_pct` line and the `below`/`above` counters in `src/one_sided_audit.py`,
and the two-sided line in `verify/independent_rank.py`); those were patched at
the same time. `tests/test_all.py::test_every_src_and_verify_module_parses_and_compiles`
was added so a syntax break in any src/ or verify/ module (as happened with
`src/validate_strata.py` after the first A4 patch) cannot slip past the gate
because no test happens to import that module.

Because `data/cards/` and `data/prospective_cards/` are not shipped in this
repository, `verify/independent_check.py` could not be re-run against the raw
card corpus at fix time. Instead, `out/independent_check.csv` was updated by
recomputing the `inside` column from the existing `lo`, `hi`, and `delta`
columns with the same `1e-9` tolerance the code now applies. Exactly one row
changed (`gemma-3-1b-it-quantized.w4a16` on TruthfulQA, `False -> True`).
`out/real_use_case.csv` -- the CSV that `paper/audit_paper.py` re-derives from --
was patched in the same way on the same row (its `inside_90` column also
flipped from `False` to `True` for exactly one row), so both CSVs and the
paper now agree on 119/131 = 90.8%. Before this second patch,
`paper/audit_paper.py` was reading the pre-fix 118 from `out/real_use_case.csv`
and printing OK against a hard-coded 90.1 literal, which the day-7 audit
identified as a vacuous re-derivation (see item 2 in the same audit).

**A patched artifact is not the same as a regenerated one.** If you fetch the
raw cards and run `python verify/independent_check.py`, the resulting
`out/independent_check.csv` should be byte-comparable to the committed version
on the `inside` column. The test
`tests/test_all.py::test_independent_check_csv_matches_tolerance_recomputation`
enforces that the committed CSV agrees with a from-scratch tolerance-based
recomputation of `inside` from its own `lo`, `hi`, and `delta` -- so a future
regeneration that produces a different `inside` value will fail the local gate
rather than silently overwriting the audited result.

## 2026-09-30, GPQA protocol split ported into the harvester

The five card-verified GPQA protocol labels, plus the `gpqa_ambiguous_46`
placeholder, were applied to `data/dataset.csv` as a direct CSV patch and were
never taught to `src/harvest.py`. A re-harvest therefore collapsed all 34 rows
back to a single `gpqa`. Since the methodological round forces a regeneration,
that would have silently reverted the split.

The split is now in the parser: four labels resolve from the card's own label
text, and the fifth needs a cross-card pass (`harvest.reconcile_gpqa_main`),
because three Llama-3.3-70B-Instruct cards carry the identical string
`GPQA (0-shot)` with baselines that disagree by 14pp. A re-harvest now
reproduces the committed CSV on every column, and
`tests/test_all.py::test_reharvest_reproduces_committed_dataset_on_every_column`
fails if that stops being true — on label columns, not only numeric ones,
which is the comparison that let the original divergence through.

**Row order is not reproducible, content is.** `harvest.main()` writes rows in
card-iteration order, which does not match the committed file's order. This
predates the port (checked against the pre-port parser). The test sorts before
comparing. Anyone diffing a regenerated `dataset.csv` byte-for-byte against the
committed one will see a reordering and no content change.

### Three decisions deliberately deferred to the methodological round

These were found while porting the split. All three move published numbers, so
none was taken two days before submission.

1. **Whether to admit 4 rows the de-duplication currently discards.** The
   harvester keeps a coarse GPQA slot (`harvest.DEDUP_FAMILY`) so that one card
   contributes at most one GPQA row, exactly as when the label was collapsed.
   Three Qwen3 NVFP4 cards and three Mistral-Small-3.1 cards each publish two
   GPQA tables under what are now distinct labels. Removing the coarse slot
   admits them and takes the raw corpus from 850 rows to 854, moving `n_rows`
   and every figure downstream of it. The plant test
   `test_reharvest_reproduces_committed_dataset_on_every_column` guards this
   boundary: it reports the row-count change and names the slot.

2. **`n_items` is wrong for `gpqa_diamond`.** Every GPQA variant carries
   `n_items` = 448, which is the Main item count. GPQA-Diamond has 198
   questions, so the 6 diamond rows carry an item count that is too large and
   their analytic noise floor (`model.noise_scale`) is correspondingly
   understated. Correcting it moves every normalized-conformal figure, so it
   belongs with the other changes that move numbers.

3. **`out/adversarial_audit.json` is stale, and regenerating it moves a figure
   quoted in `SCOPE.md`.** Re-running `src/adversarial_audit.py` moves the
   gate-rejected coverage range from 87.5%–90.6% to 88.0%–91.1%. `SCOPE.md`
   item 5 quotes the old range as current provenance, and `SCOPE.md` carries no
   claim tags and sits outside `src/audit_traceability.py`'s outward-document
   list, so no gate catches the drift. The artifact and the document have to be
   fixed together. Three other artifacts regenerate with changes from the same
   run and were likewise left alone: `out/census.json`,
   `out/census_rows.csv`, `out/adversarial_passfail_193.csv`.

## 2026-09-30, MMLU family: the harness task-name suffix was silently dropping rows

`canon_benchmark` accepted `arc_challenge_llama` and `gsm8k_llama` but rejected
`mmlu_llama`, because the MMLU pattern anchored a trailing word boundary
(`^mmlu\b`) and `_` is a word character. ARC and GSM8K had no such boundary, so
the same convention worked for them and not for MMLU. `verify/independent_check.py`
caught it: its training-row cross-check reported 818 rows against the pipeline's
817, and the per-benchmark breakdown named `mmlu`. The verifier was right.

Two rows were being dropped, both from `RedHatAI/Llama-3.3-70B-Instruct-NVFP4`,
the only card in the corpus that uses the harness task-name form for MMLU:

| card label | before -> after | delta | resolves to |
|---|---|---|---|
| `mmlu_llama` | 83.40 -> 81.28 | -2.12 | `mmlu` |
| `mmlu_cot_llama (0-shot)` | 86.42 -> 84.77 | -1.65 | `mmlu_cot` |

Both sides were changed from the card convention rather than by copying one
into the other: `src/harvest.py` gained a named `_HARNESS_SUFFIX` applied across
the MMLU family, and `verify/independent_check.py` now applies its own
pre-existing `_LLAMA_SUFFIX` constant to `mmlu` and `mmlu_cot` as it already
did to `arc_challenge` and `gsm8k`. The two implementations agree on all 21
real MMLU-family labels found on the cards, including rejecting the seven
localised variants (Portuguese, Spanish, Italian, German, French, Hindi, Thai),
which are different benchmarks and stay out.

Corpus: 850 -> 852 raw rows, 817 -> 819 in the modelling set. 102 of 736
registry keys moved. No verdict flipped (refused stays 0, insufficient
evidence stays 6 of 17). **The prospective headline is unchanged** at
119/131 = 90.8%: all 47 `prosp_*` keys are byte-identical.

### Interaction worth knowing about: the name-gate hides a third row

`RedHatAI/Phi-4-mini-instruct-FP8-dynamic` also uses `mmlu_llama`. It does not
appear in the corpus because it is one of the nine pre-registered prospective
repositories rejected by the `parse_params_b` name-gate (no `<n>B` token in the
checkpoint name; see the S9 bullet on the name-gate). **If that gate is ever
removed or relaxed, that card gains an MMLU row too**, on top of the rows the
gate currently withholds. Anyone revisiting the name-gate should expect the
prospective row count to move by more than the gated-card count alone would
suggest, and should re-measure rather than assume.

### What was deliberately NOT regenerated

`out/adversarial_audit.json` and `out/census.json` are now stale with respect
to this corpus. Neither is read by the claims registry. Regenerating
`adversarial_audit.json` moves the gate-rejected coverage range quoted in
`SCOPE.md` item 5, and that artifact and that document are to be corrected
together rather than separately; see the deferred-decisions list above.

## 2026-10-01, gpqa_diamond n_items: 448 corrected to 198

GPQA-Diamond is a 198-question subset. The collapsed `gpqa` label carried
Main's 448 items, and the protocol split inherited it, so six `gpqa_diamond`
rows claimed an item count more than twice the real one. `model.noise_scale`
is `100*sqrt(2*p*(1-p)/n)`, so their analytic noise floor was understated by
`sqrt(448/198) = 1.504`x.

`n_items` has exactly two downstream paths: the `analytic_noise_pp` column of
`model.featurize`, which only the ridge and gradient-boosting predictors
consume, and the `normalized=True` conformal variant. No registry key reads
`out/final_results.json` or `out/diagnostics.json`, so the normalised-conformal
figures are computed but unpublished; `nvfp_marginal_coverage_pct` reads the
marginal predictions and is unaffected. Every empirical noise figure
(`mae_floor_pp`, the `noise_*` keys, `GsmNoiseSd`) is estimated from the spread
of near-lossless deltas rather than from `n_items`, and none moved.

Measured before landing, confirmed after:

| figure | before | after |
|---|---|---|
| mean analytic SE, corpus | 1.6501 | 1.6621pp |
| `pred_mae::grad_boost` | 0.775945 | **0.785271** |
| `pred_mae::ridge` | 0.780341 | 0.780018 |
| `pred_mae::ridge_tuned` | 0.714370 | 0.714143 |
| `pred_mae::grad_boost_tuned` | 0.717681 | 0.717868 |
| `pred_max_abs_gap_pp` | 0.043098 | 0.043325 |
| normalised-conformal regime-C coverage (unpublished) | 90.155% | 90.336% |
| `mae_gain_pp`, `mae_floor_pp`, `n_rows`, all `noise_*` | — | unchanged |

**The tuning protocol was re-run, not reused.** `n_items` feeds a feature, so
the hyperparameters selected under the old value would have been stale. The
pre-registered protocol in `TUNING_PREREGISTRATION.md` was executed again
unchanged — same grids, same folds, same seeds — which is running the
registered protocol on corrected data rather than retuning. All six outer
folds selected identical hyperparameters, so the grid-edge finding survives
intact: ridge at the grid top in 5 of 6 folds, gradient boosting at the
slowest learning rate and heaviest L2 in 6 of 6.

All six assertions of the predictor-ordering gate still hold. The untuned
gradient-boosting losing margin *widens* from +0.0185 to +0.0278pp, so the
retraction's "true of untuned models" clause strengthens.

### A sensitivity now recorded in the paper

`pred_mae::grad_boost` moved +0.0093pp on six rows of 819 — about forty times
any other predictor's movement and roughly a third of the headline effect. It
had already moved 23% on the two rows the `mmlu_llama` fix added. Twice is a
property of the estimator, not a coincidence, so §5 now carries a caveat
naming both instances and stating that only the sign of that figure is
claimed. The two measurements are registered (`gb_sens_*`) so the sentence is
macro-backed.

### Not corrected here, and why

`gpqa_diamond_cot_5shot` measures the same 198-question subset and still
carries 448. It is one row (Mistral-Small-3.1-24B-Instruct-2503 FP8-dynamic,
baseline 45.96), and its analytic SE would move 3.3298 -> 5.0088pp. The defect
is identical and the fix is one constant. It is held for the de-duplicated-rows
change rather than regenerating the artifact chain and re-running the tuning
protocol a third time for a single row. Leaving two Diamond labels with
different item counts is a known inconsistency in the interim, recorded here so
it cannot be mistaken for a judgement that the second label is correct.

### Five stale findings flushed in the same regeneration

Regenerating `out/adversarial_audit.json` moved six figures, only one of which
the `n_items` correction can explain. The other five predated it: the A4 fix
and the GPQA protocol split had both landed without anyone regenerating this
artifact, so its findings described a corpus two changes old.

| finding | stale | current |
| --- | --- | --- |
| gate-rejected coupling bound | 87.5%-90.6% | 88.0%-91.1% |
| adversarial pass count | 168/186 | 169/186 |
| gemma miss count | 4 of 6 | 3 of 6 |
| naive interval width | 11.0 | 10.6 |
| clustered interval width | 11.5 | 9.0 |
| duplicate-benchmark accounting (`out/census.json`) | 12 | 6 |

This is the same failure the artifact-staleness gate exists to catch: a
generated file whose inputs moved and whose outputs did not. The gate covers
the registry's reading artifacts against `data/dataset.csv`, and these files
sit outside that check. Worth closing after Friday; recorded here because the
numbers in question were quoted in `SCOPE.md` while stale.

## 2026-10-01 — deferred item 2: the de-duplicated GPQA Diamond rows admitted

The harvester kept a coarse GPQA de-duplication slot (`DEDUP_FAMILY = {"gpqa":
"gpqa"}`) so that porting the five-way protocol split changed labels only and
not which rows survive. That slot is retired here. GPQA Diamond is a
198-question subset and GPQA Main is a different question set, so a card that
publishes both has published two measurements, and the split made the card's
own distinction visible in the label. De-duplication is now per protocol
label; a label that genuinely repeats within one card (`math_lvl5`, 6 rows) is
still first-table-wins.

Folded into the same change: `gpqa_diamond_cot_5shot` 448 -> 198 item count,
held from the 2026-09-30 entry above for exactly this moment because both
changes need one regeneration of the artifact chain and one re-run of the
tuning protocol.

### Which rows were admitted

Four, across four cards, each of which pairs a Main row with a Diamond row:

| card | label | acc_before | delta |
| --- | --- | --- | --- |
| Qwen3-32B-NVFP4 | `gpqa_diamond` | 62.94 | +1.53 |
| Qwen3-8B-NVFP4 | `gpqa_diamond` | 59.90 | -5.08 |
| Mistral-Small-3.1-24B w4a16 | `gpqa_diamond_cot_5shot` | 45.96 | -1.01 |
| Mistral-Small-3.1-24B w8a8 | `gpqa_diamond_cot_5shot` | 45.96 | -4.04 |

Checked before landing: no admitted row duplicates a measurement already in
the corpus. Each of the four cards carries a Main row as well, at a different
baseline, so the pair is two protocols and not one value twice.

The third Qwen3 NVFP4 card that publishes `GPQA (Diamond, 0-shot)`,
Qwen3-14B, is *not* admitted. With the de-duplication slot out of the way its
Diamond row reaches the recovery-integrity gate, which rejects it because the
card's printed Recovery cannot be reproduced from its own printed accuracies.
It moves from `duplicate_benchmark_discarded` to `recovery_mismatch` in
`data/rejected_rows.csv`. Two independent gates, and the second one caught a
row the first had been hiding.

### The corpus move

Raw 852 -> 856, modelling 819 -> 823. Checkpoints unchanged at 38, families
unchanged at 6, so the cluster structure the bootstrap and the cluster-robust
SE rest on is identical. All four rows clear the `acc_before >= 20` floor, so
all four enter the modelling set.

### What moved

187 registry keys moved; none were added or removed, so the manifest
composition pins are unchanged. The ones that matter:

| figure | before | after |
| --- | --- | --- |
| `n_rows_raw` / `n_rows` | 852 / 819 | 856 / 823 |
| `pred_mae::global_mean` | 0.757469 | 0.769393 |
| `pred_mae::scheme_mean` | 0.725362 | 0.735402 |
| `pred_mae::ridge` (untuned) | 0.780018 | 0.795546 |
| `pred_mae::grad_boost` (untuned) | 0.785271 | 0.785679 |
| `pred_mae::ridge_tuned` | 0.714143 | 0.724824 |
| `pred_mae::grad_boost_tuned` | 0.717868 | 0.722838 |
| `mae_gain_pp` | 0.032107 | 0.033991 |
| `mae_floor_pp` | 0.507199 | **unchanged** |
| `pred_max_abs_gap_pp` | 0.043325 | 0.046556 |
| `pred_gap_vs_floor_ratio` | 11.71 | 10.89 |
| prospective strict coverage | 119/131 = 90.84% | 121/133 = 90.98% |
| prospective cluster-robust CI | [86.41, 95.26] | [86.60, 95.35] |
| `prosp_icc` / design effect | -0.0090 / 1.000 | -0.0061 / 1.000 |
| `lo::w8a8_int` / `hi::w8a8_int` | -1.710 / +1.069 | -1.809 / +1.130 |
| `worst::w8a8_int` | -3.32 | -4.04 |
| `severe_pct::w8a8_int` | 1.03% | 1.54% |
| `lo::nvfp4` | -4.146 | -4.184 |
| `severe_pct::nvfp4` | 13.64% | 14.71% |
| `n_big_losses` / below floor | 25 / 19 | 27 / 20 |
| mean analytic SE | 1.6621 | 1.6801pp |

Every ordering claim survives. The per-scheme mean still beats the global mean
and its CI still excludes zero ([0.0110, 0.0563]pp). Both untuned baselines
still lose to the global mean. Both tuned baselines still beat it, and still
beat the shipped lookup only by margins whose intervals contain zero
(+0.0106 [-0.0049, +0.0247] for ridge, +0.0126 [-0.0117, +0.0388] for
gradient boosting). `mae_floor_pp` does not move at all: the two new Diamond
labels still carry fewer than four near-lossless rows each, so the eight
benchmarks excluded from the noise floor are the same eight.

The tuning protocol was re-run, not reused, for the same reason as in the
`n_items` entry: the corpus changed, so recorded selections would be stale.
Ridge still selects the top of the grid in 5 of 6 folds, and gradient boosting
still selects the slowest learning rate and heaviest L2 in 6 of 6, so the
grid-edge finding survives a second corpus correction. One non-edge
hyperparameter moved on one fold (gemma-2's `min_samples_leaf`, 30 -> 15).

### Two things worth flagging rather than burying

**The headline interval moved in our favour, by mechanism and not by choice.**
Both new prospective rows are inside their intervals, so strict coverage rose
from 90.84% to 90.98%. We did not choose the rows; admitting them was a data
decision taken before any of these numbers were computed, and the
pre-registration for it named the corpus move, not an expected direction.
Against that, the figures that moved *against* us: the w8a8 band widened by
about 7% and its published worst case went from -3.32pp to -4.04pp, NVFP4's
severe-loss rate rose a point, and the share of >3pp losses falling below the
interval floor went from 3.2% to 3.4% of all rows. The admitted rows are
high-magnitude, so they make the product more conservative and the predictors
worse.

**`pred_max_abs_gap_pp` is now 0.0466pp against a 0.05pp claim.** The abstract
quotes the computed bound rather than the round number, so nothing is
overstated, but the predictor-ordering gate asserts `< 0.05` and
`ratio >= 10`, and the ratio has gone 11.71 -> 10.89 across two corpus
corrections. One more correction of this size would put the 0.05 framing under
pressure. The framing should be revisited on its own terms rather than
defended by a tightening margin.

### The verifier

`verify/independent_check.py` had a single coarse `gpqa` label, which also
de-duplicated coarsely. That made it agree with the pipeline for the wrong
reason: `out/real_use_case.csv` was itself stale from before the protocol
split, so two coarse views agreed and the comparison printed full agreement.
Regenerating the prospective artifact exposed it.

Changed from the specification on both sides rather than by copying. The
pipeline resolves GPQA with an ordered table of whole protocol names. The
verifier now resolves three independent axes the card label states --- question
set, scoring, prompting --- and composes the name from them (`gpqa_protocol()`).
Same labels, different route. After the change: 823 rows on both sides, 188
prospective rows on both sides, zero rows found by only one side, zero delta
disagreements, zero verdict disagreements, exit 0.

### Stale hand-typed figures this surfaced

`README.md`, `ACCOUNTING.md` and `data/README.md` each carried a raw row count
of 850, two corpus changes behind, while `TOOL_SUMMARY.md` and the paper said
852. None of the three is in the claim registry's `DOCS` list, so nothing
checked them. `README.md`'s figures even carry `<!-- claim: ... -->` tags,
which look verified and are not, because the registry only scans the generated
docs. All three are corrected here. `ACCOUNTING.md`'s duplicate-discard
section is rewritten, since its worked example is one of the rows now
admitted.

`SCOPE.md` item 6's "79% / 70%" worst-held-out-family NVFP4 coverage figures
appear nowhere else in the repository and are computed by nothing. They
reproduce as 78.6% one-sided and 71.4% two-sided on the pre-change corpus
(Llama-3 held out, 14 rows, per-scheme conformal band fitted on the other five
families), so they were real but loosely rounded. They are now 85.7% and
78.6%, and the line says so, says they are hand-computed, and says they moved
because the band widened rather than because prediction improved.

Adding `README.md` and `ACCOUNTING.md` to the registry's `DOCS` list would
close this class rather than these instances. Not done here: it is a separate
change to the gate, and a gate change belongs in its own commit.

### Four pre-existing problems this change surfaced

None of these is an effect of admitting the rows. They were all true at
`8d1b2be` and were found by running the chain and the audits end to end.

**1. `src/publisher_census.py` re-samples Hugging Face live, and today's sample
breaks the registry.** It queries the HF model API for each publisher and
rewrites `out/publisher_census.json` and `out/publisher_cards/`. Run today it
returns a different 260-card sample than the one behind the committed
artifact: the publisher landscape has drifted. In that sample no
non-RedHatAI publisher prints a Recovery figure, so `card_quality.py`
recomputes `per_publisher` with RedHatAI alone and the registry raises
`ValueError: max() iterable argument is empty` at `cq_rec_best_other`. Anyone
following the regeneration commands in `README.md` in order breaks the
registry. Neither this script nor `card_quality.py` reads `data/dataset.csv`,
so neither belongs in a corpus-change chain run at all; the committed
artifacts are a dated snapshot and should be labelled as one. This is the
dangerous one of the four, because it fails for a reproducer and not for us.

**2. `src/audit.py` prints `1 hard failures` and exits 0.** The hard failure is
real: `every demo hold-out model comes from a family absent from both train and
calibration`. Nothing gates on the exit code and the test suite does not run
the script, so the audit has been reporting a failure nobody sees.

**3. `src/demo_holdout.py` crashes on its own leak assertion.** Definition B
merged Llama-3.1/3.2/3.3 into one `llama-3` family; two of the five hold-out
models in `HOLDOUTS` (Llama-3.3-70B-Instruct w4a16 and Llama-3.2-3B-Instruct
FP8-dynamic) are therefore in a training family, and the assertion fires
correctly. The script is a casualty of Definition B that nobody re-ran. It is
listed in `README.md`'s reproduce section, writes no artifact, and has no
downstream consumer, so the damage is to reproducibility rather than to any
figure. Problems 2 and 3 are one defect seen twice.

**4. The artifact-staleness test covers eleven artifacts and misses the four
that have actually gone stale.** `test_registry_reading_artifacts_are_not_older_than_dataset_csv`
checks `predictor_comparison`, `diagnostics`, `final_results`,
`interval_shape`, `interval_shape_finite`, `bias_correction`,
`bias_correction_empirical`, `cell_coverage`, `scheme_envelope`,
`one_sided_audit` and `audit_ranking`. It does not check
`real_use_case.csv`, `tuned_baselines.json`, `adversarial_audit.json` or
`census.json` --- the four whose staleness has now been caught by hand twice,
in the 2026-09-30 entry and again here.

**A fifth thing, found while landing.** `src/build_envelope.py` consumes
`out/real_use_case.csv`: the shipped envelope embeds a `validation` block with
the prospective row counts and coverage. My chain run put
`build_envelope.py` before `real_use_case.py` and baked the 186-row figures
into an 823-row corpus; `test_the_shipped_envelope_matches_a_fresh_build`
caught it, which is the system working.

To correct something I first wrote here: README's order was *not* wrong about
these two --- it already had `real_use_case.py` before `build_envelope.py`,
and the mistake was mine. Checking it properly turned up a larger problem.
README's reproduce block omitted `cell_coverage.py`, `bias_correction*.py`,
`interval_shape.py`, `one_sided_audit.py`, `audit_ranking.py`,
`tune_baselines.py`, `tune_baseline_cis.py`, every `gen_*_doc.py` and
`paper/gen_numbers.py`, and it ran `pytest` in the middle rather than last ---
so a first-time reproducer would hit the freshness test against artifacts the
documented chain never regenerates. The dependencies were also written down
nowhere, which is the state in which a reasonable person reorders and gets a
stale artifact instead of an error. README now carries one complete ordered
block, states the three dependencies that have actually bitten, and says which
two scripts not to run.

And a sixth: `paper/claims.json` was written by nothing, read by nothing, and
committed with values from 2026-09-30. An orphan that looks like the registry
is worse than a missing file, so it is deleted. The registry is
`src/verify_claims.py` and the generated macros are `paper/numbers.tex`.

### The five-failure event, explained

`paper/audit_paper.py` reported five simultaneous failures once on 2026-10-01
and zero on every run after. Five at once in one check family is not a flake
shape, so it was chased rather than filed. It reproduces exactly and it was
correct.

Section 4 of that audit is labelled "independent re-derivation from raw
source". For checks (c) and (d) it is: the adversarial GPU run file and
`data/dataset.csv`. For checks (a) and (b) --- which are exactly five, and
exactly the five that failed --- it is not. The registry derives `prosp_n`,
`prosp_inside`, `prosp_cov_pct`, `prosp_ci_lo` and `prosp_ci_hi` from
`out/independent_check.csv`, the *independent verifier's* per-row output.
Checks (a) and (b) recompute the same five from `out/real_use_case.csv`, the
*pipeline's* output. The section is a cross-check between the two arms, not a
re-derivation from raw data.

At that moment `out/real_use_case.csv` had been regenerated and
`out/independent_check.csv` had not. The check detected a genuine divergence
between the two arms and said so five times. Reproduced by restoring the
`8d1b2be` copy of `out/independent_check.csv`: the same five failures with the
same values, 90.98 vs 90.84, 121 vs 119, 133 vs 131, and both CP bounds.

Two things follow, and the second is the one that matters.

**The headline prospective coverage is the verifier's number, not the
pipeline's.** Everything the abstract, `README.md` and `SCOPE.md` quote for
prospective coverage comes through `out/independent_check.csv`. That is
defensible and arguably the stronger choice --- the figure a reader sees was
computed by a stdlib-only reimplementation that shares no code with the
pipeline --- but it was not written down anywhere, and it makes
`audit_paper.py` section 4 the only place the pipeline's own figure is
checked against it. Both facts are now stated, in `audit_paper.py` and in
`README.md`.

**`out/independent_check.csv` was not in the artifact-staleness test**, and it
is the input to the five most-quoted numbers in the paper. A stale copy makes
the paper quote a previous corpus's coverage while `verify_claims` passes at
667/667, because the registry and the docs would be consistently wrong
together. Only `audit_paper.py` section 4 would catch it, and only because the
pipeline artifact is the fresher of the two. It is now in the staleness test,
with the planted condition verified. This is the third time the same class has
produced a real defect, and the first time it reached a headline figure.

## 2026-10-01 — round item 1: the band is split conformal

`src/rank.py` called `ConservativeStratified().fit(d)`, which put the centre
and the half-width on the same rows. It now calls `strata.calibrated_fit`,
which partitions the 38 checkpoints once (seed 0, one third to calibration),
takes centres from the fit side and widths from the calibration side, and is
the path `fit_calibrated`'s own docstring already said anything quoted to a
user must come from. Protocol fixed in advance in
`CALIBRATION_PREREGISTRATION.md`; outcome 1, split conformal succeeds.

Leave-one-checkpoint-out coverage 93.56% against nominal 90%; 20 of 20
partition draws clear the pre-registered 87.9% threshold (median 91.56%).
NVFP4's published interval goes [-4.18, +1.87] -> [-7.64, +5.17], 2.12x wider:
the in-sample band had been understating the spread of the highest
severe-loss scheme by about half. Severe losses falling below the interval
floor drop from 20 of 27 to 15 of 27. Three schemes get *narrower*;
calibration is not a blanket widening.

### One mechanism produced four defects, and will produce more

**`fit_calibrated` returns fit-partition data in every field, when only the
centre and the width should be partitioned.** Everything else in a cell record
— `n`, `n_checkpoints`, `n_families`, `worst`, `best`, `p05`, `p95` — describes
the published evidence, not the construction, and must come from the whole
corpus. Four defects in one afternoon, all this:

1. The tool printed "143 evals" for `fp8_dynamic`, which has 200, and tripped
   its own thin-data warning on the shortfall. Found by reading the output.
2. `cell_train_rows::fp8|2-10B` read 13 where the corpus has 37 — the same
   defect in `rejected`, a third container the first fix did not reach. Found
   by the claim gate, 20 mismatches.
3. The five cells that lost a calibrated width lost their claims entirely,
   because the registry iterated `by_stratum` alone. Found by the claim gate.
4. `verify/independent_rank.py` disagreed on `p05` for all six schemes, having
   restored every descriptive field except that one.

Anyone extending this will hit it again. The separation is now explicit as
`strata._DESCRIPTIVE`, with the same list duplicated deliberately in the
stdlib verifier, and `calibrated_fit` restores `rejected`, `moe_stats` and
`n_moe_checkpoints` as well as the two cell dictionaries. A fifth container
added later will reintroduce the bug; the test that catches it is
`test_verifiers_exit_zero`, because the stdlib reimplementation has to make
the same choice independently.

### A fifth defect, different mechanism

An earlier scripted edit asserted on a string that did not match
`verify_claims.py`'s parenthesised `from strata import (...)`, raised, and
exited before writing the file. `calibrated_fit` ended up imported there and
unused, so the registry kept publishing in-sample `meta` counts while
`lo::`/`hi::` came through `rank.build_table` and were calibrated. Caught
because `widen_x::nvfp4` computed to exactly 1.0. After a scripted multi-part
edit, verify that each intended change is present; a script that printed
something is not evidence that it wrote anything.

### What the independent ranking verifier gives up

`verify/independent_rank.py` is standard-library only and cannot reproduce
numpy's `default_rng`, so it reads the calibration membership from
`out/calibration_partition.json` as a declared input. It still derives
everything that list is used for: row assignment, centres, the conformal index
and quantile, the widen-only rule, the support gates, flags, tiers and rank
order. The check no longer covers "is this the partition they say it is" and
still covers "given that partition, is every printed number right". A 13-name
list is auditable by eye; a reimplemented RNG is not. Its docstring states the
split.
