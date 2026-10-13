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

**2. `src/audit.py` reports a real hard failure that nothing watches.** The
failure is `every demo hold-out model comes from a family absent from both
train and calibration`, and it is item 3 below seen from the other side.

Correcting what this entry first said: it claimed `audit.py` exits 0 while
reporting the failure. That was wrong. It does `return 1 if FAILS else 0`
under `sys.exit(main())` and exits 1, correctly. The error was in the
measurement, not the script -- the exit code was read with `echo $?` after a
pipeline ending in `tail`, which reports `tail`'s status and not Python's. A
second measurement with the output redirected to a file gives exit 1.

The real defect is narrower and still worth fixing: nothing runs `audit.py`.
It is not in the test suite and not in CI, so a correct non-zero exit reaches
nobody. A check that fails properly into an empty room is the same class as a
check that passes while verifying nothing.

**Both fixed 2026-10-01.** `HOLDOUTS` now contains only models from the three
families that are in neither `TRAIN_FAMILIES` nor `CAL_FAMILY` --- mistral,
gemma-2 and qwen3 --- replacing the two Llama entries with
Mistral-Small-3.2-24B NVFP4 and Qwen3-32B FP8-dynamic, chosen on row count and
scheme spread rather than on what they produced. `demo_holdout.py` exits 0,
and `audit.py` then reports `0 hard failures`. A new test,
`test_the_audit_scripts_exit_zero`, runs both, so the next time either breaks
somebody hears about it. Planted the condition rather than trusting the pass:
putting a training-family model back into `HOLDOUTS` fails the test with
`src/demo_holdout.py exited 1`.

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

Leave-one-base-model-out coverage 93.56% against nominal 90%; 20 of 20
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

### Fix: publisher_census.py can no longer break the registry by accident

Both halves of the 2026-10-01 breakage are closed.

`src/publisher_census.py` refuses to overwrite `out/publisher_census.json`
without `--refresh`, and the refusal explains that the script samples the
Hugging Face API live, that the committed artifact is a dated snapshot rather
than a derivation, that it does not read `data/dataset.csv` so a corpus change
cannot make it stale, and that overwriting also rewrites
`out/publisher_cards/` which `card_quality.py` reads.

`src/verify_claims.py` no longer raises a bare
`ValueError: max() iterable argument is empty` when `out/card_quality.json`
contains only RedHatAI. It names the likely cause, gives the restore command,
and gives the deliberate-snapshot path. Verified by planting a single-publisher
`card_quality.json` and reading the message.

Scanned for the mechanism rather than the instance: nine other `max`/`min`
reductions over filtered comprehensions exist in the registry. Every one of
them iterates corpus-derived data, which is non-empty whenever the corpus is,
and that is already pinned by the row-count tests. The one that broke was the
only reduction sourced from a live-sampled artifact. That is the boundary, not
"we got lucky with the rest".

### Fix: the prose scanner is now a check rather than a habit

`src/check_prose.py`. The ad-hoc version of this scan failed its own planted
defects three times in one day: `re.I` defeated the case-sensitivity the
patterns depended on and it flagged all 249 normal sentence boundaries; the
dangling-conjunction pattern did not match the contribution-2 defect it had
been written for; and a plant was miscalibrated against a pattern that had
been removed. A scanner that needs recalibrating every time it runs is not a
check.

So the plants are the primary artifact and the patterns are subordinate to
them. `DEFECTS` holds eleven real defects, verbatim, most of which reached a
paper variant. `CLEAN` holds ten strings of real prose that an over-broad
pattern flagged at some point. The contract is that every defect is caught and
no clean string is, and `test_prose_scanner_catches_its_own_plants` enforces
both halves; a second test scans the two variants, kept separate so "the
scanner is broken" and "the paper has a defect" cannot be confused.

Writing it this way round immediately found three flaws in patterns that had
looked fine: a repeated *phrase* is not a repeated word (the plant's repeated
unit was three words, "and the width"); the decimal exemption was matched
against a 140-character window and blanketed a real comma defect sitting near
"2.68pp"; and stripping `\paragraph{...}` titles before scanning made the
"prose running into a heading" pattern unable to fire at all. Each pattern now
carries `strip_headings` and `honour_exempt` flags because one uniform
pipeline provably cannot catch all eleven.

Both failure directions planted: adding a defect no pattern matches fails with
`PLANT NOT CAUGHT`, and widening a pattern until it flags real prose fails
with `FALSE POSITIVE on clean text`. The test also refuses to let the plant
set shrink below ten defects and eight clean strings, because deleting a plant
to make the scanner pass is the failure mode it exists to guard.

## 2026-10-02 — the prospective figures were measuring the retracted band

Round item 1 moved the shipped band to split conformal but left every arm that
*measures* it building in-sample. Three of them, found one after another:

- `src/real_use_case.py` — `Conformal(...).fit(m, train)`, calibrated on the
  same rows the centre came from, and using `SchemeMean` rather than
  `ConservativeStratified`, so it was the wrong construction as well as the
  wrong calibration. The script said so in a printed note; what was wrong was
  the paper quoting its output as validation of the shipped interval.
- `verify/independent_check.py` — `conformal_q([abs(x - mu) for x in g])`,
  the quantile of residuals about the mean of the same group.
- `src/adversarial_audit.py` — `frozen_predictor()`, the same pattern again,
  feeding the gate-rejected coverage range that `SCOPE.md` quotes.

**They agreed with each other, and the agreement was worthless.**
`paper/audit_paper.py` section 4 exists to cross-check the verifier's arm
against the pipeline's and reported zero disagreements throughout, because
both implemented the same wrong premise. This is the float-boundary lesson one
level down: a comparison between two implementations of a shared assumption
cannot detect that assumption. Nothing in the check could have revealed it.

All four arms now build the shipped band, and the fourth — the adversarial
audit's from-first-principles reimplementation — reproduces every bound
exactly (`independent lo/hi matches pipeline: True`). Getting there required
matching one non-obvious detail: `fit_calibrated` applies the >=20-row/>=3-
checkpoint support gate to the FIT partition, not the whole corpus. **Nine of
the twelve size cells clear that gate on the fit side and all twelve clear it
on the whole corpus**, so three cells are unusable because the partition took
their support away rather than because the corpus lacks it. Of the nine, seven
also have >=9 calibration rows and get a width.

### What moved

| figure | in-sample | shipped |
| --- | --- | --- |
| strict coverage | 121/133 = 90.98% | 126/133 = 94.74% |
| scheme-level | 121/133 = 90.98% | 124/133 = 93.23% |
| Clopper-Pearson CI | [84.77, 95.25] | [89.46, 97.86] |
| cluster-robust CI | [86.60, 95.35] | [90.62, 98.85] |
| ICC / design effect | -0.0061 / 1.000 | +0.0187 / 1.112 |
| SE ratio | 0.899 | 1.084 |
| all prospective | 171/188 = 90.96% | 176/188 = 93.62% |
| gate-rejected range | 88.1%-91.2% | 90.7%-93.8% |
| w4a16 | 89.02% | 97.56% |
| w8a8_int | 91.67% | 86.11% |

Two reversals matter more than the headline. **W4A16 stops being the
below-nominal scheme and w8a8_int becomes it.** And the ICC turns positive
while the SE ratio crosses 1.0, so the cluster-robust interval is now *wider*
than the row-level one — which is what clustering normally does. The whole §6
paragraph arguing why it was narrower, including its two-step decomposition,
was explaining an artefact of the retracted band.

The band over-covers rather than under-covers, and the mechanism is the
finding: W4A16's calibration half-width is 2.69pp against 2.14pp in-sample,
26% wider to buy coverage it already had.

### The Gemma-3 and in/out-of-family claims are withdrawn, not updated

§6 argued that Gemma-3 sat below nominal and that small-sample groups pulled
the aggregate up. On the shipped band Gemma-3 sits *above* nominal (88.89% ->
95.56%) and the pull-up more than halves (2.18pp -> 0.87pp). Both legs
reverse. Rewriting the paragraph to argue the opposite would be fitting prose
to wherever the number landed, so the comparison is withdrawn and the
per-group composition reported instead.

The in-family/out-of-family split is withdrawn for a reason that does not
depend on the band at all. Of the 133 strict rows: DeepSeek-R1-Distill
contributes 77 from 6 checkpoints, Gemma-3 45 from 4, and SmolLM, SmolLM3 and
Nemotron-Nano 11 from 3 between them. The "out-of-family" half was
four-fifths Gemma-3 and the remainder three groups of one checkpoint each, so
a label implying a result about transfer across families was carrying a
contrast between two groups. The keys stay registered; nothing quotes their
difference.

### A guard that has to be different in kind

`out/calibration_partition.json` is deliberately absent from the
artifact-staleness test. The two stdlib verifiers take it as a declared input
so they need not reproduce numpy's generator, and requiring it to postdate
`data/dataset.csv` would couple the audit tools to the regeneration order they
exist to be independent of. Its guard is `test_verifiers_exit_zero`, which
fails if the partition it declares stops reproducing the bands the pipeline
prints. That is the property that matters, and an mtime cannot express it. The
exclusion is recorded in the test so the next person does not read it as an
oversight and "fix" it.

### Retired: the with-gated-repos coverage figure

`prosp_with_gated_inside` / `_total` / `_cov_pct` — 143 / 156 / 91.67% —
are removed. They were hard-coded from an external day-7 audit and labelled
"not locally reproducible without the 9 gated cards". Three reasons, in
increasing order of how badly they reflect on us:

1. Measured on the in-sample construction retracted above.
2. **The denominator cannot be reconstructed.** 156 is *smaller* than the 188
   ungated prospective rows. Lifting a name gate admits rows; it cannot remove
   32. So no row set definable against the current corpus reproduces it, and
   nothing in the repository records which set was used.
3. **The reproducibility label was false.** All nine gated cards are in
   `data/prospective_cards/` and have been throughout —
   `RedHatAI_phi-4-*` (3), `RedHatAI_Phi-4-mini-instruct-*` (2),
   `RedHatAI_Mistral-Nemo-Instruct-2407-*` (2),
   `RedHatAI_Devstral-Small-2507-*` (2). `load_dir(..., allow_no_size=True)`
   admits 8 of the 9 and the figure recomputes in minutes. The disclaimer was
   not a hedge about a genuine limit; it was wrong, and it read as candour.
   That is the same failure as a check that passes while verifying nothing,
   and it is the reason this entry exists rather than a one-line deletion.

Not replaced. The gate-lifted figure on the shipped band is 210/224 = 93.75%
all-rows and 160/169 = 94.67% strict, which is a different quantity over a row
set the paper has never described. Introducing it would imply continuity with
143/156 that does not exist, and a new claim is not a substitute for
withdrawing an old one.

### Both coverage figures are now registered

`prosp_cov_scheme_pct` = 93.23% on `prosp_inside_scheme` = 124 of 133 is the
headline: the band a row gets from its scheme alone. `prosp_cov_pct` = 94.74%
on 126 of 133 is what the tool prints, with the widen-only size rule applied.
The denominator is 133 either way; the two differ by which band a row gets,
not by which rows are counted, and every site that quotes one now says which.

The headline figure had been derivable but unregistered, which is not a
position this project can defend while spending a section of the paper on
denominators. Both are gated now, the manifest pin moves 321 -> 320 (three
retired, two added), and the as-printed figure remains the one all four
independent arms cross-check, because it is the one the tool shows a user.

## 2026-10-08 — the 7 October audit: four blockers and eight items

### Blocker 1: the shipped tool printed the retracted band

`src/build_envelope.py` never called `calibrated_fit`. It computed a
whole-corpus mean and the residual quantile about it on the same rows, and
wrote that into `out/scheme_envelope.json` — which `src/cli.py` reads. So
`cli.py nvfp4` printed `[-4.18, +1.87]` while `rank.py` and the paper printed
`[-7.64, +5.17]`. Built 2026-10-02, after the band change, so not staleness.
`what_it_is` still described "an in-sample residual-quantile band" while the
`validation` block beside it quoted the shipped-band figure.

`test_the_shipped_envelope_matches_a_fresh_build` passed throughout, because
it compared the artifact against a fresh build of the same wrong
construction. It is now pinned to `strata.calibrated_fit` directly, and the
plant confirms it fails on exactly the condition that passed for five days.

Every scheme's interval moved; NVFP4 by 2.12x. Descriptive fields — counts,
median, empirical band, worst observed — are unchanged, per `_DESCRIPTIVE`.

### The enumeration, which is the real fix

"All four arms now build the shipped band" was verified by fixing the three
we had found. That cannot detect a path nobody considered. Enumerating every
caller of every quantile primitive, every interval constructor and every
consumer found **eight** band-constructing paths where we had claimed four:
seven correct and `build_envelope.py` defective. Five further paths are
deliberately in-sample (the retracted-band comparisons in `verify_claims.py`
and `calibrate_bands.py`) and now carry a comment saying so, so the next
sweep does not re-derive it.

`src/validate_strata.py` was converted too: it fitted width in-sample and
`RANKING.md` quoted its conclusion. Post-conversion the conclusion is
stronger, and it is now registry-backed via `src/strata_compare.py`.

### Blockers 2-4, all one class

- **§5's clustering paragraph asserted the opposite of its macros.** It said
  the cluster-robust SE was "smaller" than the naive one (2.10 vs 1.94), that
  clustering "does not inflate this statistic" (ratio 1.084, design effect
  1.112), and that the unit change accounted for a narrowing (term -0.64).
  Rewritten; the retraction stays in the paragraph where the claim was.
- **§5 named W4A16 as below nominal** at a macro value of 97.6%. The scheme
  below nominal is W8A8-int at 86.1%, now a registered macro. `SCOPE.md:71`
  carried the contradiction in a single sentence and the audit missed it.
- **The abstract mixed the two coverage figures**, rendering 93.23% beside
  126/133 and a CI computed from 126. Twelve sites across four files now name
  which figure they mean — and "at the scheme level" was itself the wrong
  label, naming a granularity when the distinction is whether the widen-only
  size rule was applied.

### Gates added, each plant-verified

- **duplicate dict keys** (AST, over the four files that decide what the paper
  may say). Found two real duplicates, `ProspCovWeightint` and
  `BigLossBelow`; both were byte-identical so nothing rendered wrongly, but
  nothing would have said so if they had differed.
- **variant lockstep.** The two paper bodies are identical sentence-for-
  sentence for the first time, so the test is trivially green and the drift
  that produced audit item 9 is now structurally impossible.
- **the widen-only superset property, proving its own non-vacuity.** The
  property is true by construction, so a test asserting it might be incapable
  of failing. The test removes the `max()` that enforces monotonicity,
  confirms the flag goes False with 11 rows lost, and restores it.
- **functions defined and never called.** `partition_sensitivity()` in
  `calibrate_bands.py` was defined and never called from `main()` in any
  revision, while four registry keys read its output: the paper's twenty-seed
  figures were right and the pipeline that claimed to produce them did not.
- **every pack item named in `PACK_STATUS.md`**, now committed so a test can
  read it. It covered items 01, 02 and 04 and never mentioned 03, which
  carries no stamp and still asserts five retracted claims.

### A false negative in the prose scanner

The decimal exemption was matched against a 140-character window, so a
genuine defect went unreported when an unrelated decimal fell inside it — and
whether it did depended on where the line wrapped, which is why the same prose
was flagged in one variant and not the other. The exemption is now tested
against the match plus a 14-character margin. Both forms of the case are
permanent plants; the self-check is 12 plants and 10 clean strings.

### The scrub caught the file we had just committed

Committing `PACK_STATUS.md` put the GitHub handle into the anonymous
supplement three times, and `test_supplement_scrub` failed immediately. It is
excluded from the supplement build: submission-process documentation, not
research material. The gate worked on the first build after the change.

### 2026-10-08, later: "checkpoint" defined, and three structural gates

**A checkpoint is a quantized model artifact** — one (base model, scheme) pair
that was evaluated. The base model is the unquantized model several of those
derive from. Verified against `out/independent_check.csv`: 19 quantized
checkpoints from 13 base models; groups 7/8/4 and 6/4/3, both series summing.
The abstract had said 19 and then grouped by the second series, which sums to
13. Nothing was false — two objects shared one word, and no gate compared the
series because every individual figure verified on its own.
`test_the_prospective_group_counts_sum_to_their_totals` now checks both sums,
and refuses to pass if the two totals become equal, which would make it unable
to distinguish them.

**The scrub reads PDFs.** It skipped every `.pdf` by extension, so the file
most likely to carry an identity string was the one file never read. Extraction
is via `pypdf`, text joined across pages and whitespace-normalised so a
hard-wrapped byline still matches, and an unreadable PDF is reported as a
finding rather than skipped — "unknown" must not read as "clean". Proven with a
hand-built PDF whose extracted text is the byline, planted as
`paper/neurips_main.pdf`: the scrub exits 1 with
`paper/neurips_main.pdf:1 [author name] '<the byline>'` --- redacted here,
because quoting it verbatim in a tracked file is itself the leak, which the
scrub caught on the next build. That is the leak a
`[final]` build would have shipped. The plant is a permanent test.

**27 dead macros removed, now gated.** Macros defined in `numbers.tex` and
referenced by neither variant: 16 had been orphaned days earlier by the
withdrawn Gemma-3 comparison, the withdrawn in-family/out-of-family split and
the withdrawn small-sample pull-up. The registry keys are kept — still
computed, still verified, and the retraction text depends on several — but
LaTeX nothing references is how a retired figure gets quoted again later. 329
macros to 302. The gate carries its own non-vacuity check: it asserts a macro
the paper certainly uses reads as live, so a regex matching nothing could not
make the dead-list empty.

**`tools/patch.py` single-edit tracking.** A bare `edit()` call reported
`Edits already applied in this run: unknown`, and that is the path actually
used — the one property justifying the tool was missing from it. Now logged at
module level, so any failure names the boundary.

### Committing PACK_STATUS.md tripped three gates in a row

Bringing one external file under version control surfaced three real leaks it
had been carrying unwatched: the GitHub handle three times (supplement scrub),
and an absolute personal path (`test_no_secret_shaped_strings_in_tracked_tree`).
The handle is handled by excluding the file from the supplement build; the path
is redacted. A fourth gate, `test_pack_status_names_every_pack_item`, is the
one added for it. The file was written honestly and nothing mechanical read
it; the moment something did, it found three problems.

### 2026-10-08: PDF metadata, and a check that could not be built

**The scrub reads PDF metadata.** Page text cannot show the document info
dictionary or the XMP packet, and that is where an anonymized PDF usually
deanonymizes itself: hyperref writes `\author{...}` into `/Author`, and
`/Producer` routinely carries a local username or an absolute path. The scrub
had no metadata handling at all. Every field is now returned and its VALUE
scanned with the same identity patterns used on prose, so a producer that
invents its own keys is covered without the code knowing the key names. An
unreadable packet is reported as a finding — "unknown" must not read as
"clean".

Plant-verified, one case per field, with the field named in the message:
`/Author` set to the byline fires; `/Producer` carrying a home-directory path
fires on the path pattern; a `/Creator` naming a person fires; and
Overleaf-style clean metadata (`pdfTeX-1.40.25`) stays silent, which matters
because a gate that fires on an ordinary build gets turned off. Caveat
recorded: **this has not been tested against a PDF actually built by
Overleaf.** The check is producer-agnostic by construction — it scans values,
not toolchain names — but no real Overleaf output has been through it.

Writing this entry put the byline verbatim into a tracked file, and the scrub
failed the next build on `PROVENANCE.md`. Documenting a leak created one. The
quotation is redacted.

### The symmetric dead-key check: attempted, not built, and why

Dead macros were removable because `numbers.tex` states them literally. Keys
are not: the key-to-macro mapping runs through loops over (scheme, band),
(stat, scheme) and cell tuples, and the key-to-document mapping runs through
generated docs whose claim tags are themselves emitted by loops. Two static
attempts flagged **466 of 853** and then **375 of 853** keys, both including
keys provably live — `cell_rows::w4a16|<2B` is quoted through
`\RowsCellWfourSmall`, and `tune_diff::global_vs_ridge_tuned` appears in
`NEGATIVE_RESULT.md`. A gate with a 44% false-positive rate is the gate
nobody trusts, which is the failure mode this project keeps documenting.

The reliable mechanism is dynamic, not static: instrument `paper/gen_numbers.py`
to record every key it resolves and the doc generators to record every key
they tag, then diff that against the registry's output. The keys are resolved
at run time, so the record is exact. That is the version to build.

What is measured meanwhile is the bounded set this cycle created: **22 keys**
unreferenced by any macro and unmentioned in any doc, tex or test. Sixteen are
the withdrawn Gemma-3 comparison, the withdrawn in-family / out-of-family
split and the withdrawn small-sample pull-up; six (`calib_cal_ckpt`,
`calib_partitions`, `jackknife_cov_pct`, `gpqa_other_rows`, `noise_n_ge100`,
`target_sd_pp`) predate it. Whether a withdrawn claim's values should remain
computable is a judgement rather than a measurement, so the list is recorded
here and the decision is open.

### 2026-10-08 --- the audit record split, and why the key removal did not run

The Audit section is now two halves, sorted on whether a reviewer needs an
entry to judge the result. Nine error items and six paragraphs stay in
\S\ref{sec:audit}; eight items and four paragraphs move to a new
`\section{Process errors}` appendix, plus one paragraph written there to hold
the gate-blindness analysis that had been embedded in the stratification item.
Nothing was dropped: `test_the_error_record_is_complete_across_both_its_halves`
pins the total item and paragraph counts across both halves, so moving an entry
passes and deleting one fails. Both were plant-verified. By words the Audit
section falls from 35% of the main matter to 27%, with about 1,700 words moved
rather than cut.

Two gates were widened by the restructure before anything else could notice,
and both are fixed in the same commit. `paper/audit_paper.py` exempted the
retracted-phrasing patterns for any match appearing after `\section{Audit}`
anywhere in the file, which already covered the Conclusion and would now cover
the whole appendix; it is bounded to the Audit span or after `\appendix`, and
plant-verified at three sites (a planted retraction in the Conclusion now
fires; in the Audit body and in the appendix it stays exempt, correctly).
`tests/test_all.py` asserted the support-floor reasoning survives by splitting
on `\section{Audit}` and reading to end-of-file, so appendix text would have
satisfied a claim about the body; it is bounded to the body.

`src/check_prose.py` flagged `\appendix \section{Process errors}` as prose
running into a heading --- a false positive, because the pattern read the
control word `appendix` as an ordinary lowercase word. The lowercase run now
has to start a real word, and the string is pinned in `CLEAN`. All 12 plants
still fire.

**The key removal did not run, and the reason matters more than the removal.**
Re-deriving the bounded set above does not reproduce it. Excluding templated
keys, the literal unreferenced count is about 45, not 22 --- and the six
pre-existing keys named above no longer appear unreferenced at all, because
*naming them in this file* made them match a check whose criterion is "the key
string appears in a tracked file". The write-up destroyed the evidence, which
is the same mechanism as the byline that leaked by being documented.

Worse, two of those six are not dead. The paper quotes their values spelled out
as words: `calib_partitions` is 20 against "over twenty seeds", and
`calib_cal_ckpt` is 13 against "a thirteen-name list". The untraced-literal
gate scans for numerals and a spelled-out numeral is not one, so both pass
every check here while being precisely what the registry exists to prevent.
`jackknife_cov_pct` is genuinely unquoted, and its value is 88.09% pooled
against the 66.18% NVFP4 figure the paper does quote --- so a reader learns the
secondary construction failed on one scheme without learning how it did
overall. Removing keys on the strength of the check as it stands would have
deleted the trace for two live claims and buried a third. Deferred, with the
class recorded in `AUDIT_DISCIPLINE.md`.

### 2026-10-13 --- 88.2 reproduces exactly; my first reading of it was wrong

**Correcting the entry this replaces.** An earlier version of this section
said 88.2 "does not reproduce under any reading", put it "3.6 points below
every aggregate LOFO figure the current code produces", and concluded it had
been "computed under the retracted construction and never regenerated". All
three statements were wrong, and the error was mine: I compared the paper's
figure against the SHIPPED band (`ConservativeStratified.fit_calibrated`) and
against the fallback-excluded pooled figure, and the four sentences in
question use neither. They use the SUBGROUP estimator --- a per-scheme mean
centre with a Mondrian-by-scheme conformal width, calibrated on a family
disjoint from the fitting pool --- which is the construction the section 6
subgroup table has always used, and which is genuine split conformal.

**The reproduction.** Under that estimator, against `77873d9:data/dataset.csv`
(817 modelling rows):

| paper | recomputed at `77873d9` | rounds to |
| --- | --- | --- |
| 87.3 --- aggregate LOFO, Definition A | 87.2587 | 87.3 |
| 88.2 --- aggregate LOFO, Definition B | 88.1963 | 88.2 |
| 85.1 --- Llama-3.3's own fold, Definition A | 85.1064 | 85.1 |
| 92.2 --- merged Llama-3 fold, Definition B | 92.2344 | 92.2 |

All four reproduce to the published rounding. They were never computed under
the retracted in-sample band. What they had done is drift about a quarter
point as the corpus grew from 817 to 823 modelling rows, which is why they
needed regenerating --- not because the construction behind them was wrong.

**The provenance argument was wrong too, and could not have settled it.** I
also argued from commit order that the estimator post-dated the figures.
It does not: `77873d9` lands at 12:40 and `d2a90ef` at 13:24 the same day, 44
minutes later, so commit order pointed AWAY from the estimator. `d2a90ef`
added only the registry wiring and the table; `src/predictor.py`,
`src/strata.py` and `src/model.py` are byte-identical between the two
commits, so the estimator's implementation was already present at `77873d9`
and the figures could have been --- and were --- computed with it. Only
re-running the construction settled this. Provenance by timestamp is not
evidence about which construction produced a number when the construction
already exists on both sides of the boundary.

**Closeness of value, again.** The discarded reading's appeal was that 88.2
sits beside `calib_cov_min_pct` at 88.21 and `cell_boot_lo::w8a8_int|>10B` at
88.17. Those are unrelated quantities and the proximity is coincidence;
registering the figure against either would have been the defect, not the
fix. The rule this record keeps relearning: do not choose a figure's
provenance by closeness to a value. Reproduce the construction instead.

**What shipped.** All four figures, and the twenty-eight others in the same
two paragraphs and the section 6 subgroup table, are now registered in
`src/verify_claims.py` and consumed as macros, so none is hand-typed. They
are additionally recomputed from the specification, in the standard library,
against `data/dataset.csv`, by `verify/independent_check.py` --- which shares
no code with `src/` --- and the qwen3 and mistral fold intervals get an
independent cluster bootstrap in `verify/independent_rank.py`. Replanting the
pre-audit 88.2 into `paper/numbers.tex` makes the independent arm report
`DISAGREE` and exit 1; replanting the in-sample construction into the arm's
own band moves nine of the thirty-two and is likewise caught.

**What the PENDING list now holds.** The four LOFO figures are off it: they
are registered, not frozen. The list is now the twelve pre-2026-10-08 exempt
numerals with no recorded disposition (`0.03 0.36 25 31.63 33 36 43 46 46.10
67 89.6 500,000`), of which three are coverage-shaped and reach the paper.
They are reported as their own counted category in the dev gate, named in the
sentinel line as `pending=N`, and are a **hard failure under
`--pre-submission`**, which the pack build now runs. The pack cannot be
stamped while any remains.

### 2026-10-13 --- the pending set, measured rather than listed

**One of the three was never in the paper.** The gate reported three pending
figures, `31.63`, `46`, `46.10`. The predicate was
`(?<![\w.\\])` + value + `(?![\d])`, and that lookahead permits a following
`.`, so `46` matched the leading digits of `46.10`. The paper contains no
figure `46`; it contains the label `gpqa_ambiguous_46`, in which the digits
are part of an identifier. A value that is a prefix of a longer decimal
anywhere in the body reported as present, in the one category whose whole job
is to say what is still there. Fixed to `(?!\d|\.\d)` and replayed in both
directions.

**The other two reproduce exactly.** Both are `acc_before` values of specific
rows, not derived statistics:

| figure | object | recomputed from `data/dataset.csv` |
| --- | --- | --- |
| 46.10 | `acc_before` of the two Llama-3.3-70B-Instruct rows (FP8-dynamic, w8a8) labelled `gpqa_ambiguous_46` | 46.1000 and 46.1000 |
| 31.63 | `acc_before` of the same checkpoint's NVFP4 row on `gpqa_main` | 31.6300 |

Both are exact, so the paper's `$\approx$ 46.10` is more hedged than the data
warrants --- the two rows agree to the digit. The adjacent "by 14pp" is the
difference between them, which is 14.4700pp.

**The remaining nine were never in the paper either.** `0.03`, `0.36`, `25`,
`33`, `36`, `43`, `67`, `89.6` and `500,000` have zero whole-numeral sites in
the reduced body of either variant. The PENDING entry's own message said
"31.63, 46.10 and 89.6 are coverage-shaped and want checking first", which
implied 89.6 was in the paper. It is not. Twelve names were carried as
pending figures; two were figures.

### 2026-10-13 --- the 42-head word-number pass, and why only fractions landed

"about a quarter-point" shipped as an untraced figure that no check could
see. `_WNOUN` already covered `points?`; the HEAD was the gap --- the
vocabulary had no fractions, so "quarter" could not match however many
counted nouns followed it.

A superset of 44 heads the shipped vocabulary did not contain --- fractions,
magnitudes (hundred, thousands, dozens), multiplicatives (double, twofold,
tenfold) and vague quantifiers (several, most, many, both, no, a few) ---
returned **34 sites across both variants, of which 4 are figures**: `half the
schemes` (3 of 6, already stated as macros in the previous sentence), `several
points` (the gap now registered as 1.3--12.1pp), `a few hundred clustered
rows` (`\Nrows` is 823) and `few checkpoints`. The other 30 are negations
("no cell is refused", followed immediately by the macros), back-references
("both arms", "both bands"), ordinals ("a third verdict" is the third KIND)
and qualitative statements.

An 88% false-positive rate is the shape of the 818-entry noun allowlist and
the 290-entry word allowlist this project measured and refused. So only the
fraction heads were added. They fire on exactly three sites: `half the
schemes`, which was reworded to stop restating a figure the previous sentence
gives as macros, and the two `a third verdict` ordinals, which are exempted
with that reason. The other 30 heads are recorded here instead of becoming an
allowlist nobody can review.

**Three figures registered out of this**: the qwen3 bootstrap's distinct
resample count (10, = C(2k-1,k) for k=3 base models), the Llama-3 cousin
subgroup count (2), and the drift the stripper entry describes (0.2446pp,
measured by replaying the same estimator over `77873d9`'s 817 rows and
today's 823). `paper/gen_numbers.py` now also emits `\XWord` and `\XWordCap`
beside `\X` for integer counts, so a count opening a sentence is spelled from
the registry instead of by hand; `paper/audit_paper.py` inherits each word
macro's unit from its digit macro rather than re-declaring it.

### 2026-10-13 --- the 29 keys that rest on inputs a clone cannot verify

`data/CARD_MANIFEST.sha256` pins 178 of 584 card files. The other 406 ---
`out/publisher_cards` (260), `out/lora_cards` (136), `out/qwen35_cards` (10)
--- are not committed, so they cannot be hashed from a checkout. Naming the
keys they feed, with the figures:

| group | keys | the figures they carry |
| --- | --- | --- |
| `pub_*` | 14 | 260 cards inspected across 10 publishers; 38 Red Hat cards, 26 paired, 147 verified rows; 40 Intel cards, 9 paired, 0 verified rows; 222 cards from 9 other publishers, 10 paired, 1 verifiable, 6 verified rows |
| `lk_*` | 11 | pooled Pearson -0.078 and Spearman 0.207 over 6 rows; rank Spearman -0.6997 at p = 0.0534; 4 kurtosis models, 3 LoRA models, 8 LoRA pairs, 2 overlapping; forgetting -2.2pp to +0.2pp |
| `qwen35_*` | 4 | 10 publisher repos, 40 rows, 40 verified, 32 rejects |

What a clone can verify is the artifact each card set was reduced to, and that
artifact is what the registry reads: `out/publisher_census.json`,
`out/track2_lowrank.json`, `data/qwen35/published_rows.csv` and
`published_rejects.csv`. Those four are now pinned in
`data/DERIVED_INPUTS.sha256` and checked by `tools/pin_derived_inputs.py`.
One unverifiable hop remains, cards to artifact, and it is named rather than
standing in for the whole chain.

**The ordering defect is fixed.** Both re-harvest tests verify the card
manifest before reading a card, so a corpus that fails its manifest can no
longer coexist with a re-harvest that passes. Replayed by corrupting one
pinned card: the re-harvest refuses and names it.

### 2026-10-13 --- the repository name defeated the anonymisation

Scanned the shipped supplement zip for the REPO NAME rather than for the
identity strings, and found **7 occurrences across 5 files** in the committed
pack: `README.md`, `TOOL_SUMMARY.md`, `src/gen_bias_doc.py`,
`src/publisher_census.py`, `src/qwen35_published.py`.

The scrub rewrote the URL HOST and kept the path, so the anonymised form read
`https://ANONYMISED/` followed by the repository's own name -- and that name
resolves to exactly one repository on GitHub. A reviewer who searches it finds
the public repo, the author's name and the handle. The bare `cd <repo>` lines
in the quick-start blocks and a User-Agent string carried it too.

Fixed: the path is rewritten as well as the host, the `cd` form and the
User-Agent are covered, and the name is now an entry in `IDENT_STRINGS` so the
scrub refuses it. A fresh build is clean -- 656 files, zero hits for the repo
name, the author name or the handle, in entry names and contents.

**This makes the supplement rebuild a hard dependency, not just the PDF.** The
committed zip carries the string.

### 2026-10-13 --- the seven-pattern scan, with a positive control

| target | result |
| --- | --- |
| zip entry names | 0 for all seven |
| zip contents, 637 files | 3 non-zero, all explained below |
| pack PDF, text and `/Info` | 0 |
| repository anonymous PDF, text and `/Info` | 0 |
| anonymous variant `.tex` | 0 |
| **named variant `.tex` (positive control)** | **1 author-name hit** |
| **named PDF text (positive control)** | **1 author-name hit** |

The control fires, so the zeros are measurements rather than a broken scan.

The three non-zero counts in the zip contents: the programme acronym (6) is
PDF binary noise and URL-encoded emoji in third-party Qwen cards, no real
occurrence; the university name (12) is a statistics paper's URL in
`RESEARCH.md` and a function-calling leaderboard named after the university in
third-party cards, neither an affiliation claim; and the repo name (7) was the
real finding above.

**The gitignored local literal file is the chosen approach** for keeping the
scrub's patterns out of public history; the hashed-token idea is dropped.

### 2026-10-13 --- the nominal level had no key

Asked whether the paper's 90 ties to a registry alpha. **It did not.** Alpha
lived in `strata.ALPHA` and the printed 90 was a hand-typed constant on the
exempt list; the two registry keys that happen to equal 90.0 are a cell
bootstrap bound and a bias-correction entry, different objects again. Now
registered as `alpha` and `nominal_level_pct`, and the paper's seven
`\(90\%\)` sites and its one `\alpha = 0.10` are macros. Registry 370 -> 372.

### 2026-10-13 --- five folds: my justification was wrong, the figure was not

The reasoning I gave -- "`n_families` is 6, one fold per family" -- yields six
folds, not five, as pointed out. The correct statement adds the step I elided:
there are six LOFO folds, the sentence contrasts the other folds with the
Llama-3 one, and six minus the one being contrasted is five. No family is
excluded.

**No registry key equals 5.** Nine scalar keys happen to, and all nine are
different objects (`pooled_pairs_per_row`, `adv_recipes`,
`prosp_ckpts_with_miss` and so on). So the figure could not have been
registered without inventing a key for `n_families - 1`, which is why the
sentence now says "the remaining folds" and carries no count. The wording
needed no correction; my explanation of it did.

### 2026-10-12 (third) --- the leak headline was two claims, and I merged them

**Reconciliation.** "Never reached a commit" was true of MY leak and false as
a statement about the string. The string is present in the tree at **70 of
118 HEAD-reachable commits**, all of them public (`origin/main` equals HEAD,
nothing local-only), in four files that must hold it: the scrub, whose pattern
list cannot search for what it does not contain; the two test files that
exercise the scrub; and the pack checklist. Five commits add or change it:
`3de5f2c` and `c95dfd1` (2026-09-27), `0d95501` and `d1b0fe4` (2026-10-08),
`59f4cd9` (2026-10-11). What never reached a commit is the pair of files I
edited on 2026-10-12, `tests/test_all.py` and `src/audit.py`: zero commits
each by `git log -S`.

**Pattern variants, counts only, over all 118 commits.** The project folder
name, 70 commits / 4 files. The programme's full name, 70 / 2. The programme
acronym alone, 0. The author's name, 117 / 7. The GitHub handle, 118 / 6. The
university name, 115 / 8. The personal email, 73 / 2.

Those last four are the expected state of a **named public repository under
the author's own account** -- the name is in `LICENSE`, the handle in
`README.md`, and so on. The anonymity requirement is on the SUBMISSION
ARTIFACTS, not on the repository, which is why the scrub exists. At HEAD every
file holding any of these patterns is in `DROP_FILES`, with one benign
exception: three third-party model cards under `data/rh_card_scan_*` match the
university name because the Nemotron cards cite a function-calling leaderboard
named after it, which is a benchmark and not an affiliation. The scrub already
anchors its affiliation patterns on the phrases that name the university AS an
affiliation rather than on the bare word, for exactly this reason -- and the
first draft of this paragraph quoted one of those phrases verbatim, so the
scrub failed the build on it. **Fourth instance in one session** of a finding
that cannot be recorded by quoting the string: the byline, the orphan-key
list, the project folder name, and now the scrub's own pattern while
explaining the scrub's own pattern. The rule in `AUDIT_DISCIPLINE.md` wants
reading before writing, not after.

**Artifacts: clean.** The supplement zip, 637 files, zero hits for any
variant, and all four by-design holders absent from it. Zip entry NAMES
checked as well as contents. The pack PDF and both repository PDFs: zero in
page text and zero in `/Info` metadata. The anonymous variant: zero.

**Keeping the scrub's patterns out of public history.** They are in history
already and no rewrite is proposed. For the future the options, in the order I
would take them: hold the literal strings in a gitignored local file that the
scrub reads and fails loudly when absent, so the committed code carries only
labels; or commit sha256 tokens of the strings instead of the strings, which
keeps the scrub self-testing (it can still verify it catches a planted string
whose hash it knows) while the committed form reveals nothing. The second is
better for a public repository because it survives a fresh clone. Neither is
urgent: the strings being public in a named repository is not the risk the
scrub guards against.

### 2026-10-12 (third) --- the nine numerals, one by one

| value | where | disposition |
| --- | --- | --- |
| `0.9` | `\lceil (n+1)\cdot 0.9 \rceil`, the conformal index | not a figure: it is $1-\alpha$ inside a formula |
| `1` | the `(n+1)` of the same index | not a figure: formula arithmetic |
| `3` | "DeepSeek-R1-Distill of Llama-3 and Qwen-2.5 bases" | not a figure: a model-name digit |
| `88.2` | "aggregate LOFO coverage ... changes from 87.3% to 88.2%" | **needs your call**, see below |
| `90` | "against a nominal $90\%$" | not a figure: the nominal level |
| `90.01` | "Regime C mondrian coverage from 89.15% at 168 folds to 90.01% at 60 folds" | historical before/after, same class as the 89.15 and 60 beside it |
| `91.0` | "`band_emp_coverage_pct` from 93.4% to 91.0%" | **was a figure. Now `\BandEmpCov`** |
| `95` | "a 95% cluster-robust CI" | not a figure: the chosen confidence level |
| `168` | the fold count in the same regeneration sentence | historical before/after |

**The four that were exempt only because the fix revealed them**: `0.9`,
`168`, `90.01`, `91.0`.

**One was a figure and is now traced.** `91.0` is the live value of
`band_emp_coverage_pct`, which the sentence names in `\texttt{}` two words
earlier, and `\BandEmpCov` renders exactly `91.0`. Exempting it was the wrong
disposition: the literal and the macro agree today and would diverge silently
on the next corpus change. Converted in both variants and removed from
`EXEMPT`.

**One needs your decision.** `88.2` is the post-Definition-B aggregate LOFO
coverage across held-out families. No registry key holds that quantity -- the
nearest values are `calib_cov_min_pct` at 88.21 and a cell bootstrap bound at
88.17, and attaching either would be the correct-number-wrong-object defect
this record is full of. So it is either a historical before/after pair like
`90.01` (the sentence is explicitly about a change) or a current figure that
wants a new key. I have not guessed.

**`five folds` was correct before it was reworded.** `n_families` is 6, one
LOFO fold per family, so "the other five folds" was right. What the rewording
removed was a correct but untraced arithmetic derivative of a registered
figure, not an error.

### 2026-10-12 (third) --- 16 artifacts under a runner that deletes first

The earlier runner compared without removing the output, so a generator that
wrote nothing scored IDENTICAL. The fixed runner deletes the artifact, asserts
it was rewritten, and reports the pinned inputs. All 16 rewritten, all 16
reproduce:

| result | artifacts |
| --- | --- |
| identical (15) | predictor_comparison, diagnostics, final_results, interval_shape, interval_shape_finite, bias_correction, bias_correction_empirical, cell_coverage, one_sided_audit, audit_ranking, strata_compare, census, independent_check, adversarial_audit, real_use_case |
| identical modulo one field (1) | scheme_envelope, whose `/built` date moved 2026-10-07 to 2026-10-12. It is the ONLY artifact embedding a date; every other value matches exactly |
| outstanding (1) | tuned_baselines, still running |

**`real_use_case.csv` now reproduces under shipped code**, and getting there
found something. Its missing prospective card is 15 bytes and its content is
`Entry not found` -- **Hugging Face's 404 body**. The old fetch wrote that, the
>200-byte floor read it as absent, and every regeneration silently re-attempted
the download and got the same 404. No published figure is affected: Phi-4-mini
is name-gated and the paper discloses it as never scored. It is now pinned by
content hash as a known-not-a-card, so a file whose content we have identified
is not treated as missing, while a file that is merely small still is.

`src/tune_baselines.py` is seeded (`random_state=0` throughout) so its output
should be deterministic; its runtime is over 40 minutes on this machine, which
is why it belongs in a pre-submission target rather than a gate.

### 2026-10-12 (third) --- the word gate's reach, and a plant it misses

**"four weeks later" is NOT caught**, and the reason is structural. The
counted-noun vocabulary is derived from the nouns the paper counts with
NUMERALS, so a noun the paper counts only in WORDS never enters it. `weeks`
is such a noun. The derivation is reactive by construction.

What the derivation does cover: `characters`, `days`, `commits`, `runs`,
`comments`, `folds` and `cells` are all derived from numeral usage in the
current paper, so every noun I declared this week would have been surfaced by
it rather than guessed -- the gate fired first and I declared in response.

**How the gap closes, measured.** Matching a word-number followed by ANY
plural head, rather than only a declared noun, gives **60 distinct heads
across both variants, 18 already declared, 42 to decide**. That is reviewable,
unlike the 818 the same approach costs for numerals -- because numerals appear
in tables, model names and shell commands while word-numbers appear almost
only in prose. The 42 are mostly rhetorical (`things`, `reasons`, `sides`,
`halves`, `places`) with a handful of real counts (`groups`,
`implementations`, `bugs`, `disagreements`, `populations`, `versions`).
Recommended and not built: it is a 42-entry judgement pass that wants its own
commit.

### 2026-10-12 (third) --- smaller items

The three loop-ran guards derived their floor from a hard-coded 6; they now
read `n_schemes` from the registry, so a seventh scheme cannot leave them
checking 6 of 7.

### 2026-10-12 (second) --- my leak never reached a commit (see the reconciliation above)

Searched every commit in the pushed range and all history, with the pattern
built at run time so this file does not instantiate it.

Across ALL history the string has only ever existed in four files, every one
of them a by-design holder and every one in `DROP_FILES`: the scrub itself
(its pattern list must contain what it searches for), the two test files that
exercise the scrub, and the pack checklist. `tests/test_all.py`, `src/audit.py`,
`PROVENANCE.md`, `AUDIT_DISCIPLINE.md`, `README.md` and both paper variants:
**zero commits touching it, by `git log -S` per path.** So the leak I caused
on 2026-10-12 lived only in the working tree between the edit and the scrub
failing. It was caught twice pre-commit and never pushed.

The shipped artifacts are clean too: 637 files in the supplement zip, zero
hits; the pack PDF and both repository PDFs, zero hits in page text and zero
in metadata; the anonymous variant, zero. The four by-design holders are
absent from the zip, which is the drop list working.

No history rewrite is needed.

### 2026-10-12 (second) --- the blocked artifacts, and a 15-byte card

Re-ran the five blocked artifacts with the hash-verified snapshot extracted.
Four reproduce byte-identically: `census.json`, `independent_check.csv`,
`adversarial_audit.json`, and `real_use_case.csv`.

**`real_use_case.csv` took a second attempt and the first was a false
positive.** My runner labelled it IDENTICAL on a run where the script had
actually REFUSED and written nothing -- identical because untouched, which is
the shape of result this project keeps finding. The refusal was correct:
`data/prospective_cards/RedHatAI_Phi-4-mini-instruct-quantized.w8a8.md` is
**15 bytes**, empty but for whitespace, in the snapshot. Phi-4-mini is one of
the name-gated repositories the paper already discloses as never scored, so
the stub contributes nothing -- and the old code silently re-downloaded it from
Hugging Face on every regeneration rather than saying so. With `fetch()`
bypassed the artifact reproduces exactly.

`tuned_baselines.json` is still running after 25 minutes of nested
cross-validation and is reported as unmeasured, not as reproducing.

**So: 16 of 17 measured, 16 reproduce, 0 differ, 1 outstanding.** The earlier
"no stale artifact" was scoped to the 12 then measured and should have said so.

### 2026-10-12 (second) --- what the hidden text actually held

Every numeral and number-word in the 91 hidden prose spans, with its
disposition against the current gates:

**Nine numerals, all nine exempt by name, none untraced.** `3` (x4), `0.9`,
`1`, `88.2`, `90`, `90.01`, `91.0`, `95`, `168`. Four of them -- `0.9`, `168`,
`90.01`, `91.0` -- are exempt only BECAUSE the fix revealed them; the other
five were already exempt.

**Six word-number phrases, one of them a real finding.** `two-sided` (x5) and
`one-sided` (x2) are terminology excluded by WORDTERM. `one cell` and
`one scheme` are rhetorical and named in WORD_EXEMPT. `one of` is
`every one of the \CalibPartitions{} clears`, where the figure beside it is a
macro. And **`five folds`** was a genuine untraced figure, `\Nfam` minus one;
it is gone from both variants, reworded to "the remaining folds" rather than
registered, because an arithmetic derivative of a registered figure should not
become a key.

**3,253 against 3,162 reconciled, and the gap is exactly the span count.** The
broken rule removed 98 runs: the 7 real comments it was for, and 91 runs of
ordinary prose it was not. Those 91 runs hold 3,253 characters. The body came
out 3,162 shorter because the stripper left one space in place of each run,
and 3,253 - 91 = 3,162. The denominator, 103,841, is the body with the 7 real
comment TEXTS removed and a space left in each place -- so it does not include
comment text. The paper entry and the reproduce script both say this now.

### 2026-10-12 (second) --- 203 in both configurations

The 203-against-202 discrepancy was mine, not the configurations'. Both
collect **203 nodes with identical node IDs**; no test exists in one and not
the other. The corpus-extracted run totalling 202 predated
`test_the_card_corpus_matches_its_committed_manifest`, so I was comparing two
runs of different suites.

### 2026-10-12 (second) --- the word gate's allowlist, measured before replacing

Asked to replace the counted-noun allowlist with a check that flags any
numeral or number-word followed by an UNDECLARED noun, and to report the count
first. Across the paper and 18 documents: **906 distinct head nouns, 88
declared, 818 undeclared.**

The list is dominated by things that are not nouns: `yes` (150, table cells),
`redhatai` (135), `sided`, `nvfp`, `b-instruct-quantized` (58),
`b-quantized` (46), `shot`, `alpha`, `texttt`, `https`. Declaring or
exempting 818 entries is the 290-entry allowlist problem at triple the scale
and would be the gate nobody trusts, which is the failure this document is
about. **Not implemented**, with the measurement as the reason.

What the derived check already does and the proposal drops: it keeps plural
head nouns only, and abandons on a function word, a finite verb or a macro
name. That is why it yields 29 counted nouns rather than 906, and it already
fails the build when the paper begins counting something new.

### 2026-10-12 (second) --- the refactor, verified span by span

Span LISTS, not counts, are identical before and after: 104 spans in the
anonymous variant and 97 in the named one, equal element for element in rule
and text. The two inline pattern copies in
`test_each_input_stripper_keeps_what_it_must` are gone -- zero `re.*` calls in
it.

**An AST scan replaces the grep sweep**: every regex literal anywhere in
`tests/`, 32 of them. One was still a copy of a shipped stripper's pattern --
`re.sub(r"<!--.*?-->")` in the ranking-doc test -- and it now calls
`check_docs.strip_tags`. The remaining 31 are the secret scanner's own
patterns, parsers of script output, and document-structure queries; none
duplicates code under test.

**The six items from item 1, accounted for.** The two `.git`-gated silent
returns both RUN in this clone (`.git` is present and the trailer baseline
resolves), so they were latent rather than active; the shallow-clone one now
skips loudly. The two no-assertion tests have explicit assertions --
`log_request` must return None and must not create the path, and the compile
sweep must see at least 30 modules. The four registry-loop tests assert the
table holds at least the 6 schemes before looping.

### 2026-10-12 (second) --- the manifest covers 178 of 584

`data/CARD_MANIFEST.sha256` pins `data/cards` (144) and
`data/prospective_cards` (34). It does NOT cover `out/publisher_cards` (260,
14 `pub_*` keys), `out/lora_cards` (136, 11 `lk_*` keys) or
`out/qwen35_cards` (10, 4 keys) -- 406 files feeding 29 registry keys, pinned
only inside the archive, outside git. Recorded in
`data/CARD_MANIFEST.sha256.README` rather than implied covered.

**And the re-harvest does not check the hashes before running.** It reads
whatever cards are present, so a corpus failing the manifest and a re-harvest
passing can both be true in one run. The manifest test is the one to believe;
making the re-harvest depend on it is the right fix and is not done.

### 2026-10-12 (second) --- a network rate-limit was reported as a paper defect

`paper/audit_paper.py` compares the bibliography's author surnames against
arXiv over the network and recorded a failure to reach it as a WARNING. arXiv
rate-limited us mid-session (HTTP 429, then a read timeout) and
`test_both_paper_variants_pass_audit_paper_cleanly`, which requires zero
warnings, went red on a clean tree. Third instance of the state confusion
`src/audit.py` had: the script is now three-state as well, the sentinel line
carries `unchecked=N`, and the test tolerates and prints unchecked while still
failing on a real warning.

### 2026-10-12 --- a test that validated a copy of the code under test

**The span-rule test was hollow.** It was written to assert what each stripper
removed, and it held private copies of their patterns: `CMT` re-declared in
the test file, `HTML` likewise, and the claim-tag rule importing
`audit_traceability.CLAIM` directly. Planting the escaped-percent bug in the
SHIPPED `paper/audit_paper.py` left it green. Worse, `audit_paper.py` itself
then reported zero failures and zero warnings, because with 3% of the body
hidden again the figures that would have fired were out of scope. A test that
validates a copy of the code under test is not a test of that code, and the
previous commit's "plant-verified against both real defects" has been
corrected above.

**The fix is structural.** The reduction for the paper now lives in
`paper/texbody.py` as `strip_tex(tex) -> (body, spans)`, which
`paper/audit_paper.py` imports; `audit_traceability.visible_with_spans` and
`check_docs.strip_tags_with_spans` do the same for the documents, and
`audit_traceability.classify` was repeating the four substitutions inline, so
it now calls the single reduction too. `check_prose.strip_comment_lines` is
named for the same reason. Behaviour is byte-identical: all five reported
counts match across both variants before and after.

The test's predicates are character walkers with backslash parity, not
regexes, and the claim-tag predicate deliberately does not import `CLAIM` --
it walks `<!--`, `claim:`, a key, `=`, and a numeric value. Acceptance,
replayed against the shipped modules rather than copies:

```
plant: escaped-percent bug in paper/texbody.py
  removed_span_matches  -> FAIL      stripper_keeps -> FAIL
  numeral_hidden canary -> FAIL      (audit_paper.py itself still 0/0)
plant: fence/inline order swapped in audit_traceability.VISIBLE_RULES
  removed_span_matches  -> FAIL      stripper_keeps -> FAIL
```

**An end-to-end canary.** `test_the_real_audit_fails_on_a_numeral_hidden_
behind_a_percent` copies the paper, plants an untraced numeral immediately
after a printed percentage on the same line, and runs the actual script. With
the stripper correct the audit reports it; with the stripper broken the audit
passes while a hand-typed figure sits in the paper. That second state is what
shipped for 18 days.

**`foo\% real comment`, both directions**, through the shipped `strip_tex`:
the text after the escaped percent is KEPT, the text after a bare percent is
REMOVED, and exactly one span comes back, classified as a comment.

**Sweep of every plant test.** 35 tests mention a plant, canary, non-vacuity
or replay. After the refactor the only one declaring its own `re.compile` is
`test_no_secret_shaped_strings_in_tracked_tree`, whose patterns ARE the check
rather than a copy of another module's. The caveat worth stating: the sweep
finds assigned patterns, so a test inlining `re.search(r"...")` to duplicate a
shipped pattern would not show up -- which is how
`test_each_input_stripper_keeps_what_it_must` still held two inline copies
after the first pass, found by reading it rather than by the sweep.

### 2026-10-12 --- tests that could not fail, and a script that could not say "unchecked"

**Three silent returns.** `test_reharvest_reproduces_committed_dataset_on_
every_column` and `test_gpqa_protocol_labels_survive_a_reharvest` returned
bare when `data/cards` was absent, which is every clone; the co-author trailer
test returned bare when its baseline commit did not resolve, which is every
shallow clone. All three counted as passes. They skip loudly now, naming the
missing input and how to supply it.

**The gate prints the three-way split.** `tests/conftest.py` adds a GATE
SUMMARY line and `pytest.ini` adds `-rs`, so skips are always named. The
number that mattered for weeks was the one nobody printed.

**`src/audit.py` is three-state.** Its card re-read counted an absent card as
a MISS, so on every clone it reported "60 misses" and exited 1: "could not
check" reported as "checked and failed". Now 0 clean, 1 a check ran and
failed, 3 an input was absent. `test_the_audit_scripts_exit_zero` skips on 3
and still fails on 1 -- which removes one of the two persistent reds by
distinguishing the states rather than by weakening the check.

**A gate against tests that check nothing.**
`test_no_test_can_pass_without_checking_anything` requires every test to hold
an assert, a `pytest.skip`/`fail`/`raises`, or an explicit `raise`; and a test
whose assertions are ALL inside a loop must either assert the loop ran or sit
on a hand-checked allowlist with the reason its iterable cannot be empty. It
found two tests with no assertion at all --
`test_logging_failure_never_breaks_the_tool`, a bare call with a
`# must not raise` comment, and
`test_every_src_and_verify_module_parses_and_compiles`, which raises
`AssertionError` explicitly and which my first version of the gate flagged
wrongly, so the gate was too narrow before it was right. Four tests whose
loops iterate registry data now assert the loop ran.

### 2026-10-12 --- the 17 artifacts, regenerated and compared

Regenerated into a temp copy from the committed `data/dataset.csv` and
compared byte for byte. **12 reproduce, 0 differ, 5 blocked. No stale
artifact.**

| | artifacts |
| --- | --- |
| reproduces (12) | predictor_comparison, diagnostics, final_results, interval_shape, interval_shape_finite, bias_correction, bias_correction_empirical, cell_coverage, one_sided_audit, audit_ranking, strata_compare, scheme_envelope |
| differs (0) | --- |
| blocked (5) | tuned_baselines (slow: killed at 22 minutes of nested CV), adversarial_audit and census (script-skipped, absent input), real_use_case.csv and independent_check.csv (need `data/prospective_cards`) |

**The exhaustive grep behind the open gap.** Every test was searched for
`build_artifact`, `main()`, a subprocess into `src/`, `import_module`,
`runpy`, `read_bytes`, `sha256`, `md5`, `filecmp` and `json.load` over `out/`.
No test regenerates an artifact and compares it. There is no Makefile and no
CI configuration in the repository, so there is no regenerate-and-diff step
anywhere. The gap claim no longer rests on two candidates.

**The cards are now pinned in the repository.** Until today the only hashes
lived in a manifest inside a snapshot archive outside git, so a clone could
not tell whether a card corpus it obtained was the one the figures were
computed from. `data/CARD_MANIFEST.sha256` pins all 178 cards;
`test_the_card_corpus_matches_its_committed_manifest` verifies them when
present and skips loudly when not.

**`src/real_use_case.py` refuses to fetch.** A missing prospective card was
silently downloaded from Hugging Face at regeneration time, so rerunning the
chain without the cards would rebuild the headline prospective figure from
whatever the cards say today, with no hash to tell the difference. It now
raises with the count of missing cards and requires `allow_network=True`,
which is a deliberate corpus refresh.

**Both gate configurations.** Without the corpus: 3 skipped, and the skips are
named. With the 2026-09-30 snapshot extracted (hash-verified, `dataset.csv`
unchanged before and after): **201 passed, 1 failed, 0 skipped** -- the
re-harvest tests do reproduce the committed dataset from those cards, which I
had expected might fail since the GPQA protocol split came later. The one
failure in both configurations is the submission pack's stale LIVE stamp.

**The affiliation folder name leaked while I was writing these skip
messages**, and then leaked a second time into this very paragraph. The path
went into `tests/test_all.py` and `src/audit.py`, neither of which is dropped
from the supplement, and the scrub failed the build on it. I then named it
here to record that -- and PROVENANCE.md ships too, so the scrub failed again
on the write-up. Third instance of the same shape this fortnight: the byline
that leaked by being documented, the orphan-key list that made its own keys
look referenced, and this. The rule is now explicit in
`AUDIT_DISCIPLINE.md`: a finding about a forbidden string cannot be recorded
by quoting the string. Both uses are gone; the snapshot is referred to by what
it is.

### 2026-10-12 --- the frozen figures, re-measured from history

`tools/measure_percent_defect.py` lifts the four `body = re.sub(...)` lines
verbatim out of `708a84d:paper/audit_paper.py` and executes them against that
revision's paper. It exits 3 with a reason on a shallow clone.

| | anonymous | named |
| --- | --- | --- |
| escaped percent signs | 132 | 132 |
| real comment marks | 7 | 0 |
| N, body that should be read | 103,841 | 103,710 |
| body that was read | 100,679 | 100,548 |
| hidden characters | 3,162 | 3,162 |
| hidden fraction | 3.0450% | 3.0489% |
| scanned fraction | 96.9550% | 96.9511% |

All four original figures confirmed. **I was wrong to report that the paper's
`3.0%` needed correcting**: 3.0450 rounds to 3.0 at one decimal. The entry now
quotes two decimals anyway, states N, and covers both variants. The named
variant has zero real comments -- the 7 are NeurIPS style comments in the
anonymous preamble -- so there the broken rule removed nothing but prose.

`708a84d` is confirmed as the parent of `2aa90e4` (`git rev-parse 2aa90e4^`).
The exposure interval is `db34a29` (2026-09-23, the commit that ADDED
`paper/audit_paper.py` -- the defect was there from the file's first line) to
`2aa90e4` (2026-10-11): 18 days, and `git rev-list --count db34a29..2aa90e4`
= 103 commits. The paper today holds 108 escaped percent signs per variant.

**What the hidden text contained**, read by the fixed stripper at `708a84d`:
the broken rule removed 98 spans, of which 7 were real comments and **91 were
ordinary prose** following an escaped percent, 3,253 characters in all. Inside
them: 9 distinct numerals (`0.9`, `88.2`, `91.0`, `168`, `90.01`, and already-
exempt ones), and 6 word-number phrases including `five folds` and `one cell`,
both of which the word gate caught once it could see them. The longest hidden
fragments are plain argument -- a sentence about OpenLLM v1 scores, the
name-gate withdrawal passage, the noise-variance passage.

**And a figure in my own correction was wrong.** The remediation entry said
the second hand-typed site "was found five days later". `git log -S` puts the
key's registration and first conversion at `0d95501` (2026-10-08) and the
second site at `59f4cd9` (2026-10-11): three days. Surfaced by the word gate
only after `days` entered the counted-noun vocabulary, which is the derivation
earning its place.

### 2026-10-11 (fifth) --- the entry's own figure was the error it describes

**The 96.8% was a present-tense counterfactual labelled as a historical
claim.** The appendix said every clean result reported beforehand was computed
over 96.8% of the paper. That figure came from running the broken stripper
over TODAY's paper, which is the answer to "what fraction of the current text
would the broken stripper read" presented as the answer to "what fraction did
the gates read then". The tell was in the entry's own provenance note: the
count had moved from 132 to 136 as the entry was written. Correct number,
wrong object, inside the entry about a check making an unasserted claim about
its own input.

The figure is doing work in the argument, so it is now measured against the
paper as committed at **708a84d**, the last commit with the defect live, and
frozen: 132 escaped percent signs against 7 real comment marks, 3,162
characters hidden, 3.0% of the body, **97.0% scanned**. The entry says the
fraction varied as the paper grew and claims no single number for the whole
period. The five live registry keys were removed (375 -> 370) because a live
measurement answers the wrong question; the numerals are exempted by name in
`paper/audit_paper.py` with the reasoning.

**The retention band is replaced by removed-span classification.** A band
could not do the job and both of today's findings prove it. It missed its own
target: pinned 93-98%, the replanted bug scored 93.0%. And it is blind to the
fence defect entirely, because there the quantity removed was correct and only
the mechanism was wrong --- no percentage check can see that.

Each stripper now reports its removed spans with the rule that removed each,
and every span must match the construct its rule names: a comment begins with
an unescaped percent, a claim tag looks like a claim tag, a fenced span begins
and ends on a fence, an inline span is not adjacent to another backtick.
Content-independent, nothing to loosen, no drift. Retention percentages are
printed for a reader and deliberately not asserted.

**CORRECTION, 2026-10-12.** The claim below was wrong, and wrong in the way
this entry is about. Both plants were applied to the test's own private copies
of the strippers' patterns, not to the code that ships. Planting the
escaped-percent bug in `paper/audit_paper.py` itself left the span test GREEN,
and left `audit_paper.py` reporting zero problems, because the defect re-hides
the evidence that would have fired. See the 2026-10-12 entry: the reductions
now live in importable modules that the gate and the test both use, and the
acceptance test breaks the shipped module.

Plant-verified against both real defects --- against copies of the patterns,
which is the defect, not the verification:

```
pct    -> the 'LaTeX comment' rule removed a span that is not a LaTeX comment: '%'
fence  -> an 'inline code' span is part of a fence rather than a span of its
          own: '``' between '\n' and '`'
```

**And the first version of the inline predicate passed with the bug
replanted.** It looked for "```" inside the removed span; the defect produces
a span that contains no fence because it is part of one. The working test is
adjacency. That is the fourth gate today that was vacuous until its defect was
replayed, and the pattern now has its own section in `AUDIT_DISCIPLINE.md`.

### 2026-10-11 (sixth) --- the declared-change gap is real and stays open

Asked whether a full-regeneration test closes the gap the dataset hash leaves
--- a declared corpus change whose artifacts were not rebuilt. It does not.
There is no test that regenerates every consumed artifact and compares.

- `test_the_shipped_envelope_matches_a_fresh_build` covers **one** of the
  **17** artifacts the staleness gate lists, and compares selected keys rather
  than bytes: `cells_refused`, `cells_insufficient_evidence`, the two scheme
  equivalents, `known_limitations`, `validation`, plus a direct pin against
  `strata.calibrated_fit`.
- `test_reharvest_reproduces_committed_dataset_on_every_column` regenerates
  `data/dataset.csv` from the cards, not the artifacts, and **returns early
  when `data/cards` is absent** --- which is every clone, since the card corpus
  is deliberately not redistributed. In the default gate on a checkout it is a
  no-op.

So the composition is: the sha256 pin makes a SILENT corpus change impossible,
and nothing makes an UNREBUILT declared change impossible. 16 of 17 consumed
artifacts have no fresh-build comparison at all. The gap is recorded rather
than implied closed.

### 2026-10-11 (fourth) --- input coverage, swept

The escaped-percent defect is recorded above. This is the sweep it forced, and
two more findings came out of it.

**Bytes in against bytes scanned, for every reducing check.** Measured, pinned
within a band, and the band is tight enough to fail on the defect that
prompted it --- verified by reintroducing the bug and watching the test go red
at 93.0% against a 96-98% band. The first band I wrote (93-98%) let the
reintroduced bug pass, which is the whole failure mode of this document, found
by planting rather than by reasoning.

| check / input | retained |
| --- | --- |
| `audit_paper` body / either variant | 95.9%, 96.3% |
| `check_prose` raw / either variant | 98.3%, 98.7% |
| `check_prose` heading-stripped | 95.4%, 95.8% |
| `audit_traceability` visible / 6 docs | 68.8% - 88.4% |
| `check_docs` text / 44 docs | 91.2% overall, 44.9% worst (`RANKING.md`) |

The low numbers are legitimate and now say why in the test: claim tags are
large and numerous (`RANKING.md` carries 217), and every numeral inside an
inline-code span was read by hand --- shell commands, size-band labels like
`w4a16|<2B`, model names, formula constants, package versions, commit hashes,
one IP address. None is a figure.

**Each stripper carries a planted construct it must keep.** The escaped
percent for `audit_paper` and `check_prose`, a prose figure and a fenced block
for `audit_traceability`, an unclosed `<!--` for `check_docs`. Each probe also
asserts the stripper still removes what it is for, so "keeps everything"
cannot pass either.

**Found in the sweep: `audit_traceability` was removing fenced code blocks by
accident.** It stripped inline code first, so the inline pattern chewed each
fence three backticks at a time and the fenced-block pattern matched zero
bytes on every document. The content was removed, but by backtick pairing
rather than by the rule written for it; a fence whose backticks paired
differently would have left shell commands in the scan or eaten the prose
around them. It cost 18 characters of README, which was luck. Fences are now
stripped first.

**Found in the sweep: the artifact freshness gate compared timestamps only.**
It catches the safe direction --- a dataset whose mtime moves while an artifact
does not, which is the red a byte-identical restore produced earlier today ---
and it cannot see the unsafe one. A corpus edit whose timestamp is preserved
(`touch -r`, an rsync or checkout that keeps times, an edit then a restored
timestamp) leaves every derived artifact looking current against data it was
not built from, which is exactly the C1/C2 failure arriving by a route the
mtime check is blind to. `data/dataset.csv` is now pinned by sha256 with its
line count, and the two reds are distinguishable: the hash says the corpus
moved, the mtime says only that a file was touched. Plant-verified by changing
one `acc_after` value and restoring the timestamp --- the mtime gate passes and
the hash gate fails.

**The appendix figures are measured, not recorded.** The five numbers in that
entry are computed by running the broken stripper and the fixed one over the
current paper and differencing, so they stay true as the paper changes. They
have already moved: writing the entry added escaped percent signs, so the
count is 136 rather than the 132 first observed, and the hidden span 3,359
characters rather than 3,162. The claim is present-tense about the current
file, which is why that is right rather than awkward.

### 2026-10-11 (third) --- the gate's own input, and measured declarations

**3% of the paper was never read by any check in `audit_paper.py`.** The scan
body was built with `re.sub(r"%.*", " ", body)`, and an escaped `\%` is a
percent sign rather than a comment: 132 of them against 7 real comments, so
every line was truncated at its first printed percentage. 3,162 characters
were invisible to the numeral gate, the word gate, the derived-noun check and
the unit gate. Found while deriving which macros count things, because
`\RefuseBelow\%` appeared to be followed by a noun from two lines down.

The recovered 3% contained four hand-typed numerals nobody had seen (`0.9`,
the conformal index's `1-alpha`; and `168`, `90.01`, `91.0`, day-7
before-and-after values in the same appendix item as three figures that were
already exempt), and two figures written as words. One of those was real: "the
other five folds" is `\Nfam` minus one, and the sentence now says "the
remaining folds" rather than carrying an arithmetic derivative of a registered
figure. Every check in that file was correct and every one was running on a
mutilated copy of its input; a gate reading 97% of a document reports the same
"0 problems" as a gate reading all of it. Recorded in `AUDIT_DISCIPLINE.md`.

**The unit-declaration set is now measured.** It began as 26 declarations
against ~314 defined macros, so ~288 were exempt by omission --- the
protection covered the macros somebody thought of, which is the shape the
derived noun vocabulary exists to avoid. The same move applied again: a macro
followed by a counted noun IS a counting macro and must declare its unit, and
the build fails otherwise. 77 counting macros, 79 declarations, 0 undeclared,
1 in `UNITLESS` with a reason (`\AdvMaxB`, whose unit is B for billions of
parameters, printed as a letter). Plant-verified twice: removing
`\TuneOuterFolds`'s declaration reports it, and a brand-new macro used in a
counting position reports it.

Two sentences were reworded rather than have a declaration widened around
them: `\MinCkpt{} gates both the coverage verdict` used "gates" as a verb,
and `\NRhCardsHarvested{} report none` uses "report" as one.

**The cards-to-checkpoints bijection is asserted as a data property, not a
guarantee.** `n_rh_cards_harvested` and `n_quant_checkpoints` both read 102.
A card describes one quantized artifact so the map is onto, but nothing makes
it injective --- two repos could legitimately be two w4a16 quantizations of
one base at different group sizes, and then the counts would diverge while
both macros rendered fine and README's sentence explaining the match went
silently wrong. The test asserts no (base model, scheme) pair is served by two
cards, says in its own body that this is data rather than structure, and tells
a future reader to reword README rather than delete the test. README now says
the counts match on this corpus and why. Plant-verified by adding a second
card for one pair.

**"Base checkpoint" is now forbidden by assertion.** The third term --- neither
the defined "checkpoint" nor the defined "base model" --- survived in the
paper's own family definition until this evening. A negative test now refuses
it in `paper/` and every generated document, excluding third-party card text
under `data/`. Plant-verified by putting it back into the family definition.

### 2026-10-11 --- figures written as words: the sweep and the gate

The central traceability gate matches `\d+`, so every figure the paper wrote
as a word has been invisible to it since the registry existed. Closed in two
halves: a gate that makes the class impossible, then a sweep that cleared
today's instances.

**The gate.** `paper/audit_paper.py` now fails the build on a figure written as
a word. Matching every word form from `zero` to `ninety-nine` plus `hundred`,
`dozen`, `both` and `half` flags about 300 sites in this paper and roughly 290
are rhetorical ("one of two grounds", "both halves", "one level up"), so the
pattern matches the SHAPE of a count instead: a word number, up to two
modifiers, then one of a closed list of things this project counts. That is 63
sites, which is reviewable, and `WORD_EXEMPT` carries a reason for each
rhetorical or definitional one. Plant-verified both ways --- "nineteen
checkpoints drawn from thirteen base models" fires twice, "seventeen cells"
fires, and "one of the two readings, and both halves of the record" stays
silent.

**What the sweep found.** Nine figures were quoting a registry value as a
word, and two of those had macros that already existed and were in use
*elsewhere in the same section*. Eight figures had no key at all and now do:
`strat_widen_cells_gone` / `_new`, `strat_schemes_narrower`,
`calib_rows_nvfp4`, `calib_cells_attempted` / `_survived` / `_row_starved`,
and `strat_cells_in_sample_pop`. Registry: 360 -> 368 scalars.

**Two of the sites were wrong, not merely untraced.**

*"Two of the old widenings disappear, two new ones appear."* The set
difference is **four** and two is the net change: 6 cells widen under the
retracted band, 4 under the shipped one, 2 under both, so 4 go and 2 arrive.
Registered as the set difference with the reason, at both sites.

*"`\SizeWidthsOld{}` of `\NCellsTotal{}` cells raised their scheme's width."*
`n_cells_total` is 17 --- the cell population of the COVERAGE artifact. The
in-sample band fitted 12 cells, and the shipped band fits 7. Both sites were
using a correct number as the denominator of a population it does not
describe, which is the Class B defect the audit section documents, in the body,
twice. The two 12-cell populations are also not the same set: both hold 12 and
5 members differ on each side, so reusing `calib_cells_attempted` would have
repeated the error with a different macro. `strat_cells_in_sample_pop` is
separate for that reason, and both sentences now name their own population.

**A fix that was not a fix.** `strat_widen_cells_both` was registered in an
earlier cycle in response to this same complaint, and `\StratCellsBoth` was
used at one site while a near-duplicate paragraph two pages later still said
"only two cells are the same in both". Registering the key and converting one
of its two sites looked like closing the item.

### 2026-10-11 --- the jackknife+ pooled figure

The paper quoted jackknife+'s NVFP4 coverage (`\JackknifeNvfpCov`, the worst
per-scheme figure in the work) and never its pooled coverage, which was
registered as `jackknife_cov_pct` and had no macro. Reporting the scheme a
rejected construction failed on while omitting how it did overall overstates
the case against it, in the direction that flatters the construction we ship.
`\JackknifeCov` now appears beside it: 88.09% pooled, close to nominal and
above the 1-2*alpha = 80% jackknife+ actually guarantees.

Every jackknife+ mention is in the process-error appendix; none is in the body,
so the asymmetry was inside one paragraph rather than between the body and the
appendix. It was still worth fixing where the claim is made.

### 2026-10-11 --- two gates that could not go red for the right reason

`test_both_paper_variants_pass_audit_paper_cleanly` asserted on
`stdout.splitlines()[-6:]`, so a truncated or interleaved capture failed it
while the audit was clean --- observed once on 2026-10-08, passing in
isolation and in three later runs. `audit_paper.py` now emits one
machine-readable line, `AUDIT_PAPER_RESULT variant=... failures=N
warnings=N`, and the test requires exactly one of them anywhere in the output:
a truncated capture now fails as "no sentinel" rather than as a false audit
failure, and the two conditions are distinguishable.

`src/build_supplement.py` drops itself from the build tree because its regex
patterns quote the author name, and nothing asserted that entry existed.
Removing it ships a file enumerating every identity string being scrubbed,
which is both a leak and a roadmap; disabling the entry and running the scrub
produces 13 hits across 5 labels and exit 1.
`test_the_self_referential_files_are_pinned_in_drop_files` pins the entry for
six such files and, in a second half, shows the scan actually finds an
identifier in each one --- so the pin is load-bearing rather than cargo cult.
Plant-verified: removing the `src/make_pdfs.py` entry names that file in the
failure.

### 2026-10-11 (later) --- the gate's own coverage boundary, and the jackknife reason

**The counted-noun list was a guess, and it was the boundary.** The word gate
only sees figures counting a noun on its list, and the first list was chosen by
hand. It is now derived from the paper's own numeral usage and the build fails
when a derived noun is uncovered. The derivation added `numbers`, `losses`,
`bases`, `plots`, `estimates` and `misses`; `misses` is the one that matters,
since "two misses against two" was a blocking defect's supporting asymmetry.
Three sites fired under the wider vocabulary and all three were rhetorical
("wants one number" twice, "all five quantization recipes hit" where `hit` was
a verb); the singulars of `hit`/`miss`/`loss`/`base` were dropped for that
reason. Plant-verified: "14 notebooks and 6 dashboards" fails and names both,
"11 misses" passes.

**The reason for preferring split conformal is arithmetic, and the paper did
not say so.** Now that the pooled jackknife+ figure is printed --- 88.09%,
above the 1-2*alpha = 80% it guarantees --- the measurement visibly does not
justify rejecting it, and a reader would ask. The pre-registration always had
the right reason and the paper never carried it: a 90% *guarantee* from
jackknife+ needs alpha=0.05, whose conformal index needs 19 leave-one-out
units per scheme, and only 3 of 6 schemes have them (w8a8_int 24, w4a16 23,
fp8_dynamic 22; w8a16 15, fp8 13, nvfp4 5). Verified against the artifact
before writing. The appendix now states that jackknife+ performed acceptably,
that the choice is arithmetic rather than empirical, and that the NVFP4 figure
is its per-scheme worst case rather than the grounds. Registered as
`jackknife_schemes_with_19_units`.

**A fix that exhibited the class it was fixing** is now its own appendix entry.
Converting one site of a figure and registering its key produces every visible
sign of closure, so a fix must enumerate the figure's sites rather than the
site that prompted it. It is the one entry in the record made *after* the
lesson was written down.

**Two gates fired on the prose written for this entry**, which is the
intended behaviour and worth recording as evidence they are not vacuous: the
numeral check flagged alpha and the 19-unit index requirement (both exempted
as construction constants, like the 9 at alpha=0.10), and the word gate
flagged "leave-one-out units" (the method name, now excluded) plus a
meta-count of our own edits, which was removed rather than registered.

**The completeness pin caught a miscount of mine.** I pinned the paragraph
total at 15 on the assumption that the mechanical-verification note was a new
block; it extended the existing one, so the real total is 14. That is what
the pin is for.



