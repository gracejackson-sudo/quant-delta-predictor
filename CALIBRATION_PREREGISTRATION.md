# Pre-registration: replacing the in-sample band with a calibrated one

Round item 1. Written and fixed **before** anything is run. Nothing in the
"Decision bands" or "Pre-registered responses" sections may be changed after
seeing a result; if it turns out to be the wrong protocol, we report that it
was the wrong protocol and what it said.

Corpus: `data/dataset.csv` at `4224438`, 856 raw rows / 823 modelling rows,
38 checkpoints, 6 families. Worked from the verified card snapshot, not a
re-fetch.

## 1. What is actually wrong today

`src/rank.py:195` calls `ConservativeStratified().fit(d)`. That path computes,
for each cell, a centre (the cell's mean delta) and a half-width (the
conformal quantile of `|delta - mean|`) **from the same rows**. The index is
`ceil((n+1)(1-alpha))` with `alpha = 0.10`, which is the split-conformal
formula, but applied in-sample it carries no split-conformal guarantee. It is
a descriptive spread, presented with the arithmetic of a coverage guarantee.

`src/strata.py` already contains the correct path, `fit_calibrated(dtr,
dcal)`, whose own docstring says:

> Fitting both on the same rows makes the widths in-sample and the resulting
> coverage optimistic -- measured at 94% for NVFP4 that way versus 74% when
> calibration is genuinely held out. Anything quoted to a user must come from
> this path.

The shipped tool does not use that path. So this is not a missing capability.
The code states the requirement, quantifies the error at 20 points on the
worst scheme, and then does not meet it. That is the defect, and it is worse
than "the band is in-sample" because the repository already knew.

## 2. The binding constraint, stated before running because it is arithmetic

A conformal quantile at level `1-alpha` over `n` calibration units is finite
only when `ceil((n+1)(1-alpha)) <= n`:

| alpha | nominal | minimum units |
| --- | --- | --- |
| 0.10 | 90% | **9** |
| 0.05 | 95% | **19** |

That is where the `len(g) >= 9` guard in `fit_calibrated` comes from.

Checkpoints per scheme in the current corpus:

| scheme | rows | checkpoints |
| --- | --- | --- |
| w8a8_int | 195 | 24 |
| w4a16 | 194 | 23 |
| fp8_dynamic | 200 | 22 |
| w8a16 | 86 | 15 |
| fp8 | 80 | 13 |
| **nvfp4** | **68** | **5** |

At the checkpoint unit and `alpha = 0.10`, five of six schemes clear the
nine-unit floor using *all* their checkpoints as calibration, and any split
leaves fewer. **NVFP4 cannot clear it at all**, with any construction, because
no construction manufactures independent units. NVFP4 is also the scheme with
the widest band, the highest severe-loss rate (14.7%) and the largest
in-sample optimism (94% vs 74%). The scheme that most needs honest calibration
is the one that can least support it.

This is knowable now, so the response to it is fixed now, in section 6, rather
than chosen after seeing which option flatters NVFP4.

## 3. Construction choice

Two candidates, both run, one pre-designated primary.

**Primary: split conformal, checkpoint-blocked.** Centres from the fit
partition, half-widths from the conformal quantile of residuals on a disjoint
calibration partition, residuals taken about the centre
`predict_interval()` will actually use. Mechanism already implemented as
`strata.ConservativeStratified.fit_calibrated`; this round makes the shipped
path use it. Guarantee: marginal `1-alpha` coverage under exchangeability of
calibration and test units.

**Secondary: jackknife+, leave-one-checkpoint-out.** For each checkpoint, a
centre refit without it; the interval from the quantiles of
`{centre_{-i} +/- |residual_i|}`. No split, so every row informs the centre.
Guarantee: `1-2*alpha`, i.e. **80% at alpha = 0.10** -- not 90%.

Primary is split conformal, and the reason is fixed in advance and is not
about which will score better: the paper wants to claim a 90% interval, split
conformal's guarantee is `1-alpha` and jackknife+'s is `1-2*alpha`. Reaching a
90% guarantee with jackknife+ needs `alpha = 0.05`, which needs 19 units per
scheme, which only w8a8_int, w4a16 and fp8_dynamic have. So jackknife+ at a
90% *guarantee* is arithmetically unavailable for half the schemes and is not
a candidate for the shipped claim. It is run at `alpha = 0.10` for the
empirical comparison, and its 80% guarantee is reported as 80%.

`alpha` stays 0.10 throughout. We do not move it to make either construction
look better.

Both constructions are added alongside the existing code. `fit()` stays
callable and the in-sample figures stay reproducible, because the paper has to
be able to quote what the old path gave.

## 4. The leave-one-out unit

**The checkpoint.** Not the row.

Rows within one checkpoint share the model, the quantization recipe and one
harness invocation. Section 6 of the paper already argues that the coverage
*indicator* is not clustered by checkpoint (ICC -0.006, design effect 1.000)
-- but that is a statement about the binary inside/outside flag, and it is
true because the interval is wide relative to within-run spread. The
*residuals* that calibration quantiles are computed from are a different
quantity and are correlated within a run. Conformal exchangeability is an
assumption about the calibration and test units, so the unit has to be the
thing that is plausibly exchangeable, which is the run and not the row.

Taking the row as the unit would be the self-serving choice: it multiplies the
apparent calibration sample by about 22x (823 rows, 38 checkpoints) and would
let NVFP4 clear the nine-unit floor on 68 rows from 5 runs. We are not doing
that, and we are saying so before seeing what it would have bought.

The family (6 units) is more conservative still and is arithmetically
impossible for a per-scheme 90% quantile. Family-level leave-one-out coverage
is reported as a sensitivity on the pooled figure only.

## 5. What is held fixed

One variable at a time. Everything below is frozen before the run:

- Corpus: `data/dataset.csv` at `4224438`. No re-harvest, no row changes.
- `alpha = 0.10`.
- Cell definition, size bands, and the `min_rows = 20` / `min_ckpt = 3`
  stratum gates, unchanged.
- The centre is the cell mean, unchanged. Only the half-width's provenance
  changes.
- Partition rule for split conformal: checkpoints sorted by `base_model`,
  assigned to calibration by a single fixed seed (`numpy` default_rng(0)),
  calibration fraction 1/3. **One partition. No reshuffling.** If the first
  partition produces an unusable calibration set we report that, we do not
  draw another.
- A repeated-partition sensitivity (20 partitions, seeds 0-19) is run *once*,
  after the primary result is recorded, and is reported as a spread. It cannot
  change which partition's numbers are the headline.
- `MIN_CELL_CHECKPOINTS = 3` and the three-state verdict rule are untouched by
  this item.

## 6. Decision bands

Fixed now. The primary endpoint is **pooled two-sided empirical coverage of
the calibrated band on held-out checkpoints, against the nominal 90%**.

The bands are derived rather than chosen. At 823 rows with a design effect of
about 1.0, the standard error on a coverage estimate near 90% is
`sqrt(0.9 * 0.1 / 823) = 1.05pp`. So:

| outcome | pooled coverage | reasoning |
| --- | --- | --- |
| **SUCCESS** | **>= 87.9%** | within 2 SE of nominal; indistinguishable from 90% at this sample size |
| **AMBIGUOUS** | **85.8% to 87.9%** | detectably below nominal, but inside the range a width adjustment could plausibly explain |
| **FAILURE** | **< 85.8%** | more than 4 SE below nominal; the construction does not deliver its stated level |

Over-coverage is not a failure. Pooled coverage more than 2 SE above nominal
(**> 92.1%**) is recorded as over-coverage and its width cost reported,
because an interval that over-covers is paying for it in usefulness.

Three secondary endpoints, with their own pre-registered readings:

**(a) Per-scheme coverage.** For each scheme clearing the nine-unit floor,
report coverage. A scheme below **80.0%** against a nominal 90% is a
scheme-level FAILURE and is reported as one even if the pooled figure
succeeds. 80.0% is the same threshold the existing cell verdict rule uses for
"coverage measurably poor", via the 85% bootstrap-straddle test, one
conventional step down; it is not a new invention for this round.

**(b) Width.** Report every cell's half-width before and after. No threshold
on growth: wider is the expected and honest direction. But a half-width that
exceeds the full observed range of that cell's deltas makes the interval
vacuous, and that is a FAILURE for that cell regardless of its coverage.

**(c) Cell survival.** Report how many of the 12 currently-accepted stratum
cells still have >= 9 calibration units after the split. Losing cells is an
expected outcome and is **not** a failure: `fit_calibrated` drops a stratum
cell that cannot be calibrated, which is the correct behaviour. If fewer than
8 of 12 survive, we report the tool's coverage as scheme-level only and say so
in the paper rather than shipping strata we cannot calibrate.

**NVFP4, decided now.** It has 5 checkpoints and cannot support a
checkpoint-level per-scheme 90% quantile. Its interval backs off to the pooled
calibrated quantile, the backoff is flagged in the tool's output and counted
in the registry, and its coverage is reported separately and excluded from
"per-scheme coverage" in (a) because it has no per-scheme calibration. We do
not lower the unit to the row to make NVFP4 work, and we do not quietly keep
its in-sample width -- which is what `fit_calibrated` does today for groups
under 9, and which is itself a defect this round should expose rather than
inherit.

## 7. Pre-registered responses to every outcome

**Outcome 1 -- split conformal SUCCEEDS.** It ships. The paper reports
calibrated coverage as the headline, states the construction and the unit,
reports the width cost against the old in-sample widths, and retracts the
in-sample band explicitly rather than quietly replacing it. Jackknife+ is
reported alongside as a secondary with its 80% guarantee stated.

**Outcome 2 -- split conformal is AMBIGUOUS.** It still ships, because a band
with a real guarantee and coverage a little under nominal is strictly better
than a band with no guarantee. The paper reports the shortfall as a measured
shortfall, not as nominal. We do not adjust widths upward to reach 90%: that
is post-hoc tuning of the exact kind Section 5 already concedes once, and the
whole value of this item is that the width comes from held-out residuals.

**Outcome 3 -- split conformal FAILS and jackknife+ clears the SUCCESS band.**
Jackknife+ ships, and the paper says plainly that the choice was made by this
rule, fixed in advance, and that its guarantee is 80% while its measured
coverage is higher. This is the one case where the secondary becomes primary,
and the condition is stated here so it cannot be reached by preference.

**Outcome 4 -- both FAIL.** The honest result, and the one that costs most. It
means the shipped intervals cannot be given a guarantee at this sample size
with this cell structure. We do not ship a calibrated band we know
under-covers, and we do not keep quoting the in-sample band as if it had a
guarantee. The tool reports the band as a descriptive spread with that word in
the output, the paper's contribution 2 is rewritten to claim a descriptive
envelope rather than a calibrated interval, and the abstract loses the word
"calibrated". That is a significant retraction and we take it if that is what
the numbers say.

**Outcome 5 -- the comparison cannot run.** If the split leaves too few
calibration units to produce finite widths on a majority of schemes, that is
a result and not a failed experiment: it says the corpus is too small for
per-scheme calibration at the checkpoint unit. We report the arithmetic,
report pooled-only calibration as the fallback, and do not rescue it by moving
to the row unit or by lowering alpha.

## 8. What we will not do

- Not change the unit after seeing coverage.
- Not reshuffle the partition after seeing coverage.
- Not move `alpha`.
- Not adjust widths by any factor chosen to hit 90%.
- Not drop NVFP4 from the reported schemes to improve the pooled figure.
- Not re-harvest, re-fetch, or change a single row of the corpus.
- Not quote jackknife+'s guarantee as `1-alpha`.
