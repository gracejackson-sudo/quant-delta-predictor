# Pre-registration: nested cross-validated tuning of the ridge and gradient-boosting baselines

**Written and committed before the tuning was run.** Round item 4. Nothing
below was changed after seeing a result; the commit that adds this file
contains no result.

## Why this exists

A reviewer noted, correctly, that an untuned model losing to the global mean is
uninformative about predictability. The shipped comparison uses ridge at a
fixed `alpha=3.0` and histogram gradient boosting at fixed hyperparameters,
neither cross-validated. If a tuned model still loses, the negative result is
much stronger. If it wins, the negative result weakens or reverses and we
report that.

The obvious wrong way to do this is to tune against the leave-one-family-out
number the paper reports. That is tuning on the test folds, and a reviewer who
already recommended reject-and-resubmit would find it immediately.

## Protocol

### Outer loop: unchanged from the shipped construction

Leave-one-family-out over the \(6\) training families (gemma-2, granite,
llama-3, mistral, qwen-2.5, qwen-3), giving 6 outer folds. For each fold:

* **Outer test set** = every row of the held-out family. Used exactly once, to
  compute that fold's MAE. Never used for fitting, for hyperparameter
  selection, or for any decision.
* **Outer training set** = every row of the other 5 families.

Row-weighted mean MAE across the 6 outer folds is the reported figure, which
is the same metric and the same folds as the existing `pred_mae::*` keys, so
the tuned and untuned numbers are comparable.

### Inner loop: entirely inside the outer training set

Within each outer fold's training set, and using only those 5 families:

* **Inner split** = leave-one-family-out again, over the 5 training families,
  giving 5 inner folds.
* For each candidate hyperparameter setting, fit on 4 inner-training families
  and score on the 1 inner-validation family; take the row-weighted mean MAE
  across the 5 inner folds.
* **Selection** = the setting with the lowest inner MAE. Ties broken by the
  first setting in grid order, which is fixed in advance.
* The selected setting is then refit on all 5 outer-training families and
  evaluated once on the outer test family.

Inner splitting is by **family**, not by row or by checkpoint, deliberately.
The outer estimand is transfer to an unseen family, so the hyperparameters
should be chosen for that kind of generalisation rather than for within-family
interpolation. Splitting inner folds by row would select hyperparameters
tuned to a shift the deployment never sees.

### The confirmation a reviewer will check

**No fold's test data informs its own hyperparameters.** For outer fold \(k\)
with held-out family \(F_k\): every inner fit, every inner score and the
selection argmin use only rows from the 5 families that are not \(F_k\). No
row of \(F_k\) enters the design matrix, the target vector, or the selection
criterion at any point before the single final evaluation. Hyperparameters are
therefore selected **6 separate times**, once per outer fold, and the 6
selections may differ. We report them, because a selection that changes
sharply across folds is itself evidence about how little signal there is.

A single global hyperparameter choice would be the leak: picking one setting by
looking at all 6 outer folds and then reporting those folds. We do not do that,
and the per-fold selections are reported so it is checkable.

## Grids, fixed in advance

Ridge, on the standardised 38-feature design already used (`model.featurize`):

```
alpha in {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000}
```

Histogram gradient boosting:

```
learning_rate     in {0.01, 0.05, 0.1}
max_depth         in {2, 3, None}
min_samples_leaf  in {5, 15, 30}
l2_regularization in {0.0, 1.0, 10.0}
max_iter           = 300          (fixed)
random_state       = 0            (fixed)
early_stopping     = False        (fixed, to keep every fit deterministic)
```

81 gradient-boosting settings and 11 ridge settings, each evaluated on 5 inner
folds within each of 6 outer folds.

**Grid-edge rule, registered in advance.** If a selected value lands on a grid
boundary we report that as a limitation of this run. We do **not** extend the
grid and re-run. Extending a grid after seeing which edge was hit is the
retuning this pre-registration exists to prevent.

## One variable at a time

Same 819 rows, same 6 outer folds, same MAE metric, same feature matrix, same
`random_state=0`. The only thing that changes is whether hyperparameters are
fixed constants or selected by inner cross-validation.

The untuned `pred_mae::ridge` and `pred_mae::grad_boost` keys stay computed and
reported. The tuned figures are added alongside as new keys, so both paths run
and the old numbers remain reproducible.

## Where the thinness tripwire currently sits

`tests/test_all.py::test_predictor_ordering_matches_the_papers_negative_claim`
fails if either losing margin falls below one third of the per-scheme mean's
own advantage over the global mean. At the time of writing:

| quantity | value |
|---|---|
| per-scheme mean's advantage (`gain`) | 0.0321pp |
| tripwire threshold (`gain/3`) | 0.0107pp |
| ridge margin above the global mean | +0.0229pp (headroom +0.0122pp) |
| gradient boosting margin | +0.0185pp (headroom +0.0078pp) |

Tuning moves exactly these two numbers, so both are reported against the
threshold and not merely as a pass/fail on the ordering.

## Pre-registered responses to all four outcomes

Let `gain` be the per-scheme mean's advantage over the global mean, and let a
model's *margin* be its MAE minus the global mean's MAE (positive = it loses).

**Outcome 1 — ordering holds, margins widen.** Both tuned margins stay positive
and above `gain/3`. The negative result is strengthened: the baselines lose even
when tuned. We report the tuned figures alongside the untuned ones, say that
tuning does not rescue them, and the abstract's claim stands as written.

**Outcome 2 — ordering holds, margins thin.** Both tuned margins stay positive
but at least one falls below `gain/3`. The ordering claim survives but
"worse than guessing the average" becomes a thin description. The tripwire test
fires, and per the framing already agreed we soften the abstract and §5 to
"neither tuned ridge nor tuned gradient boosting beats guessing the average"
and state the margin explicitly rather than leaving the reader to assume it is
comfortable. We do not re-pin the tripwire to make the test pass.

**Outcome 3 — a tuned model beats the global mean.** Its margin goes
non-positive. The abstract's central negative claim, that both are worse than
guessing the average, is **false as written** and is rewritten, not re-pinned.
The per-scheme mean may still be the best predictor, in which case the shipped
artifact's justification survives and the claim narrows from "card features
carry no signal" to "card features carry signal only once the model class is
tuned, and still less than a six-cell lookup". We report which fold or folds
produced the win and the selected hyperparameters, because a win driven by one
fold is a different finding from a uniform one.

**Outcome 4 — a tuned model beats the per-scheme mean.** The headline negative
result **reverses**. We report it as the paper's primary finding changing: per-
model prediction from card features does carry usable signal once the model
class is tuned, the shipped six-number artifact is no longer the best available
predictor, and §5, the abstract, contribution 1 and `NEGATIVE_RESULT.md` all
need rewriting around the new result. We do not suppress it, do not re-run with
a different grid, and do not look for a reason to discard it. We also report
the checkpoint-bootstrap CI on the difference, because a win inside noise is
still a win that has to be reported honestly as being inside noise.

## The framing that survives all four

Independently of the ordering: every card-feature predictor tested lands within
about 0.05pp of the global mean, against an evaluation-noise floor of about
0.51pp. If tuning scrambles the ranking but leaves that true, the paper says so
in those terms, because that statement does not depend on which predictor came
last.
