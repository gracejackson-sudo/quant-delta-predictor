# quant_delta_predictor

**Before you compress a model: how much accuracy did other people lose doing the same thing?**
A small command-line tool that answers from published results, and says so when it does not
have enough evidence to answer.

## Try it in about a minute

```bash
git clone https://github.com/gracejackson-sudo/quant-delta-predictor && cd quant-delta-predictor
python3 -m venv .venv && ./.venv/bin/pip install numpy pandas scipy
./.venv/bin/python src/rank.py fp8_dynamic --size 5B    # a case it can speak to
./.venv/bin/python src/rank.py w4a16 --size 1.5B        # a case it says it cannot judge
```

No network, API key or GPU is needed after the install. The first run takes about half a minute.

**What you will see.** The first command returns a historical range of accuracy change for that
scheme, with no warning flags, and still shows the worst loss ever observed, which is worth reading.
The second prints a range too, but marks it `INSUFFICIENT_EVIDENCE`, drops the scheme to Tier C, and
explains why: for 4-bit weights on sub-2B models there are too few independent published checkpoints
to trust a number. That flag is the tool declining to vouch for its own output.

## What it is, and is not

- **It is** a table of historical ranges: per quantization scheme, how much accuracy published
  checkpoints lost, checked against checkpoints it was not built from.
- **It is not** a per-model predictor. Predicting the loss for one specific model did not work:
  most of the differences between models were benchmark measurement noise.
- **Two separate limits, easy to confuse.** (1) *Noise* is why per-model prediction fails.
  (2) *Thin evidence* is why the tool flags a scheme-and-size combination: not enough independent
  published checkpoints, which has nothing to do with noise.
- **Read every range as a floor on risk, not a ceiling.** It is built from one publisher's
  checkpoints, so a badly tuned recipe can do far worse than anything in the record.

## Where to go next

1. [TOOL_SUMMARY.md](TOOL_SUMMARY.md): one page, what it does, what it flags, what it cannot do
2. [FINDINGS.md](FINDINGS.md): the research result, with corrections applied in place
3. [`paper/`](paper/): the write-up

Everything else is in the table below.

---

## All documentation
Feasibility spike: given `(base model, quantization config)`, predict the accuracy delta on
OpenLLM-style benchmarks with a calibrated prediction interval.

**Result: the calibration works (90.1% empirical coverage on unseen checkpoints at a nominal 90%);
the point prediction carries almost no signal beyond the quantization scheme, and the fitted
artifact is a small number table — a calibrated historical baseline, not a predictor.**

| document | what it holds |
|---|---|
| [TOOL_SUMMARY.md](TOOL_SUMMARY.md) | **start here** — one page: what it does, what it refuses, what it cannot do |
| [FINDINGS.md](FINDINGS.md) | the result, with corrections applied in place |
| [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) | per-model prediction has no signal beyond the scheme average |
| [BIAS_CORRECTION.md](BIAS_CORRECTION.md) | the selection bias, measured; why no corrected point estimate |
| [AUDIT_DISCIPLINE.md](AUDIT_DISCIPLINE.md) | the standing audit rule and what it has caught |
| [RESEARCH.md](RESEARCH.md) | prior-art synthesis + pre-registered predictions, written first |
| [ADVERSARIAL_AUDIT.md](ADVERSARIAL_AUDIT.md) | attempts to break the headline number |
| [ACCOUNTING.md](ACCOUNTING.md) | every model, tested or dropped, and why |
| [PROVENANCE.md](PROVENANCE.md) | which parser fixes were informed by test-set rows |
| [RANKING.md](RANKING.md) | the scheme ranking, and why size stratification was mostly rejected |
| [SCOPE.md](SCOPE.md) | the minimal public tool and hours to ship it |

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install numpy pandas scipy
# scikit-learn and pytest are only needed to re-run the research, not the tool
```

## Use the tool

No network, no API key, no GPU. Three packages: numpy, pandas, scipy.

**Run this one first.** It is the command that shows you what the tool is:

```bash
./.venv/bin/python src/rank.py w4a16 --size 1.5B
```

It does not give you a clean answer. It prints the historical range for 4-bit weights, then flags
that cell `INSUFFICIENT_EVIDENCE`: measured coverage there rests on only two checkpoints, so the tool
demotes W4A16 to Tier C and tells you to run your own evaluation. A cell is *refused* outright, with
its interval withheld (`INSUFFICIENT CALIBRATION`), only when measured coverage is poor on enough
independent checkpoints; none currently is.

That is the design. A quantization risk estimate is only worth having if it will tell you when not
to trust it, and the cells it flags are exactly the ones where a confident-sounding answer would do
the most damage.

Once you have seen it flag a cell, the rest:

```bash
./.venv/bin/python src/rank.py                      # rank every scheme
./.venv/bin/python src/rank.py fp8                  # one scheme it will answer for
./.venv/bin/python src/rank.py --form-fields        # what a contributed result needs
```

Each scheme comes back with a 90% interval, the worst loss ever observed for it, the
share of evaluations that lost more than 3pp, how many checkpoints and families back
it, and a tier. Cells resting on fewer than three distinct checkpoints, or whose coverage
estimate is too uncertain to judge, are marked `INSUFFICIENT_EVIDENCE`; `RANKING.md` lists them.

## Reproduce the research

The research scripts need two more packages than the tool does:

```bash
./.venv/bin/pip install scikit-learn pytest
```

```bash
./.venv/bin/python src/harvest.py          # rebuild dataset.csv from cards (needs the cards; see data/README.md)
./.venv/bin/python src/run_final.py        # all three split regimes and calibration variants
./.venv/bin/python src/demo_holdout.py     # train on 2 families, predict unseen models
./.venv/bin/python src/real_use_case.py    # prospective test on never-seen families (needs network)
./.venv/bin/python src/diagnose.py         # which failure mode this is
./.venv/bin/python src/validate_strata.py  # does size stratification help? mostly not - see RANKING.md
./.venv/bin/python src/build_envelope.py && ./.venv/bin/python src/cli.py --list
./.venv/bin/python -m pytest tests -q      # the test suite
```

The audits, and the full exclusion census:

```bash
./.venv/bin/python src/audit.py && ./.venv/bin/python src/adversarial_audit.py && ./.venv/bin/python src/census.py
```

Independent re-verification — stdlib only, no project imports. Regenerates the headline coverage
from the raw cards and diffs it against the pipeline row by row:

```bash
python3 verify/independent_check.py
```

## Data

850 rows of `(model, quant_config, benchmark, accuracy_before, accuracy_after)` scraped from 102
[RedHatAI](https://huggingface.co/RedHatAI) model cards, spanning 38 base checkpoints, 8 model
families, 6 quantization schemes and 16 benchmarks. Every three-column row is verified against
the card's own printed Recovery percentage; internally inconsistent rows are rejected rather than
guessed at.

## What it does NOT do

- It does not use the model family or size — those features made out-of-family accuracy *worse*.
- Its intervals never exclude zero, so it cannot tell you a config will definitely hurt.
- It is trained only on checkpoints Red Hat chose to publish, so it underpredicts damage from a
  badly-tuned recipe.
- Sub-2B, MoE and reasoning-distilled models fall outside the validated envelope.

## Standing rule on claims

Every quantitative statement in `RANKING.md` must trace to a value computed by the audit scripts.
This is enforced mechanically, not by discipline:

- `RANKING.md` is **generated** by `src/gen_ranking_doc.py` from computed values. Do not hand-edit
  figures in it.
- Each figure carries a `<!-- claim: key = value -->` tag (invisible when rendered).
- `src/verify_claims.py` recomputes every tagged value and fails on any mismatch or any tag not in
  its registry.
- `tests/test_all.py::test_ranking_doc_claims_all_verify` runs that check in CI.
- `tests/test_all.py::test_no_banned_prose_in_user_facing_output` blocks phrases that were wrong
  before ("safe to adopt", "ignores your model", "12 numbers", "26 of 29", "80%, not 90%") from
  reappearing in printed output.

```bash
./.venv/bin/python src/cell_coverage.py     # measure per-cell coverage
./.venv/bin/python src/gen_ranking_doc.py   # regenerate RANKING.md
./.venv/bin/python src/verify_claims.py     # re-verify every number
```

Each `(scheme, size band)` cell gets one of three verdicts. A cell with adequately supported
coverage is *trusted*. A cell whose coverage is measurably poor is *refused*: the tool prints
`INSUFFICIENT CALIBRATION` and **no interval**. A cell resting on fewer than three checkpoints, or
whose checkpoint-bootstrap interval straddles the 85% line, is *insufficient evidence*: the interval
is still printed, but the scheme is demoted to Tier C with a note that the cell cannot be judged.
At present no cell is refused; the cells in the third state are listed in `RANKING.md`.

## Data and licence

This repository is MIT licensed (see `LICENSE`); data provenance is recorded in `NOTICE`. The raw Hugging Face model
cards the dataset was extracted from are **not redistributed** here — they are
RedHatAI's content under several different licences. `data/dataset.csv` (the
extracted numbers) is included and is all the tool needs. See
[data/README.md](data/README.md) for what that costs you and how to fetch the
cards yourself.

Source data: evaluation results published by RedHatAI on Hugging Face —
<https://huggingface.co/RedHatAI>.

## References

- Frantar, Ashkboos, Hoefler & Alistarh. *GPTQ: Accurate Post-Training
  Quantization for Generative Pre-trained Transformers.*
  [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) — the method behind
  every w4a16 checkpoint in this dataset.
- Xiao et al. *SmoothQuant.*
  [arXiv:2211.10438](https://arxiv.org/abs/2211.10438) — the activation
  smoothing used for the w8a8 checkpoints.
- Zeng & Papailiopoulos. *You Don't Need to Run Every Eval* (BenchPress).
  [arXiv:2606.24020](https://arxiv.org/abs/2606.24020) — published the core
  predict-then-conformal mechanism first; see `NEGATIVE_RESULT.md` §5.
- Barber, Candès, Ramdas & Tibshirani. *Conformal Prediction Beyond
  Exchangeability.* Ann. Statist. 51(2), 2023 — why the coverage guarantee is
  conditional on the published population, not on your own recipe.
