# quant_delta_predictor

Feasibility spike: given `(base model, quantization config)`, predict the accuracy delta on
OpenLLM-style benchmarks with a calibrated prediction interval.

**Result: the calibration works (90.1% empirical coverage on unseen checkpoints at a nominal 90%);
the point prediction carries almost no signal beyond the quantization scheme, and the fitted
artifact is a small number table — a calibrated historical baseline, not a predictor.**

| document | what it holds |
|---|---|
| [FINDINGS.md](FINDINGS.md) | the result, with corrections applied in place |
| [RESEARCH.md](RESEARCH.md) | prior-art synthesis + pre-registered predictions, written first |
| [ADVERSARIAL_AUDIT.md](ADVERSARIAL_AUDIT.md) | attempts to break the headline number |
| [ACCOUNTING.md](ACCOUNTING.md) | every model, tested or dropped, and why |
| [PROVENANCE.md](PROVENANCE.md) | which parser fixes were informed by test-set rows |
| [RANKING.md](RANKING.md) | the scheme ranking, and why size stratification was mostly rejected |
| [SCOPE.md](SCOPE.md) | the minimal public tool and hours to ship it |

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install numpy pandas scikit-learn scipy pytest
```

## Use the tool

This is the entry point. No network, no API key; numpy and pandas are enough.

```bash
./.venv/bin/python src/rank.py                      # rank every scheme
./.venv/bin/python src/rank.py w4a16 --size 1.5B    # one cell -- this one it refuses
./.venv/bin/python src/rank.py --form-fields        # what a contributed result needs
```

The second command is worth running first: `w4a16|<2B` is a cell where measured coverage
is 68.8% against the 90% claimed, so the tool prints `INSUFFICIENT CALIBRATION` and explains
itself rather than returning a number.

## Reproduce the research

Rebuild the dataset from the cached model cards (no network needed):

```bash
./.venv/bin/python src/harvest.py
```

Final evaluation across all three split regimes and calibration variants:

```bash
./.venv/bin/python src/run_final.py
```

The held-out demo — train on 2 families, predict real unseen models:

```bash
./.venv/bin/python src/demo_holdout.py
```

The prospective test — freeze the predictor, then download model cards for families never used
in development (needs network):

```bash
./.venv/bin/python src/real_use_case.py
```

Diagnostics, the three audits, the adversarial audit, and the full exclusion census:

```bash
./.venv/bin/python src/diagnose.py && ./.venv/bin/python src/audit.py && ./.venv/bin/python src/adversarial_audit.py && ./.venv/bin/python src/census.py
```

Fully independent re-verification — stdlib only, no project imports, regenerates the headline
coverage from the raw cards and diffs it against the pipeline row by row:

```bash
python3 verify/independent_check.py
```

Does size stratification help? (it mostly does not - see RANKING.md):

```bash
./.venv/bin/python src/validate_strata.py
```

The shippable artifact and the single-scheme lookup:

```bash
./.venv/bin/python src/build_envelope.py && ./.venv/bin/python src/cli.py --list
```

Tests:

```bash
./.venv/bin/python -m pytest tests -q
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

Where a `(scheme, size band)` cell has measured coverage below threshold, the tool prints
`INSUFFICIENT CALIBRATION` and **no interval**, rather than a number that would look as confident
as a well-calibrated one.

## Data and licence

This repository is MIT licensed (see `LICENSE`). The raw Hugging Face model
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
