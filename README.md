# quant_delta_predictor

**Before you compress a model: how much accuracy did other people lose doing the same thing?**
A small command-line tool that answers from published results.

## What it measured

The shipped intervals were prospectively validated on 131<!-- claim: prosp_n = 131.0000 --> rows from 19<!-- claim: prosp_clus_n = 19.0000 --> unseen quantized checkpoints
(Gemma-3, DeepSeek-R1-Distill, SmolLM, SmolLM3, NVIDIA-Nemotron-Nano) — none of which were used to
build the tool. **Two-sided empirical coverage at a nominal 90% is 90.8<!-- claim: prosp_cov_pct = 90.8397 -->% (119<!-- claim: prosp_inside = 119.0000 -->/131<!-- claim: prosp_n = 131.0000 -->, 95% Clopper–Pearson
CI [84.5<!-- claim: prosp_ci_lo = 84.5455 -->, 95.2<!-- claim: prosp_ci_hi = 95.1767 -->]).** The whole ranking chain — every mean, half-width, worst, severe rate, per-scheme
coverage, cluster bootstrap, tier and flag — is independently re-verified by a stdlib-only
reimplementation (`verify/independent_rank.py`) that shares no code with the pipeline; zero field-level
disagreements on the ranking, statistical equivalence within 1pp on the cluster-bootstrap CI, exact
agreement on the noise-floor pool behind the abstract's variance claim.

The point prediction carries almost no signal beyond the quantization scheme; the fitted artifact is a
per-scheme envelope with calibrated coverage and cell-level refusal flags, not a per-model predictor.
A separate GPU adversarial arm on deliberately-bad quantization configs — outside the training corpus —
measured losses down to **−39.5<!-- claim: adv_worst_delta = -39.5000 -->pp** (RedHatAI publishes only recipes that worked, so the corpus
understates the left tail; the adversarial arm is the direct measure of how bad it can get).

## Refusal is a feature

A quantization risk estimate is only worth having if it will tell you when not to trust it. When a
`(scheme, size)` cell rests on too few independent published checkpoints, the tool prints the
historical range, marks it `INSUFFICIENT_EVIDENCE`, drops the scheme to Tier C, and tells you to run
your own evaluation. A cell is *refused* outright, with its interval withheld
(`INSUFFICIENT CALIBRATION`), only when measured coverage is poor on enough independent checkpoints;
none currently is. Currently 6<!-- claim: n_cells_insufficient_evidence = 6.0000 --> of 17<!-- claim: n_cells_total = 17.0000 --> cells are at insufficient evidence and 0<!-- claim: n_cells_refused = 0.0000 --> are refused; the cells
in each state are listed in `RANKING.md`.

## Try it in about a minute

```bash
git clone https://github.com/gracejackson-sudo/quant-delta-predictor && cd quant-delta-predictor
python3 -m venv .venv && ./.venv/bin/pip install numpy pandas scipy
./.venv/bin/python src/rank.py fp8_dynamic --size 5B    # a case it can speak to
./.venv/bin/python src/rank.py w4a16 --size 1.5B        # a case it says it cannot judge
```

No network, API key or GPU is needed after the install. Runs after the install take a few seconds; the
very first one can take longer while numpy, pandas and scipy load for the first time.

**What you will see.** The first command returns a historical range of accuracy change for that
scheme, the worst loss ever observed, the share of evaluations that lost more than 3pp, and how many
checkpoints and families back it — with no warning flags. The second command prints a range too, but
marks it `INSUFFICIENT_EVIDENCE`, drops the scheme to Tier C, and explains why: for 4-bit weights on
sub-2B models there are too few independent published checkpoints to trust a number.

## What it is, and is not

- **It is** a table of historical ranges: per quantization scheme, how much accuracy published
  checkpoints lost, checked against checkpoints it was not built from.
- **It is not** a per-model predictor. Predicting the loss for one specific model did not work:
  most of the differences between models were benchmark measurement noise.
- **Read every range as a floor on risk, not a ceiling.** It is built from one publisher's
  checkpoints, so a badly tuned recipe can do far worse than anything in the record.

## Where to go next

1. [`paper/main.tex`](paper/main.tex) / [`paper/neurips_main.tex`](paper/neurips_main.tex): the current write-up. **This is the source of truth for every figure** — it is regenerated from the claims registry (`src/verify_claims.py`) on every build and audited by `paper/audit_paper.py`.
2. [TOOL_SUMMARY.md](TOOL_SUMMARY.md): one page, what it does, what it flags, what it cannot do
3. [FINDINGS.md](FINDINGS.md): the original 2026-09-21 narrative; kept for the record, but its figures are frozen at that date and sit outside every gate (see AUDIT_DISCIPLINE.md). Where it differs from the paper, the paper is correct.

Everything else is in the table below.

---

## All documentation

| document | what it holds |
|---|---|
| [TOOL_SUMMARY.md](TOOL_SUMMARY.md) | one page: what it does, what it flags, what it cannot do (the best next read after the first screen) |
| [FINDINGS.md](FINDINGS.md) | the original 2026-09-21 narrative (historical; outside the gate — see the preamble in that file) |
| [NEGATIVE_RESULT.md](NEGATIVE_RESULT.md) | per-model prediction has no signal beyond the scheme average |
| [BIAS_CORRECTION.md](BIAS_CORRECTION.md) | the selection bias: what is identified, what is not, and why there is no corrected point estimate |
| [AUDIT_DISCIPLINE.md](AUDIT_DISCIPLINE.md) | the standing audit rule and what it has caught |
| [EXTERNAL_FEEDBACK.md](EXTERNAL_FEEDBACK.md) | six external reviews (Reddit maintainer, a BenchPress author, an ML engineer voice call, a technical collaborator voice call, an ML engineer form submission, a Tong et al. author), point by point, with what was done |
| [ADVERSARIAL_AUDIT.md](ADVERSARIAL_AUDIT.md) | attempts to break the headline number, including the −39.5<!-- claim: adv_worst_delta = -39.5000 -->pp GPU adversarial worst case |
| [RESEARCH.md](RESEARCH.md) | prior-art synthesis + pre-registered predictions, written first |
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

Rank a specific scheme at a specific size:

```bash
./.venv/bin/python src/rank.py fp8_dynamic --size 5B
```

The tool returns a 90% interval, the worst loss ever observed, the share of evaluations that lost
more than 3pp, how many checkpoints and families back it, and a tier.

Now see it flag a cell whose coverage rests on too few checkpoints:

```bash
./.venv/bin/python src/rank.py w4a16 --size 1.5B
```

It prints the historical range for 4-bit weights, then flags that cell `INSUFFICIENT_EVIDENCE`:
measured coverage there rests on only two checkpoints, so the tool demotes W4A16 to Tier C and tells
you to run your own evaluation. A cell is *refused* outright, with its interval withheld
(`INSUFFICIENT CALIBRATION`), only when measured coverage is poor on enough independent checkpoints;
none currently is. That is the design. A quantization risk estimate is only worth having if it will
tell you when not to trust it, and the cells it flags are exactly the ones where a confident-sounding
answer would do the most damage.

The rest:

```bash
./.venv/bin/python src/rank.py                      # rank every scheme
./.venv/bin/python src/rank.py fp8                  # one scheme it will answer for
./.venv/bin/python src/rank.py --form-fields        # what a contributed result needs
```

Each `(scheme, size band)` cell gets one of three verdicts. A cell with adequately supported coverage
is *trusted*. A cell whose coverage is measurably poor is *refused*: the tool prints
`INSUFFICIENT CALIBRATION` and **no interval**. A cell resting on fewer than three checkpoints, or
whose checkpoint-bootstrap interval straddles the 85% line, is *insufficient evidence*: the interval
is still printed, but the scheme is demoted to Tier C with a note that the cell cannot be judged. At
present no cell is refused; the cells in the third state are listed in `RANKING.md`.

## Reproduce the research

The research scripts need two more packages than the tool does:

```bash
./.venv/bin/pip install scikit-learn pytest
```

```bash
./.venv/bin/python src/harvest.py          # rebuild dataset.csv from cards (needs the cards; see data/README.md)
./.venv/bin/python src/run_final.py        # all three split regimes and calibration variants
./.venv/bin/python src/demo_holdout.py     # train on 2 families, predict unseen models
./.venv/bin/python src/real_use_case.py    # prospective test on unseen checkpoints (needs network)
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

850<!-- claim: n_raw_dataset_rows = 850.0000 --> rows of `(model, quant_config, benchmark, accuracy_before, accuracy_after)` scraped from 102<!-- claim: n_rh_cards_harvested = 102.0000 -->
[RedHatAI](https://huggingface.co/RedHatAI) model cards, spanning 38<!-- claim: n_checkpoints = 38.0000 --> base checkpoints, 6<!-- claim: n_families = 6.0000 --> model
families (Definition B, which collapses Llama-3.1/3.2/3.3 into one Llama-3 family on the mechanistic-shared-pretraining-base criterion; the earlier 8-family count read each generation as separate), 6<!-- claim: n_schemes = 6.0000 --> quantization schemes and 21<!-- claim: n_benchmarks = 21.0000 --> benchmarks (GPQA is split across six<!-- claim: n_gpqa_labels = 6.0000 --> card-verified protocol labels — gpqa_main, gpqa_main_norm, gpqa_main_cot_5shot, gpqa_diamond, gpqa_diamond_cot_5shot, gpqa_ambiguous_46 — because the raw RedHatAI cards report as many as five different GPQA protocols and two Llama-3.3 rows carry a card string that does not uniquely name a protocol; the earlier collapsed label conflated all of them; see the §9 audit note). Every three-column row is verified against
the card's own printed Recovery percentage; internally inconsistent rows are rejected rather than
guessed at.

## Limitations

- **Two separate limits, easy to confuse.** (1) *Noise* is why per-model prediction fails: most
  differences between models are benchmark measurement noise, not real predictability. (2) *Thin
  evidence* is why the tool flags a scheme-and-size combination: not enough independent published
  checkpoints, which has nothing to do with noise.
- It does not use the model family or size as prediction features — those made out-of-family accuracy
  *worse*.
- Its intervals never exclude zero, so it cannot tell you a config will definitely hurt.
- It is trained only on checkpoints Red Hat chose to publish, so it underpredicts damage from a
  badly-tuned recipe. See the −39.5<!-- claim: adv_worst_delta = -39.5000 -->pp adversarial worst case in `ADVERSARIAL_AUDIT.md` for the
  direct measure of that gap.
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

## Data and licence

This repository is MIT licensed (see `LICENSE`); data provenance is recorded in `NOTICE`. The raw Hugging Face model
cards the dataset was extracted from are **not redistributed** here — they are
RedHatAI's content under several different licences. `data/dataset.csv` (the
extracted numbers) is included and is all the tool needs. One card set **is**
included: `data/rh_card_scan_2026_09_26/`, the dated snapshot behind the
recipe-documentation scan in the paper's Data section, because that claim is
not checkable without it. See
[data/README.md](data/README.md) for what that costs you, why the two are
treated differently, and how to fetch the harvest caches yourself.

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
