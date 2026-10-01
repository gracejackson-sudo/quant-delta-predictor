# Data, and what is not here

## What is included

- **`dataset.csv`** — the 850 extracted rows the tool is built on: model, scheme,
  benchmark, accuracy before, accuracy after, delta. These are numeric results
  transcribed from published model cards. `src/rank.py` needs only this file.
- **`targets.txt`, `redhatai_models.json`** — the list of public model IDs the
  harvest was run over.
- **`adversarial/`** — our own GPU measurements. Ours to publish.
- **`rh_card_scan_2026_09_26/`** — a dated snapshot of the RedHatAI quantized
  model cards as published on that date. This **is** redistributed, and it is
  the one card set that is. It backs the recipe-documentation scan reported in
  the paper's Data section (how many cards pin a library version, mention
  activation reordering or dampening, or report no recipe at all), and it is
  the evidence a reader needs to check that scan rather than take it on trust.
  It also backs the test that verifies every GPQA row's protocol label against
  the card it came from. The derived counts live in
  `out/rh_card_scan_2026_09_26.json`, which is what the claims registry reads;
  the cards themselves are here so the scan is reproducible.

## What is deliberately NOT included

**`cards/` and `prospective_cards/`** — the 178 raw Hugging Face model cards the
dataset was extracted from — are **not redistributed here.**

To state the policy precisely, because the two directories above and this one
pull in opposite directions: we redistribute the **dated recipe-scan
snapshot**, because a documentation claim nobody can check is not worth making,
and we do not redistribute the **harvest caches** that `dataset.csv` was
extracted from. That is a judgement about which cards earn their licence risk,
not a blanket rule, and it is worth knowing that it is a judgement.

Those cards are RedHatAI's published content, not ours, and they carry at least
five different licenses: Apache-2.0, MIT, the Llama 2/3.1/3.2/3.3/4 community
licenses, and the Gemma terms, with a number declaring no license at all.
Redistributing them wholesale would mean asserting rights we do not have. The
extracted numbers in `dataset.csv` are a different matter: a measured benchmark
score is a fact, not creative expression.

## What this costs you, stated plainly

Without the raw cards, three scripts cannot run:

| script | needs cards |
|---|---|
| `src/rank.py` (the tool) | no |
| `tests/` (113 tests) | no |
| `src/verify_claims.py`, `src/check_docs.py` | no |
| `src/audit.py` | **yes** |
| `src/census.py` | **yes** |
| `verify/independent_check.py` | **yes** |

That last one matters and we are not going to paper over it.
`verify/independent_check.py` is the from-scratch reimplementation cited in
the paper (§9 Audit) as independent verification of the headline coverage figure.
You cannot re-run it against source without the cards, so that particular check
is one you have to take on trust or reproduce yourself. The committed
`out/independent_check.csv` records the last run's output; `PROVENANCE.md`
(entry for 2026-09-27) documents the A4 float-boundary fix and the
regen-consistency test in `tests/test_all.py` that will catch a future
regeneration that drops the 1e-9 tolerance.

## Fetching the cards yourself

Every card is public. To rebuild the corpus:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import hf_hub_download
import os, json
os.makedirs("data/cards", exist_ok=True)
for mid in [l.strip() for l in open("data/targets.txt") if l.strip()]:
    try:
        p = hf_hub_download(mid, "README.md", repo_type="model")
        open(f"data/cards/{mid.replace('/', '_')}.md", "w").write(open(p).read())
    except Exception as e:
        print("skip", mid, e)
PY
./.venv/bin/python src/harvest.py   # rebuilds dataset.csv from the cards
```

Cards are gated or renamed from time to time, so a rebuild may recover slightly
fewer rows than the committed `dataset.csv`. `src/harvest.py` reports exactly
what it accepted and rejected.

## Attribution

Source data: evaluation results published by **RedHatAI** on Hugging Face —
<https://huggingface.co/RedHatAI>. The quantized checkpoints were produced with
[llm-compressor](https://github.com/vllm-project/llm-compressor).


## Qwen3.5 (kept separate)

`data/qwen35/published_rows.csv` holds published quantization deltas for Qwen3.5
checkpoints, extracted from Red Hat's model cards by `src/qwen35_published.py`
through the unchanged `harvest.harvest_card` parser. It is **not** part of
`dataset.csv` and is never pooled with it: Qwen3.5 uses a different architecture
(a multimodal wrapper with alternating linear- and full-attention layers) and a
different distillation recipe, so its rows are for side-by-side comparison with
Qwen2.5's only. Its cards report a different benchmark set (GPQA, GSM8K, IFEval,
MMLU-Pro), so comparisons are indicative, not like-for-like. Rejected rows and
their reasons are in `published_rejects.csv`. Raw cards are cached locally under
`out/qwen35_cards/` and are not redistributed.
