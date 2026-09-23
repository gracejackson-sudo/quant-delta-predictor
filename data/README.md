# Data, and what is not here

## What is included

- **`dataset.csv`** — the 850 extracted rows the tool is built on: model, scheme,
  benchmark, accuracy before, accuracy after, delta. These are numeric results
  transcribed from published model cards. `src/rank.py` needs only this file.
- **`targets.txt`, `redhatai_models.json`** — the list of public model IDs the
  harvest was run over.
- **`adversarial/`** — our own GPU measurements. Ours to publish.

## What is deliberately NOT included

**`cards/` and `prospective_cards/`** — the 178 raw Hugging Face model cards the
dataset was extracted from — are **not redistributed here.**

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
FINDINGS.md as independent verification of the headline coverage figure. You
cannot re-run it against source without the cards, so that particular check is
one you have to take on trust or reproduce yourself.

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
