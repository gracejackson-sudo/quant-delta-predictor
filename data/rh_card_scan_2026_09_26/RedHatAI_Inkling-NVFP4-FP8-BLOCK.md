---
license: mit
base_model:
- thinkingmachines/Inkling
library_name: transformers
tags:
- compressed-tensors
- llm-compressor
- vllm
- nvfp4
---

# RedHatAI/Inkling-NVFP4-FP8-BLOCK

This is a quantized version of `thinkingmachines/Inkling` with MoE layers quantized to NVFP4 and attention layers quantized to FP8 Block.

## Usage

This model is intended for deployment with vLLM. You can serve the model using 4xB200s:

```bash
vllm serve RedHatAI/Inkling-NVFP4-FP8-BLOCK \
  --tokenizer-mode inkling \
  --tensor-parallel-size 4 \ 
  --enable-auto-tool-choice \
  --tool-call-parser inkling \
  --reasoning-parser inkling \
  --enable-expert-parallel \
  --max-model-len 100000 \
  --max-num-seqs 32 \
  --enable-chunked-prefill \
  --trust-remote-code \
```

## Evaluation ##
Evaluations were performed with `inspectai`:

```bash
inspect eval hf/Idavidrein/gpqa/diamond \
  --model RedHatAI/Inkling-NVFP4-FP8-BLOCK \
  --reasoning-effort high \
  --model-base-url http://localhost:8000/v1 \
```

| Benchmark | `thinkingmachines/Inkling` | `RedHatAI/Inkling-NVFP4-FP8-BLOCK` |
| - | - | - |
| GPQA Diamond | 87.2 | 84.5 |

Note: A bug in `inspectai` currently misscores markdown-formatted multiple choice answers (e.g. `ANSWER: $B$` rather than `ANSWER: B`), which the Inkling models often include in their response. Scores are manually evaluated to account for discrepancy. More information at https://github.com/UKGovernmentBEIS/inspect_ai/issues/5145
