---
license: mit
base_model:
- tencent/Hy3
library_name: transformers
tags:
- compressed-tensors
- vllm
- llm-compressor
---

# RedHatAI/Hy3-NVFP4-FP8

This is a quantized version of `tencent/Hy3` with MoE layers quantized to NVFP4 and attention layers quantized to FP8 block

## Usage

This model is intended for deployment with vLLM. You can serve the model using

```bash
vllm serve RedHatAI/Hy3-NVFP4-FP8 \
    --tensor-parallel-size 4 \
    --tool-call-parser hy_v3 \
    --enable-auto-tool-choice \
    --reasoning-parser hy_v3 \
    --port 8089
```

May require https://github.com/vllm-project/vllm/pull/48769 as a prerequisite

## Creation Process
https://github.com/vllm-project/llm-compressor/pull/2928

## Evaluation ##

Observed 100% recovery on a limited sample from

```bash
inspect eval hf/Idavidrein/gpqa/diamond   --model vllm/RedHatAI/Hy3-NVFP4-FP8 --reasoning-effort high --model-base-url http://localhost:8089/v1
```