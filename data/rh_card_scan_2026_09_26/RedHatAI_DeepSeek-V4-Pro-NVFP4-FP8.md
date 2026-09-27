---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Pro
library_name: transformers
tags:
- compressed-tensors
- vLLM
---

# RedHatAI/DeepSeek-V4-Pro-NVFP4-FP8

This is a quantized version of `deepseek-ai/DeepSeek-V4-Pro` with MoE layers quantized to NVFP4 and attention layers quantized to FP8 block

## Usage

This model is intended for deployment with vLLM and requires the following branch: https://github.com/vllm-project/vllm/pull/41276.
You can serve the model using

```bash
vllm serve RedHatAI/DeepSeek-V4-Pro-NVFP4-FP8-BLOCK --tensor_parallel_size 8 --kv_cache_dtype=fp8
```

## Creation Process

This model was created using [LLM Compressor](https://github.com/vllm-project/llm-compressor). The example script can be found in `examples/quantizing_moe/deepseek_v4_pro_example.py` [[DSV4] DeepSeekV4 Pro](https://github.com/vllm-project/llm-compressor/pull/2858). Quantizing the model with data parallelism and 6xA100 takes about 3 hours.


## Evaluation ##
| Benchmark | `deepseek-ai/DeepSeek-V4-Pro-Base` | `deepseek-ai/DeepSeek-V4-Pro` | `RedHatAI/DeepSeek-V4-Pro-NVFP4-FP8` |
| - | - | -| - |
| GPQA | | 90.1 | 0.93 (380/792 samples) |
| GSM8K | 91.1 | 92.6 | 91.0 |