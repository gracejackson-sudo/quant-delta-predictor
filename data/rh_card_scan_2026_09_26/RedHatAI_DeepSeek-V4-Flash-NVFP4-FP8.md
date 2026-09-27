---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash
library_name: transformers
tags:
- compressed-tensors
- nvfp4
- vllm
---
<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
DeepSeek-V4-Flash-NVFP4-FP8
</h1>

## Model Optimizations

This model was obtained by using the following branch with LLM Compressor: https://github.com/vllm-project/llm-compressor/pull/2647

## Deployment

This model was deployed using the following branch with vLLM: https://github.com/vllm-project/vllm/pull/41276
```bash
vllm serve RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8 --tensor-parallel-size 4 --port 8089 --kv_cache_dtype="fp8"
```

## Evaluation
This model has a noticably lower accuracy recovery than the base model due to the base model being released in a quantized format and differences between mxfp4 and nvfp4.
More advanced techniques such as GPTQ can be used to increase accuracy recovery beyond this model's current state.

```bash
python tests/evals/gsm8k/gsm8k_eval.py
```

```
Results:
Accuracy: 0.910
Invalid responses: 0.000
Total latency: 173.006 s
Questions per second: 7.624
Total output tokens: 116217
Output tokens per second: 671.752
```

```bash
python3 tests/evals/mmlu_pro/mmlu_pro_eval.py --port 8089
```

```
Results:
Category: all
Accuracy: 0.554
Invalid responses: 0.000
Total latency: 112.065 s
Questions per second: 107.366
Total output tokens: 24076
Output tokens per second: 214.840
```

For more details on how this model was created and run in LLM Compressor, please contact Kyle Sayers on the vLLM Slack: https://communityinviter.com/apps/vllm-dev/join-vllm-developers-slack




## Installation

To run this model in vllm, install the following: 
```bash
uv pip install git+https://github.com/vllm-project/vllm.git@refs/pull/41276/head --no-cache
uv pip install tilelang==0.1.10 apache-tvm-ffi==0.1.10
```

## Accuracy Recovery Summary

Evaluation performed on 8×B200 GPUs using [vLLM](https://github.com/vllm-project/vllm) with FP8 KV cache.
Scores are averaged across multiple seeds (3 seeds for most benchmarks, 8 for AIME 2025).
Instruct benchmarks run with reasoning OFF (nonthinking mode); Reasoning and Coding benchmarks run with reasoning ON (thinking mode).

| Category | Benchmark | deepseek-ai/<br>DeepSeek-V4-Flash | RedHatAI/<br>DeepSeek-V4-Flash-NVFP4-FP8<br>(this model) | Recovery |
|:---------|:----------|:-----------------------------:|:--------------------------------------------------:|:--------:|
| Instruct | MMLU-CoT (5-shot) | 86.10 | 78.39 | 91.05% |
| Instruct | GSM8K Platinum (5-shot) | 96.99 | 94.07 | 96.99% |
| Instruct | MATH-500 | 91.93 | 89.73 | 97.61% |
| Reasoning | GSM8K Platinum (0-shot) | 95.62 | 94.13 | 98.44% |
| Reasoning | MATH-500 | 91.67 | 89.87 | 98.04% |
| Reasoning | AIME 2025 | 52.92 | 72.08 | 136.22% |
| Coding | LCB CodeGen v6 | 51.81 | 48.00 | 92.65% |

