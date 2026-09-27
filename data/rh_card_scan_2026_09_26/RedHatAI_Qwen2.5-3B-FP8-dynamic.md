---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-3B
tags:
- chat
- neuralmagic
- llmcompressor
---

# Qwen2.5-3B-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 11/27/2024
- **Version:** 1.0
- **License(s):** [apache-2.0](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

Quantized version of [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B).
It achieves an average score of 62.50 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 63.59.

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between INT8 and floating point representations for each output channel dimension.
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between INT8 and floating point representations.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Qwen2.5-3B-FP8-dynamic"
number_gpus = 1
max_model_len = 8192

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Give me a short introduction to large language model."

llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

outputs = llm.generate(prompt, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.


## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Qwen2.5-3B-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.9,add_bos_token=True,max_model_len=4096,enable_chunk_prefill=True,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Qwen2.5-3B</strong>
   </td>
   <td><strong>Qwen2.5-3B-FP8-dynamic (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>65.68
   </td>
   <td>65.23
   </td>
   <td>99.3%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>63.59
   </td>
   <td>62.50
   </td>
   <td>98.3%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>68.23
   </td>
   <td>63.91
   </td>
   <td>93.7%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>74.46
   </td>
   <td>74.13
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>70.64
   </td>
   <td>69.93
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>48.93
   </td>
   <td>48.71
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>63.59</strong>
   </td>
   <td><strong>62.50</strong>
   </td>
   <td><strong>98.3%</strong>
   </td>
  </tr>
</table>

