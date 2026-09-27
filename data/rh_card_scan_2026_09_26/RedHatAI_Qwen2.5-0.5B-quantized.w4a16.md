---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-0.5B
tags:
- chat
- neuralmagic
- llmcompressor
---

# Qwen2.5-0.5B-quantized.w4a16

## Model Overview
- **Model Architecture:** Qwen2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 12/17/2024
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
It achieves an average score of 41.25 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 44.03.

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only weights of the linear operators within transformers blocks are quantized.
Symmetric per-group quantization is applied, in which a linear scaling per group of 64 parameters maps the INT4 and floating point representations of the quantized weights.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic-ent/Qwen2.5-0.5B-quantized.w4a16"
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
  --model_args pretrained="neuralmagic-ent/Qwen2.5-0.5B-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.9,add_bos_token=True,max_model_len=4096,enable_chunk_prefill=True,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Qwen2.5-0.5B</strong>
   </td>
   <td><strong>Qwen2.5-0.5B-quantized.w4a16 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>47.57
   </td>
   <td>45.04
   </td>
   <td>94.7%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>34.90
   </td>
   <td>32.68
   </td>
   <td>98.8%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>34.19
   </td>
   <td>27.98
   </td>
   <td>81.8%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>51.83
   </td>
   <td>49.15
   </td>
   <td>94.8%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>55.80
   </td>
   <td>53.75
   </td>
   <td>96.3%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>39.90
   </td>
   <td>38.89
   </td>
   <td>97.5%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>44.03</strong>
   </td>
   <td><strong>41.25</strong>
   </td>
   <td><strong>93.7%</strong>
   </td>
  </tr>
</table>

