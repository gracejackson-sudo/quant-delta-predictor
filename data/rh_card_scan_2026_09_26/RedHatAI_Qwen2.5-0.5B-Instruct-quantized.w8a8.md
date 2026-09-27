---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
- chat
- neuralmagic
- llmcompressor
---

# Qwen2.5-0.5B-Instruct-quantized.w8a8

## Model Overview
- **Model Architecture:** Qwen2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 10/09/2024
- **Version:** 1.0
- **License(s):** [apache-2.0](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE)
- **Model Developers:** Neural Magic

Quantized version of [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
It achieves an average score of 43.38 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark version 1 and 23.42 on version 2, whereas the unquantized model achieves 43.64 on version 1 and 23.39 on version 2.

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) to INT8 data type.
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

model_id = "neuralmagic/Qwen2.5-0.5B-Instruct-quantized.w8a8"
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
  --model_args pretrained="neuralmagic/Qwen2.5-0.5B-Instruct-quantized.w8a8",dtype=auto,gpu_memory_utilization=0.9,add_bos_token=True,max_model_len=4096,enable_chunk_prefill=True,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Qwen2.5-0.5B-Instruct</strong>
   </td>
   <td><strong>Qwen2.5-0.5B-Instruct-quantized.w8a8 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>46.83
   </td>
   <td>46.29
   </td>
   <td>98.9%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>33.62
   </td>
   <td>33.36
   </td>
   <td>99.2%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>33.21
   </td>
   <td>33.21
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>51.31
   </td>
   <td>50.97
   </td>
   <td>99.3%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>55.01
   </td>
   <td>55.01
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>41.85
   </td>
   <td>41.47
   </td>
   <td>99.1%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>43.64</strong>
   </td>
   <td><strong>43.38</strong>
   </td>
   <td><strong>99.4%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v2</strong>
   </td>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>17.49
   </td>
   <td>16.95
   </td>
   <td>96.9%
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>31.17
   </td>
   <td>32.04
   </td>
   <td>102.8%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>32.79
   </td>
   <td>32.51
   </td>
   <td>99.2%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5 (4-shot)
   </td>
   <td>0.21
   </td>
   <td>0.17
   </td>
   <td>***
   </td>
  </tr>
  <tr>
   <td>GPQA (0-shot)
   </td>
   <td>25.67
   </td>
   <td>26.12
   </td>
   <td>101.8%
   </td>
  </tr>
  <tr>
   <td>MuSR (0-shot)
   </td>
   <td>33.02
   </td>
   <td>32.75
   </td>
   <td>99.2%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>23.39</strong>
   </td>
   <td><strong>23.42</strong>
   </td>
   <td><strong>100.1%</strong>
   </td>
  </tr>
</table>
*** Reference value too low to report meaningful recovery.
