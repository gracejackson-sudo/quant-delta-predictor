---
tags:
- int8
- vllm
- llm-compressor
language:
- en
pipeline_tag: text-generation
license: apache-2.0
base_model:
- Qwen/Qwen2.5-3B
---

# Qwen2.5-3B-quantized.w8a16

## Model Overview
- **Model Architecture:** Qwen2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
- **Intended Use Cases:** Similarly to [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B), this is a base language model.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 10/09/2024
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B).
It achieves an OpenLLMv1 score of 63.8, compared to 63.6 for [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B).

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) to INT8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights of the linear operators within transformers blocks are quantized.
Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the INT8 and floating point representations of the quantized weights.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Qwen2.5-3B-quantized.w8a16"
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

The model was evaluated on the OpenLLMv1 benchmark, composed of MMLU, ARC-Challenge, GSM-8K, Hellaswag, Winogrande and TruthfulQA.
Evaluation was conducted using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the [vLLM](https://docs.vllm.ai/en/stable/) engine.

### Accuracy

<table>
  <tr>
   <td><strong>Category</strong>
   </td>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Qwen2.5-3B</strong>
   </td>
   <td><strong>Qwen2.5-3B-quantized.w8a16<br>(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="8" ><strong>OpenLLM v1</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>65.68
   </td>
   <td>65.65
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>53.58
   </td>
   <td>53.07
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td>GSM-8k (5-shot, strict-match)
   </td>
   <td>68.23
   </td>
   <td>70.05
   </td>
   <td>102.7%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>51.83
   </td>
   <td>51.78
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>70.64
   </td>
   <td>70.56
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
  <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>49.93
   </td>
   <td>48.88
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>63.59</strong>
   </td>
   <td><strong>63.78</strong>
   </td>
   <td><strong>100.3%</strong>
   </td>
  </tr>
</table>

### Reproduction

The results were obtained using the following command:

```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Qwen2.5-3B-quantized.w8a16",dtype=auto,max_model_len=4096,add_bos_token=True,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```
