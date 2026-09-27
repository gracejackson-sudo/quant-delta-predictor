---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-235B-A22B
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP8
---

# Qwen3-235B-A22B-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen3MoeForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP8
  - **Weight quantization:** FP8
- **Intended Use Cases:**
  - Reasoning.
  - Function calling.
  - Subject matter experts via fine-tuning.
  - Multilingual instruction following.
  - Translation.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 05/05/2025
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing activations and weights of [Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-235B-A22B-FP8-dynamic"
number_gpus = 4
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=256)

messages = [
    {"role": "user", "content": prompt}
]

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


  ```python
  from llmcompressor.modifiers.quantization import QuantizationModifier
  from llmcompressor.transformers import oneshot
  from transformers import AutoModelForCausalLM, AutoTokenizer
  
  # Load model
  model_stub = "Qwen/Qwen3-235B-A22B"
  model_name = model_stub.split("/")[-1]

  model = AutoModelForCausalLM.from_pretrained(model_stub)

  tokenizer = AutoTokenizer.from_pretrained(model_stub)

  # Configure the quantization algorithm and scheme
  recipe = QuantizationModifier(
      ignore=["lm_head"],
      targets="Linear",
      scheme="FP8_dynamic",
  )

  # Apply quantization
  oneshot(
      model=model,
      recipe=recipe,
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-FP8-dynamic"
  model.save_pretrained(save_path)
  tokenizer.save_pretrained(save_path)
  print(f"Model and tokenizer saved to: {save_path}")
  ```
</details>
 


## Evaluation

The model was evaluated on the OpenLLM leaderboard tasks (version 1), using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [vLLM](https://docs.vllm.ai/en/stable/).

<details>
  <summary>Evaluation details</summary>

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-235B-A22B-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=4 \
    --tasks openllm \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```
</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>Qwen3-235B-A22B
   </th>
   <th>Qwen3-235B-A22B-FP8-dynamic<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>84.77
   </td>
   <td>84.61
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>71.84
   </td>
   <td>70.90
   </td>
   <td>98.7%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>74.22
   </td>
   <td>74.98
   </td>
   <td>101.0%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>76.56
   </td>
   <td>76.10
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>73.95
   </td>
   <td>75.06
   </td>
   <td>101.5%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>61.18
   </td>
   <td>60.93
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>73.75</strong>
   </td>
   <td><strong>73.76</strong>
   </td>
   <td><strong>100.0%</strong>
   </td>
  </tr>
</table>