---
language:
  - en
  - fr
  - es
  - it
  - pt
  - zh
  - ar
  - ru
base_model:
- HuggingFaceTB/SmolLM3-3B
pipeline_tag: text-generation
tags:
- smollm3
- fp8
- vllm
- conversational
- compressed-tensors
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/SmolLM3-3B-FP8-dynamic
description: This model was obtained by quantizing activation and weights of SmolLM3-3B to FP8 data type.
readme: https://huggingface.co/RedHatAI/SmolLM3-3B-FP8-dynamic/main/README.md
tasks:
- text-to-text
- text-generation
provider: HuggingFaceTB
license_link: https://www.apache.org/licenses/LICENSE-2.0
---

## Model Overview
- **Model Architecture:** SmolLM3-3B
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 07/28/2025
- **Version:** 1.0
- **License(s):** Apache-2.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing activation and weights of [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) to FP8 data type.
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

model_id = "RedHatAI/SmolLM3-3B-FP8-dynamic"
number_gpus = 1

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False)

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
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from llmcompressor.modifiers.quantization import QuantizationModifier
  from llmcompressor.transformers import oneshot
  
  # Load model
  model_stub = "HuggingFaceTB/SmolLM3-3B"
  model_name = model_stub.split("/")[-1]
  
  tokenizer = AutoTokenizer.from_pretrained(model_stub)
  
  model = AutoModelForCausalLM.from_pretrained(
      model_stub,
      device_map="auto",
      torch_dtype="auto",
  )
  
  # Configure the quantization algorithm and scheme
  recipe = QuantizationModifier(
      targets="Linear",
      scheme="FP8_dynamic",
      ignore=["lm_head"],
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

This model was evaluated on the well-known reasoning tasks: AIME24, Math-500, and GPQA-Diamond.
In all cases, model outputs were generated with the [vLLM](https://docs.vllm.ai/en/stable/) engine, and evals are collected through [LightEval](https://github.com/huggingface/lighteval) library.


<details>
  <summary>Evaluation details</summary>

  ```
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export MODEL="RedHatAI/SmolLM3-3B-FP8-dynamic"
    export MODEL_ARGS="model_name=$MODEL,dtype=auto,max_model_length=65536,gpu_memory_utilization=0.9,tensor_parallel_size=1,add_special_tokens=False,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

    export TASK=aime24 # {aime24, math_500, gpqa:diamond}

    lighteval vllm $MODEL_ARGS "lighteval|${TASK}|0|0" \
        --use-chat-template \
        --output-dir out_dir
  ```
</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>HuggingFaceTB/SmolLM3-3B
   </th>
   <th>RedHatAI/SmolLM3-3B-FP8-dynamic<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="8" ><strong>Reasoning</strong>
   </td>
   <td>AIME24 (pass@1:64)
   </td>
   <td>45.31
   </td>
   <td>47.50
   </td>
   <td>104.83%
   </td>
  </tr>
  <tr>
   <td>MATH-500 (pass@1:4)
   </td>
   <td>89.30
   </td>
   <td>88.30
   </td>
   <td>98.88%
   </td>
  </tr>
  <tr>
   <td>GPQA-Diamond (pass@1:8)
   </td>
   <td>41.22
   </td>
   <td>40.91
   </td>
   <td>99.25%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>58.61</strong>
   </td>
   <td><strong>58.90</strong>
   </td>
   <td><strong>100.5%</strong>
   </td>
  </tr>
  <tr>
</table>



