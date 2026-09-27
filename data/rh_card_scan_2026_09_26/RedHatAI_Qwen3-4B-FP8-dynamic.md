---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-4B
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP8
---

# Qwen3-4B-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen3ForCausalLM
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
- **Release Date:** 05/02/2025
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing activations and weights of [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) to FP8 data type.
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

model_id = "RedHatAI/Qwen3-4B-FP8-dynamic"
number_gpus = 1
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=256)

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
  model_stub = "Qwen/Qwen3-4B"
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

The model was evaluated on the OpenLLM leaderboard tasks (versions 1 and 2), using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and on reasoning tasks using [lighteval](https://github.com/neuralmagic/lighteval/tree/reasoning).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>

  **lm-evaluation-harness**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-4B-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks openllm \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-4B-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks mgsm \
    --apply_chat_template\
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-4B-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=16384,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks leaderboard \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-4B-FP8-dynamic
    dtype: auto
    gpu_memory_utilization: 0.9
    max_model_length: 40960
    generation_parameters:
      temperature: 0.6
      top_k: 20
      min_p: 0.0
      top_p: 0.95
      max_new_tokens: 32768
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|aime24|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|aime25|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|math_500|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|gpqa:diamond|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks extended|lcb:codegeneration \
    --use_chat_template = true
  ```

</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>Qwen3-4B
   </th>
   <th>Qwen3-4B-FP8-dynamic<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>66.76
   </td>
   <td>66.34
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>50.17
   </td>
   <td>49.91
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>60.80
   </td>
   <td>66.11
   </td>
   <td>108.7%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>52.80
   </td>
   <td>53.51
   </td>
   <td>101.3%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>58.41
   </td>
   <td>60.54
   </td>
   <td>103.7%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>51.79
   </td>
   <td>51.52
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>56.79</strong>
   </td>
   <td><strong>57.99</strong>
   </td>
   <td><strong>102.1%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v2</strong>
   </td>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>29.82
   </td>
   <td>28.13
   </td>
   <td>94.3%
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>82.09
   </td>
   <td>83.16
   </td>
   <td>101.3%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>29.69
   </td>
   <td>27.27
   </td>
   <td>91.7%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5 (4-shot)
   </td>
   <td>50.63
   </td>
   <td>51.42
   </td>
   <td>101.6%
   </td>
  </tr>
  <tr>
   <td>GPQA (0-shot)
   </td>
   <td>0.00
   </td>
   <td>0.00
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td>MuSR (0-shot)
   </td>
   <td>11.37
   </td>
   <td>11.00
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>33.93</strong>
   </td>
   <td><strong>33.49</strong>
   </td>
   <td><strong>98.7%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Multilingual</strong>
   </td>
   <td>MGSM (0-shot)
   </td>
   <td>26.67
   </td>
   <td>26.00
   </td>
   <td>97.5%
   </td>
  </tr>
  <tr>
   <td rowspan="6" ><strong>Reasoning<br>(generation)</strong>
   </td>
   <td>AIME 2024
   </td>
   <td>71.35
   </td>
   <td>69.37
   </td>
   <td>97.2%
   </td>
  </tr>
  <tr>
   <td>AIME 2025
   </td>
   <td>59.58
   </td>
   <td>60.73
   </td>
   <td>101.9%
   </td>
  </tr>
  <tr>
   <td>GPQA diamond
   </td>
   <td>55.56
   </td>
   <td>54.04
   </td>
   <td>97.3%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5
   </td>
   <td>95.60
   </td>
   <td>96.80
   </td>
   <td>101.3%
   </td>
  </tr>
  <tr>
   <td>LiveCodeBench
   </td>
   <td>53.03
   </td>
   <td>51.45
   </td>
   <td>97.0%
   </td>
  </tr>
</table>