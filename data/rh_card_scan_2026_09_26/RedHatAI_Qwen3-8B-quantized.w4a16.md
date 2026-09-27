---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-8B
hardware_tag:
- Intel Xeon
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
---

# Qwen3-8B-quantized.w4a16

## Model Overview
- **Model Architecture:** Qwen3ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
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

This model was obtained by quantizing the weights of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a asymmetric per-group scheme, with group size 64.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-8B-quantized.w4a16"
number_gpus = 1
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
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
  
# Load model
model_stub = "Qwen/Qwen3-8B"
model_name = model_stub.split("/")[-1]

num_samples = 1024
max_seq_len = 8192

model = AutoModelForCausalLM.from_pretrained(model_stub)

tokenizer = AutoTokenizer.from_pretrained(model_stub)

def preprocess_fn(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}
  
ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.map(preprocess_fn)

# Configure the quantization algorithm and scheme
recipe = GPTQModifier(
    ignore=["lm_head"],
    sequential_targets=["Qwen3DecoderLayer"],
    targets="Linear",
    dampening_frac=0.01,
    scheme="W4A16",
  )

  # Apply quantization
  oneshot(
      model=model,
      dataset=ds, 
      recipe=recipe,
      max_seq_length=max_seq_len,
      num_calibration_samples=num_samples,
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-quantized.w4a16"
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
    --model_args pretrained="RedHatAI/Qwen3-8B-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks openllm \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-8B-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks mgsm \
    --apply_chat_template\
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-8B-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=16384,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks leaderboard \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-8B-quantized.w4a16
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
   <th>Qwen3-8B
   </th>
   <th>Qwen3-8B-quantized.w4a16<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>71.95
   </td>
   <td>69.74
   </td>
   <td>96.9%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>61.69
   </td>
   <td>61.77
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>75.97
   </td>
   <td>78.62
   </td>
   <td>103.5%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>56.52
   </td>
   <td>57.79
   </td>
   <td>102.2%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>65.98
   </td>
   <td>66.22
   </td>
   <td>100.4%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>53.17
   </td>
   <td>53.71
   </td>
   <td>101.0%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>64.21</strong>
   </td>
   <td><strong>64.64</strong>
   </td>
   <td><strong>100.7%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v2</strong>
   </td>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>34.57
   </td>
   <td>25.71
   </td>
   <td>74.4%
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>84.77
   </td>
   <td>85.44
   </td>
   <td>100.8%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>25.47
   </td>
   <td>21.17
   </td>
   <td>83.1%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5 (4-shot)
   </td>
   <td>51.05
   </td>
   <td>51.38
   </td>
   <td>100.7%
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
   <td>10.02
   </td>
   <td>9.31
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>34.31</strong>
   </td>
   <td><strong>32.17</strong>
   </td>
   <td><strong>93.8%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Multilingual</strong>
   </td>
   <td>MGSM (0-shot)
   </td>
   <td>25.97
   </td>
   <td>24.73
   </td>
   <td>95.3%
   </td>
  </tr>
  <tr>
   <td rowspan="6" ><strong>Reasoning<br>(generation)</strong>
   </td>
   <td>AIME 2024
   </td>
   <td>74.58
   </td>
   <td>74.17
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>AIME 2025
   </td>
   <td>65.21
   </td>
   <td>61.98
   </td>
   <td>95.1%
   </td>
  </tr>
  <tr>
   <td>GPQA diamond
   </td>
   <td>58.59
   </td>
   <td>55.56
   </td>
   <td>94.8%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5
   </td>
   <td>97.60
   </td>
   <td>96.20
   </td>
   <td>98.6%
   </td>
  </tr>
  <tr>
   <td>LiveCodeBench
   </td>
   <td>56.27
   </td>
   <td>52.29
   </td>
   <td>92.9%
   </td>
  </tr>
</table>