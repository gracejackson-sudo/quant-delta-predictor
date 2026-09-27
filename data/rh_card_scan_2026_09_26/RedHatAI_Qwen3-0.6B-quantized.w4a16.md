---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-0.6B
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
---

# Qwen3-0.6B-quantized.w4a16

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

This model was obtained by quantizing the weights of [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a asymmetric per-group scheme, with group size 64.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-0.6B-quantized.w4a16"
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
model_stub = "Qwen/Qwen3-0.6B"
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
    config_groups={
        "group0": {
            "targets": ["Linear"]
            "weights": {
                "num_bits": 4,
                "type": "int",
                "strategy": "group",
                "group_size": 64,
                "symmetric": False,
                "actorder": "weight",
                "observer": "mse",
            }
        }
    }
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
    --model_args pretrained="RedHatAI/Qwen3-0.6B-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks openllm \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-0.6B-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks mgsm \
    --apply_chat_template\
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-0.6B-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=16384,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks leaderboard \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-0.6B-quantized.w4a16
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
   <th>Qwen3-0.6B
   </th>
   <th>Qwen3-0.6B-quantized.w4a16<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>42.82
   </td>
   <td>39.80
   </td>
   <td>93.00%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>32.85
   </td>
   <td>30.72
   </td>
   <td>93.5%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>1.82
   </td>
   <td>2.20
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>43.04
   </td>
   <td>41.02
   </td>
   <td>95.3%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>54.54
   </td>
   <td>54.62
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>51.61
   </td>
   <td>48.77
   </td>
   <td>94.5%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>37.78</strong>
   </td>
   <td><strong>36.19</strong>
   </td>
   <td><strong>95.8%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v2</strong>
   </td>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>17.25
   </td>
   <td>14.27
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>62.83
   </td>
   <td>55.81
   </td>
   <td>88.8%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>4.23
   </td>
   <td>1.63
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5 (4-shot)
   </td>
   <td>18.26
   </td>
   <td>10.26
   </td>
   <td>---
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
   <td>0.00
   </td>
   <td>0.00
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>17.10</strong>
   </td>
   <td><strong>13.66</strong>
   </td>
   <td><strong>---</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Multilingual</strong>
   </td>
   <td>MGSM (0-shot)
   </td>
   <td>19.70
   </td>
   <td>19.90
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td rowspan="6" ><strong>Reasoning<br>(generation)</strong>
   </td>
   <td>AIME 2024
   </td>
   <td>9.69
   </td>
   <td>3.44
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td>AIME 2025
   </td>
   <td>13.13
   </td>
   <td>6.98
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td>GPQA diamond
   </td>
   <td>29.29
   </td>
   <td>27.78
   </td>
   <td>94.8%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5
   </td>
   <td>71.60
   </td>
   <td>70.60
   </td>
   <td>98.6%
   </td>
  </tr>
  <tr>
   <td>LiveCodeBench
   </td>
   <td>12.83
   </td>
   <td>8.35
   </td>
   <td>---
   </td>
  </tr>
</table>