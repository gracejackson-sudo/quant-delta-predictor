---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-7B
tags:
- chat
- neuralmagic
- llmcompressor
- int4
---

# Qwen2.5-7B-quantized.w4a16

## Model Overview
- **Model Architecture:** Qwen2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 10/18/2024
- **Version:** 1.0
- **License(s):** [apache-2.0](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-group scheme, with group size 128.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen2.5-7B-quantized.w4a16"
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

## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from llmcompressor.modifiers.quantization import GPTQModifier
  from llmcompressor.transformers import oneshot
  from datasets import load_dataset
  
  # Load model
  model_stub = "Qwen/Qwen2.5-7B"
  model_name = model_stub.split("/")[-1]
  
  num_samples = 3072
  max_seq_len = 8192
  
  tokenizer = AutoTokenizer.from_pretrained(model_stub)
  
  model = AutoModelForCausalLM.from_pretrained(
      model_stub,
      device_map="auto",
      torch_dtype="auto",
  )
  
  def preprocess_fn(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}
  
  ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
  ds = ds.map(preprocess_fn)
  
  # Configure the quantization algorithm and scheme
  recipe = GPTQModifier(
      targets="Linear",
      scheme="W4A16",
      ignore=["lm_head"],
      sequential_targets=["Qwen2DecoderLayer"],
      dampening_frac=0.1,
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

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/387Bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 387Bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Qwen2.5-7B-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.9,add_bos_token=True,max_model_len=4096,enable_chunk_prefill=True,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <th>Benchmark
   </th>
   <th>Qwen2.5-7B
   </th>
   <th>Qwen2.5-7B-quantized.w4a16<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>74.15
   </td>
   <td>73.47
   </td>
   <td>99.1%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>59.39
   </td>
   <td>58.70
   </td>
   <td>98.9%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>79.76
   </td>
   <td>79.08
   </td>
   <td>99.1%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>80.17
   </td>
   <td>79.39
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>75.69
   </td>
   <td>76.01
   </td>
   <td>100.4%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>56.38
   </td>
   <td>55.48
   </td>
   <td>98.4%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>70.92</strong>
   </td>
   <td><strong>70.35</strong>
   </td>
   <td><strong>99.2%</strong>
   </td>
  </tr>
</table>

