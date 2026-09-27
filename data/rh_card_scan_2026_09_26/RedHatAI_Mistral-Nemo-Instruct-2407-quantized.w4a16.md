---
language:
- en
pipeline_tag: text-generation
license: llama2
---

# Mistral-Nemo-Instruct-2407-quantized.w4a16

## Model Overview
- **Model Architecture:** Mistral-Nemo
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 8/16/2024
- **Version:** 1.0
- **License(s)**: [Apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
- **Model Developers:** Neural Magic

Quantized version of [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407).
It achieves an average score of 70.13 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 71.61.

### Model Optimizations

This model was obtained by quantizing the weights of [Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized. Symmetric group-wise quantization is applied, in which a linear scaling per group maps the INT4 and floating point representations of the quantized weights.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library. Quantization is performed with 1% damping factor, group-size as 128 and 512 sequences sampled from [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Mistral-Nemo-Instruct-2407-quantized.w4a16"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you? Please respond in pirate speak."},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_id, tensor_parallel_size=2)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

### Use with transformers

The following example contemplates how the model can be deployed in Transformers using the `generate()` function.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "neuralmagic/Mistral-Nemo-Instruct-2407-quantized.w4a16"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you? Please respond in pirate speak"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

## Creation

This model was created by using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as presented in the code snipet below.

```python
from transformers import AutoTokenizer
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from datasets import load_dataset
import random

model_id = "mistralai/Mistral-Nemo-Instruct-2407"

num_samples = 512
max_seq_len = 4096

tokenizer = AutoTokenizer.from_pretrained(model_id)

preprocess_fn = lambda example: {"text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n{text}".format_map(example)}

dataset_name = "neuralmagic/LLM_compression_calibration"
dataset = load_dataset(dataset_name, split="train")
ds = dataset.shuffle().select(range(num_samples))
ds = ds.map(preprocess_fn)

examples = [
    tokenizer(
        example["text"], padding=False, max_length=max_seq_len, truncation=True,
    ) for example in ds
]

recipe = GPTQModifier(
  targets="Linear",
  scheme="W4A16",
  ignore=["lm_head"],
  dampening_frac=0.01,
)

model = SparseAutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="auto",
  trust_remote_code=True,
)

oneshot(
  model=model,
  dataset=ds,
  recipe=recipe,
  max_seq_length=max_seq_len,
  num_calibration_samples=num_samples,
)

model.save_pretrained("Mistral-Nemo-Instruct-2407-quantized.w4a16")
```


## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Mistral-Nemo-Instruct-2407-quantized.w4a16",dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096,trust_remote_code=True \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Mistral-Nemo-Instruct-2407 </strong>
   </td>
   <td><strong>Mistral-Nemo-Instruct-2407-quantized.w4a16(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>68.35
   </td>
   <td>66.92
   </td>
   <td>97.91%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>65.53
   </td>
   <td>64.93
   </td>
   <td>99.08%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>74.45
   </td>
   <td>70.43
   </td>
   <td>94.60%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>84.32
   </td>
   <td>83.6
   </td>
   <td>99.15%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>82.16
   </td>
   <td>80.58
   </td>
   <td>98.08%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>54.85
   </td>
   <td>54.33
   </td>
   <td>99.05%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>71.61</strong>
   </td>
   <td><strong>70.13</strong>
   </td>
   <td><strong>97.93%</strong>
   </td>
  </tr>
</table>