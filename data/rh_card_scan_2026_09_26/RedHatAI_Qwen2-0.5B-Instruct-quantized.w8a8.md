---
language:
- en
pipeline_tag: text-generation
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
---

# Qwen2-0.5B-Instruct-quantized.w8a8

## Model Overview
- **Model Architecture:** Qwen2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/12/2024
- **Version:** 1.0
- **License(s):** [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Model Developers:** Neural Magic

Quantized version of [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct).
It achieves an average score of 42.75 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 43.06.

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between INT8 and floating point representations for each output channel dimension.
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between INT8 and floating point representations.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.
GPTQ used a 1% damping factor and 256 sequences of 8,192 random tokens.


## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Qwen2-0.5B-Instruct-quantized.w8a8"
number_gpus = 1

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

### Use with transformers

The following example contemplates how the model can be deployed in Transformers using the `generate()` function.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "neuralmagic/Qwen2-0.5B-Instruct-quantized.w8a8"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
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
    temperature=0.7,
    top_p=0.8,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

## Creation

This model was created by using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as presented in the code snipet below.

```python
from transformers import AutoTokenizer
from datasets import Dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import random

model_id = "Qwen/Qwen2-0.5B-Instruct"

num_samples = 256
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_id)

max_token_id = len(tokenizer.get_vocab()) - 1
input_ids = [[random.randint(0, max_token_id) for _ in range(max_seq_len)] for _ in range(num_samples)]
attention_mask = num_samples * [max_seq_len * [1]]
ds = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})

recipe = GPTQModifier(
  targets="Linear",
  scheme="W8A8",
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

model.save_pretrained("Qwen2-0.5B-Instruct-quantized.w8a18)
```



## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Qwen2-0.5B-Instruct-quantized.w8a8",dtype=auto,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Qwen2-0.5B-Instruct</strong>
   </td>
   <td><strong>Qwen2-0.5B-Instruct-quantized.w8a8 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>43.72
   </td>
   <td>43.41
   </td>
   <td>99.3%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>31.83
   </td>
   <td>31.57
   </td>
   <td>99.2%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>37.68
   </td>
   <td>37.38
   </td>
   <td>99.2%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>49.50
   </td>
   <td>49.58
   </td>
   <td>100.2%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>56.27
   </td>
   <td>55.01
   </td>
   <td>97.8%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>39.38
   </td>
   <td>39.55
   </td>
   <td>100.4%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>43.06</strong>
   </td>
   <td><strong>42.75</strong>
   </td>
   <td><strong>99.3%</strong>
   </td>
  </tr>
</table>