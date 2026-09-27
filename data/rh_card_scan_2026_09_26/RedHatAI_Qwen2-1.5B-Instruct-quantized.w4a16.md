---
language:
- en
pipeline_tag: text-generation
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
---

# Qwen2-1.5B-Instruct-quantized.w4a16

## Model Overview
- **Model Architecture:** Qwen2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/11/2024
- **Version:** 1.0
- **License(s):** [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Model Developers:** Neural Magic

Quantized version of [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct).
It achieves an average score of 53.98 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 55.17.

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 25%.

Only the weights of the linear operators within transformers blocks are quantized. Symmetric group-wise quantization is applied, in which a linear scaling per group maps the INT4 and floating point representations of the quantized weights.
[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) is used for quantization with 10% damping factor, group-size as 128 and 512 sequences sampled from [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus).


## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Qwen2-1.5B-Instruct-quantized.w4a16"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_id, tensor_parallel_size=1)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

### Use with transformers

This model is supported by Transformers leveraging the integration with the [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) data format.
The following example contemplates how the model can be used using the `generate()` function.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "neuralmagic/Qwen2-1.5B-Instruct-quantized.w4a16"

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
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

## Creation

This model was created by applying the [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) library as presented in the code snipet below.
Although AutoGPTQ was used for this particular model, Neural Magic is transitioning to using [llm-compressor](https://github.com/vllm-project/llm-compressor) which supports several quantization schemes and models not supported by AutoGPTQ.

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import random

model_id = "Qwen/Qwen2-1.5B-Instruct"

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

quantize_config = BaseQuantizeConfig(
  bits=4,
  group_size=128,
  desc_act=True,
  model_file_base_name="model",
  damp_percent=0.1,
)

model = AutoGPTQForCausalLM.from_pretrained(
  model_id,
  quantize_config,
  device_map="auto",
)

model.quantize(examples)
model.save_pretrained("Qwen2-1.5B-Instruct-quantized.w4a16")
```



## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Qwen2-1.5B-Instruct-quantized.w4a16",dtype=auto,tensor_parallel_size=1,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Qwen2-1.5B-Instruct </strong>
   </td>
   <td><strong>Qwen2-1.5B-Instruct-quantized.w4a16(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>55.64
   </td>
   <td>54.38
   </td>
   <td>97.74%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>42.83
   </td>
   <td>41.80
   </td>
   <td>97.61%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>58.07
   </td>
   <td>55.27
   </td>
   <td>95.17%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>67.42
   </td>
   <td>65.26
   </td>
   <td>96.80%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>63.69
   </td>
   <td>63.69
   </td>
   <td>100.00%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>43.37
   </td>
   <td>43.50
   </td>
   <td>100.31%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>55.17</strong>
   </td>
   <td><strong>53.98</strong>
   </td>
   <td><strong>97.85%</strong>
   </td>
  </tr>
</table>