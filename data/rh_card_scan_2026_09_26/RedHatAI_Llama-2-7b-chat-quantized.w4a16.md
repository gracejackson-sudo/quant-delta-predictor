---
language:
- en
pipeline_tag: text-generation
license: llama2
---

# Llama-2-7b-chat-quantized.w4a16

## Model Overview
- **Model Architecture:** Llama-2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/11/2024
- **Version:** 1.0
- **License(s)**: [LLama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/LICENSE.txt)
- **Model Developers:** Neural Magic

Quantized version of [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat).
It achieves an average score of 52.64 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 53.40.

### Model Optimizations

This model was obtained by quantizing the weights of [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 25%.

Only the weights of the linear operators within transformers blocks are quantized. Symmetric group-wise quantization is applied, in which a linear scaling per group maps the INT4 and floating point representations of the quantized weights.
[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) is used for quantization with 10% damping factor, group-size as 128 and 512 sequences sampled from [Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus).


## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Llama-2-7b-chat-quantized.w4a16"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_id, tensor_parallel_size=2)

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

model_id = "neuralmagic/Llama-2-7b-chat-quantized.w4a16"

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

model_id = "meta-llama/Llama-2-7b-chat"

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
model.save_pretrained("Llama-2-7b-chat-quantized.w4a16")
```



## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-2-7b-chat-quantized.w4a16",dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Llama-2-7b-chat </strong>
   </td>
   <td><strong>Llama-2-7b-chat-quantized.w4a16(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>47.33
   </td>
   <td>46.7
   </td>
   <td>98.66%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>53.24
   </td>
   <td>53.75
   </td>
   <td>100.95%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>23.19
   </td>
   <td>21.75
   </td>
   <td>93.79%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>78.64
   </td>
   <td>77.35
   </td>
   <td>98.36%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>72.45
   </td>
   <td>70.4
   </td>
   <td>98.66%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>45.58
   </td>
   <td>45.94
   </td>
   <td>100.78%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>53.40</strong>
   </td>
   <td><strong>52.64</strong>
   </td>
   <td><strong>98.58%</strong>
   </td>
  </tr>
</table>