---
language:
- en
pipeline_tag: text-generation
license: llama3
license_link: https://llama.meta.com/llama3/license/
---

# Meta-Llama-3-8B-Instruct-quantized.w8a16

## Model Overview
- **Model Architecture:** Meta-Llama-3
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/2/2024
- **Version:** 1.0
- **License(s):** [Llama3](https://llama.meta.com/llama3/license/)
- **Model Developers:** Neural Magic

Quantized version of [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
It achieves an average score of 68.69 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 68.54.

### Model Optimizations

This model was obtained by quantizing the weights of [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) to INT8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights of the linear operators within transformers blocks are quantized. Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the INT8 and floating point representations of the quantized weights.
[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) is used for quantization with 1% damping factor and 256 sequences of 8,192 random tokens.


## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16"
number_gpus = 1

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

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

This model is supported by Transformers leveraging the integration with the [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) data format.
The following example contemplates how the model can be used using the `generate()` function.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16"

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
import random

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

num_samples = 256
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_id)

max_token_id = len(tokenizer.get_vocab()) - 1
examples = []
for _ in range(num_samples):
  examples.append(
  {
    "input_ids": [random.randint(0, max_token_id) for _ in range(max_seq_len)],
    "attention_mask": max_seq_len*[1],
  }
)

quantize_config = BaseQuantizeConfig(
  bits=8,
  group_size=-1,
  desc_act=False,
  model_file_base_name="model",
  damp_percent=0.01,
)

model = AutoGPTQForCausalLM.from_pretrained(
  model_id,
  quantize_config,
  device_map="auto",
)

model.quantize(examples)
model.save_pretrained("Meta-Llama-3-8B-Instruct-quantized.w8a16")
```



## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16",dtype=auto,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Meta-Llama-3-8B-Instruct </strong>
   </td>
   <td><strong>Meta-Llama-3-8B-Instruct-quantized.w8a16(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>66.54
   </td>
   <td>66.55
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>62.63
   </td>
   <td>61.52
   </td>
   <td>98.2%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>75.21
   </td>
   <td>75.89
   </td>
   <td>100.9%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>78.81
   </td>
   <td>78.69
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>76.48
   </td>
   <td>76.01
   </td>
   <td>98.2%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>52.49
   </td>
   <td>52.60
   </td>
   <td>100.2%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>68.69</strong>
   </td>
   <td><strong>68.54</strong>
   </td>
   <td><strong>99.8%</strong>
   </td>
  </tr>
</table>