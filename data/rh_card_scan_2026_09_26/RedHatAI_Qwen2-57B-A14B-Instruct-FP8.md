---
tags:
- fp8
- vllm
license: apache-2.0
---

# Qwen2-57B-A14B-Instruct-FP8

## Model Overview
- **Model Architecture:** Qwen2-57B-A14B-Instruct
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Meta-Llama-3-7B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-7B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/17/2024
- **Version:** 1.0
- **License(s):** [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
- **Model Developers:** Neural Magic

Quantized version of [Qwen2-57B-A14B-Instruct](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct).
It achieves an average score of 74.03 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 74.96.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen2-57B-A14B-Instruct](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct) to FP8 data type, ready for inference with vLLM >= 0.5.0.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-tensor quantization is applied, in which a single linear scaling maps the FP8 representations of the quantized weights and activations.
[AutoFP8](https://github.com/neuralmagic/AutoFP8) is used for quantization with 512 sequences of UltraChat.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Qwen2-57B-A14B-Instruct-FP8"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False)

llm = LLM(model=model_id)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [AutoFP8 with calibration samples from ultrachat](https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py) with the MoE gates kept at original precision, as specified below.
However, note that [auto_fp8/modeling.py](https://github.com/neuralmagic/AutoFP8/blob/main/auto_fp8/modeling.py) had to be adjusted, with line 152 ```if re.search(regex_pattern, name):``` replaced with ```if re.search(regex_pattern, name) and re.search(regex_pattern + "_proj", name) is None:```. This way, the ```gate_proj``` layers will not be left unquantized.
Although AutoFP8 was used for this particular model, Neural Magic is transitioning to using [llm-compressor](https://github.com/vllm-project/llm-compressor) which supports several quantization schemes and models not supported by AutoFP8.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "Qwen/Qwen2-57B-A14B-Instruct"
quantized_model_dir = "Qwen2-57B-A14B-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True, model_max_length=4096)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(512))
examples = [tokenizer.apply_chat_template(batch["messages"], tokenize=False) for batch in ds]
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to("cuda")

quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="static"
    ignore_patterns=["re:.*lm_head", "re:.*gate"],
)

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
```

## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Qwen2-57B-A14B-Instruct-FP8",dtype=auto,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Qwen2-57B-A14B-Instruct</strong>
   </td>
   <td><strong>Qwen2-57B-A14B-Instruct-FP8(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>75.76
   </td>
   <td>75.49
   </td>
   <td>99.64%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>66.89
   </td>
   <td>65.96
   </td>
   <td>98.60%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>80.59
   </td>
   <td>77.10
   </td>
   <td>95.66%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>85.96
   </td>
   <td>85.71
   </td>
   <td>99.70%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>78.45
   </td>
   <td>78.14
   </td>
   <td>99.60%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>62.11
   </td>
   <td>61.80
   </td>
   <td>99.50%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>74.96</strong>
   </td>
   <td><strong>74.03</strong>
   </td>
   <td><strong>98.76%</strong>
   </td>
  </tr>
</table>