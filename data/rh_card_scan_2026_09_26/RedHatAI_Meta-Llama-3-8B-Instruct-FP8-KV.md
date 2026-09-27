---
tags:
- fp8
- vllm
base_model:
- meta-llama/Meta-Llama-3-8B-Instruct
---

# Meta-Llama-3-8B-Instruct-FP8-KV

## Model Overview
Meta-Llama-3-8B-Instruct quantized to FP8 weights and activations using per-tensor quantization, ready for inference with vLLM >= 0.5.0.
This model checkpoint also includes per-tensor scales for FP8 quantized KV Cache, accessed through the `--kv-cache-dtype fp8` argument in vLLM.

```python
from vllm import LLM
model = LLM(model="neuralmagic/Meta-Llama-3-8B-Instruct-FP8-KV", kv_cache_dtype="fp8")
result = model.generate("Hello, my name is")
```

## Usage and Creation
Produced using [AutoFP8 with calibration samples from ultrachat](https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py).

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "meta-llama/Meta-Llama-3-8B-Instruct"
quantized_model_dir = "Meta-Llama-3-8B-Instruct-FP8-KV"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("mgoin/ultrachat_2k", split="train_sft")
examples = [tokenizer.apply_chat_template(batch["messages"], tokenize=False) for batch in ds]
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to("cuda")

quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="static",
    ignore_patterns=["re:.*lm_head"],
    kv_cache_quant_targets=("k_proj", "v_proj"),
)

model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
```

## Evaluation

### Open LLM Leaderboard evaluation scores
|                      | Meta-Llama-3-8B-Instruct | Meta-Llama-3-8B-Instruct-FP8 | Meta-Llama-3-8B-Instruct-FP8-KV<br>(this model) |
| :------------------: | :----------------------: | :--------------------------: | :---------------------------------------------: |
| gsm8k<br>5-shot      | 75.44                    | 74.37                        | 74.98                                           |