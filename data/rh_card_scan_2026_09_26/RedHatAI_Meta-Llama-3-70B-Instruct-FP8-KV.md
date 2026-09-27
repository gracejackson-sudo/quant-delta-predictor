---
tags:
- fp8
- vllm
---

# Meta-Llama-3-70B-Instruct-FP8-KV

## Model Overview
Meta-Llama-3-70B-Instruct quantized to FP8 weights and activations using per-tensor quantization, ready for inference with vLLM >= 0.5.0.
This model checkpoint also includes per-tensor scales for FP8 quantized KV Cache, accessed through the `--kv-cache-dtype fp8` argument in vLLM.

```python
from vllm import LLM
model = LLM(model="neuralmagic/Meta-Llama-3-70B-Instruct-FP8-KV", kv_cache_dtype="fp8")
result = model.generate("Hello, my name is")
```

## Usage and Creation
Produced using [AutoFP8 with calibration samples from ultrachat](https://github.com/neuralmagic/AutoFP8).

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "meta-llama/Meta-Llama-3-70B-Instruct"
quantized_model_dir = "Meta-Llama-3-70B-Instruct-FP8-KV"

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

Model evaluation results obtained via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

|  Benchmark                                                | Meta-Llama-3-70B-Instruct | Meta-Llama-3-70B-Instruct-FP8 | Meta-Llama-3-70B-Instruct-FP8-KV<br>(this model) |
| :-------------------------------------------------------: | :-----------------------: | :---------------------------: | :----------------------------------------------: | 
| [ARC-c](https://arxiv.org/abs/1911.01547)<br> 25-shot     | 72.69                     | 72.61                         | 72.57                                            |
| [HellaSwag](https://arxiv.org/abs/1905.07830)<br> 10-shot | 85.50                     | 85.41                         | 85.32                                            |
| [MMLU](https://arxiv.org/abs/2009.03300)<br> 5-shot       | 80.18                     | 80.06                         | 79.69                                            |
| [TruthfulQA](https://arxiv.org/abs/2109.07958)<br> 0-shot | 62.90                     | 62.73                         | 61.92                                            |
| [WinoGrande](https://arxiv.org/abs/1907.10641)<br> 5-shot | 83.34                     | 83.03                         | 83.66                                            |
| [GSM8K](https://arxiv.org/abs/2110.14168)<br> 5-shot      | 92.49                     | 91.12                         | 90.83                                            |
| **Average<br>Accuracy**                                   | **79.51**                 | **79.16**                     | **79.00**                                        |
| **Recovery**                                              | **100%**                  | **99.55%**                    | **99.36%**                                       |
