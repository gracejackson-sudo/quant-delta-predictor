---
tags:
- fp8
- vllm
pipeline_tag: text-generation
provider: Sarvam AI
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/sarvam-30b-FP8-dynamic
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
readme: https://huggingface.co/RedHatAI/sarvam-30b-FP8-dynamic/blob/main/README.md
description: Sarvam-30B is an advanced Mixture-of-Experts (MoE) model with 2.4B non-embedding active parameters, designed primarily for practical deployment. It combines strong reasoning, reliable coding ability, and best-in-class conversational quality across Indian languages. Sarvam-30B is built to run reliably in resource-constrained environments and can handle multilingual voice calls while performing tool calls.
base_model: 
- sarvamai/sarvam-30b
validated_on:
  - RHOAI 3.4 
  - RHAIIS 3.4
  - vLLM 0.18.0
language:
  - en
  - hi
  - bn
  - ta
  - te
  - mr
  - gu
  - kn
  - ml
  - pa
  - or
  - as
  - ur
  - sa
  - ne
  - sd
  - kok
  - mai
  - doi
  - mni
  - sat
  - ks
  - bo
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  sarvam-30b-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** sarvamai/sarvam-30b
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Version:** 1.0
- **Model Developers:** RedHatAI
- **ModelCar Storage URI:** oci://registry.redhat.io/rhai/modelcar-sarvam-30b-fp8-dynamic:3.0
- **Validated on vLLM:** 0.18.0
- **Validated on RHAIIS:** 3.4
- **Validated on RHOAI:** 3.4


This model is a quantized version of [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [sarvamai/sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) to FP8 data type, ready for inference with vLLM.

Only the weights and activations of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend.


1. Install vLLM from main:
```
uv pip install -U git+https://github.com/vllm-project/vllm.git \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --no-deps \
  --no-cache
```

2. Run using vLLM
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/sarvam-30b-FP8-dynamic"
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

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor), as presented in the code snippet below.


<details>
  <summary>Creation details</summary>

Install specific llm-compression version:
```
uv pip install git+https://github.com/vllm-project/llm-compressor.git
uv pip install --upgrade torchvision --break-system-packages --no-cache
```

```python
from compressed_tensors.offload import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "sarvamai/sarvam-30b"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```

</details>

## Evaluation

This model was evaluated on the well-known text benchmarks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). 

```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/sarvam-30b-FP8-Dynamic",dtype=auto,add_bos_token=True,max_model_len=16384,tensor_parallel_size=2,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --show_config
  ```



### Accuracy


| Benchmark | sarvamai/sarvam-30b | RedHatAI/sarvam-30b-FP8-Dynamic | Recovery (%) |
|---|---|---|---|
| BBH (exact_match) | 63.32 | 62.95 | 99.42% |
| GSM8K (strict-match) | 72.33 | 72.40 | 100.10% |
| GSM8K (flexible-extract) | 69.67 | 70.81 | 101.63% |
| IFEval (inst_level_strict_acc) | 34.17 | 31.65 | 92.63% |
| MMLU-Pro (exact_match) | 45.69 | 45.81 | 100.25% |
| ARC-Challenge (acc) | 58.28 | 57.76 | 99.12% |
| HellaSwag (acc) | 53.98 | 53.98 | 100.00% |
| MMLU (acc) | 66.20 | 66.15 | 99.92% |
| TruthfulQA MC2 (acc) | 50.34 | 50.58 | 100.48% |
| Winogrande (acc) | 61.09 | 61.17 | 100.13% |
