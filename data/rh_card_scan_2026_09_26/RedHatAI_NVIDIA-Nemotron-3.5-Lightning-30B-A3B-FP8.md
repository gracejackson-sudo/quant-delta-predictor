---
base_model:
- nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
- nemotron-3.5
- conversational
pipeline_tag: text-generation
library_name: transformers
---

# RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8

## Model Overview

- **Model Architecture:** NemotronHForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2026-08-13
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16).

## Model Optimizations

This model was obtained by quantizing the weights and activations of [nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16) to the FP8 data type, using a per-tensor static scheme. This reduces the number of bits per parameter from 16 to 8, cutting GPU memory and disk requirements by approximately 50% and roughly doubling matrix-multiply compute throughput on FP8-capable hardware. Only the weights and activations of the linear operators within the transformer blocks are quantized, using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from UltraChat, as presented in the code snippet below.

<details>

```python
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"

with load_context(AutoModelForCausalLM):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = GPTQModifier(
    targets="Linear",
    scheme="FP8",
    ignore=[
        r"re:.*conv1d.*",
        r"backbone\.embeddings",
        r"re:.*_latent_proj.*",
        r"re:.*mixer.gate\..*",
        r"re:mtp.layers.*",
        "backbone.norm_f",
        "lm_head",
    ],
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```
</details>

## Deployment

### vLLM Serving

```
vllm serve RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8 \
    --max-num-seqs 128 \
    --enable-prefix-caching \
    --async-scheduling \
    --mamba-backend flashinfer \
    --mamba-ssm-cache-dtype float16 \
    --enable-mamba-cache-stochastic-rounding \
    --mamba-cache-philox-rounds 5 \
    --moe-backend flashinfer_cutlass \
    --reasoning-parser nemotron_v3 \
    --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice
```