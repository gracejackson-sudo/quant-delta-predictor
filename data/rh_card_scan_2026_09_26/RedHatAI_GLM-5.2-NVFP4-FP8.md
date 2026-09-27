---
tags:
- fp8
- fp4
- vllm
- llm-compressor
- compressed-tensors
language:
- en
- zh
library_name: transformers
license: mit
pipeline_tag: text-generation
base_model: zai-org/GLM-5.2
---

# RedHatAI/GLM-5.2-NVFP4-FP8

## Model Overview
- **Model Architecture:** GlmMoeDsaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** Mixed (FP8 attention, FP4 MoE)
  - **Activation quantization:** Mixed (FP8 attention, FP4 MoE)
- **Release Date:** 2026-06-29
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2), with the MoE experts in FP4 (NVFP4) and the attention weights in FP8, and was evaluated against the unquantized model.

## Model Optimizations

This model was obtained by applying mixed-precision quantization to [zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2): the MoE expert weights and activations are quantized to FP4 (NVFP4), while the attention weights and activations are quantized to FP8 (block-scaled), ready for inference with vLLM.

This optimization reduces the number of bits per parameter from 16 to an average of roughly 4–5, cutting disk size and GPU memory requirements by approximately 70–75% compared to the BF16 original.

The weights and activations of the linear operators within the attention and MoE blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

This model is intended for deployment with vLLM and requires the following fix: https://github.com/vllm-project/vllm/pull/47780.

```bash
vllm serve RedHatAI/GLM-5.2-NVFP4-FP8 \
    --tensor-parallel-size 4 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --enable-auto-tool-choice \
    --kv-cache-dtype fp8
```

For additional serving options (e.g. MTP speculative decoding, larger tensor parallelism, or 1M-context configuration), refer to the [vLLM recipe for GLM-5.2](https://recipes.vllm.ai/zai-org/GLM-5.2).

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from UltraChat (`HuggingFaceH4/ultrachat_200k`), as presented in the code snippet below.

<details>

```python
import torch
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Load the model
init_dist()
model_id = "zai-org/GLM-5.2"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        max_memory={},
        offload_folder="./offload",
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
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


# Configure the quantization algorithm to run.
recipe = QuantizationModifier(
    config_groups={
        "attention_shared_experts": QuantizationScheme(
            targets=[r"re:.*self_attn\..*"],
            **FP8_BLOCK,
        ),
        "mlp": QuantizationScheme(
            targets=[r"re:.*mlp\..*"],
            **NVFP4,
        ),
    },
    ignore=[
        r"re:^model\.layers\.[0-2]\..*"
        r"re:.*mlp\.gate.*",  # not technically necessary
        r"re:.*indexer\.weights_proj$",  # sensitive to quantization
        r"lm_head",
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    batch_size=4,
    recipe=recipe,
    shuffle_calibration_samples=False,
)

# Save to disk compressed.
# Note: base checkpoint generation_config needs fixing for newer transformers versions
model.generation_config.top_p = None
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
```
</details>

## Evaluation

This model was evaluated on GPQA Diamond using lighteval, served with vLLM (OpenAI-compatible API).

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>zai-org/GLM-5.2</th>
      <th>RedHatAI/GLM-5.2-NVFP4-FP8</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Reasoning</b></td>
      <td>GPQA Diamond (0-shot, pass@1)</td>
      <td>91.2</td>
      <td>89.1</td>
      <td>97.7%</td>
    </tr>
  </tbody>
</table>
