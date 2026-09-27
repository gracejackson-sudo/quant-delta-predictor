---
language:
- en
- zh
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- int4
- vllm
- llm-compressor
- compressed-tensors
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
---

# Qwen3.8-27B-INT4

## Model Overview
- **Model Architecture:** Qwen3_5ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** None
- **Release Date:** 2026-08-17
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) to INT4 data type while keeping activations in original precision, with FP8 KV cache, ready for inference with vLLM.

This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```bash

vllm serve RedHatAI/Qwen3.8-27B-INT4 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

For optimal peformance, consider using the DSpark draft model [RedHatAI/Qwen3.8-27B-speculator.dspark](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) for speculative decoding, shown below.

```bash
vllm serve RedHatAI/Qwen3.8-27B-INT4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"model":"RedHatAI/Qwen3.8-27B-speculator.dspark","num_speculative_tokens":8,"method":"dspark"}'
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from open-perfectblend, using AWQ smoothing and the W4A16 GPTQ scheme, exported in compressed-tensors format.

<details>

```python
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.8-27B"

# Load model.
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)


recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        targets="Linear",
        scheme="W4A16",
        ignore=[
            "re:visual.*",
            "re:model.visual.*",
            r"re:.*lm_head",
            "re:.*embed_tokens$",
            r"re:.*linear_attn\.in_proj_a$",
            r"re:.*linear_attn\.in_proj_b$",
        ],
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "strategy": "tensor",
            "dynamic": False,
            "observer": "static_minmax",
        },
    ),
]

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(
    "mlabonne/open-perfectblend",
    split=f"train[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.shuffle(seed=42)

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def preprocess_function(example):
    messages = [
        {
            "role": ROLE_MAP.get(msg["from"], msg["from"]),
            "content": [{"type": "text", "text": msg["value"]}],
        }
        for msg in example["conversations"]
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        processor_kwargs={
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": MAX_SEQUENCE_LENGTH,
            "add_special_tokens": False,
        },
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
    data_collator=data_collator,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-INT4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

</details>

## Evaluation

This model was evaluated on GSM8K Platinum, IFEval, MMLU-Pro, MATH-500, GPQA Diamond, and AIME 2025 using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness) and [lighteval](https://github.com/neuralmagic/lighteval), all served with vLLM (OpenAI-compatible API). Each benchmark was run with 3 seeds (1234, 2345, 3456; 8 seeds for AIME 2025) and the results averaged; recovery is computed against the BF16 model.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Qwen/Qwen3.8-27B (BF16)</th>
      <th>RedHatAI/Qwen3.8-27B-INT4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Instruction Following</b></td>
      <td>IFEval (0-shot, prompt-level strict)</td>
      <td>92.24%</td>
      <td>91.93%</td>
      <td>99.67%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (exact-match)</td>
      <td>84.46%</td>
      <td>83.45%</td>
      <td>98.81%</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Reasoning</b></td>
      <td>GSM8K Platinum (strict-match)</td>
      <td>95.75%</td>
      <td>96.77%</td>
      <td>101.07%</td>
    </tr>
    <tr>
      <td>MATH-500 (pass@1)</td>
      <td>83.73%</td>
      <td>83.33%</td>
      <td>99.52%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (pass@1)</td>
      <td>89.23%</td>
      <td>87.88%</td>
      <td>98.49%</td>
    </tr>
    <tr>
      <td>AIME 2025 (pass@1)</td>
      <td>95.42%</td>
      <td>94.17%</td>
      <td>98.69%</td>
    </tr>
  </tbody>
</table>