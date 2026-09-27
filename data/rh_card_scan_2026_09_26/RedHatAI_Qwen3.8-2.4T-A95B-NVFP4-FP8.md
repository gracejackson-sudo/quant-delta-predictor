---
library_name: transformers
pipeline_tag: text-generation
tags:
- nvfp4
- fp4
- fp8
- vllm
- llm-compressor
- compressed-tensors
base_model: Qwen/Qwen3.8-2.4T-A95B
---

# Qwen3.8-2.4T-A95B-NVFP4-FP8

## Model Overview
- **Model Architecture:** Qwen3_5MoeForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** Mixed (FP4 MoE experts, FP8 attention)
  - **Activation quantization:** Mixed (FP4 / FP8)
- **Release Date:** 2026-08-14
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by applying mixed-precision quantization to [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B): the MoE expert linear layers use FP4 (NVFP4) weights and activations, while the attention linear layers use FP8 (W8A8 block) weights and activations, ready for inference with vLLM.

This optimization reduces the number of bits per parameter in the quantized layers from 16 to 4 or 8 depending on the layer, reducing the disk size and GPU memory requirements by approximately 50-75%.

Only the weights and activations of the linear operators in the MoE experts and attention blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```bash
vllm serve RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8 \
    --data-parallel-size 8 \
    --enable-expert-parallel 8 \
    --reasoning-parser qwen3 \
    --max-num-seqs 140
```

NOTE: Because of the need for data parallelism, this model may use more model memory for replicated attention layers. This can lead to less memory for kv cache and cache preemption when handling large concurrency. For high concurrency tasks, consider using `RedHatAI/Qwen3.8-2.4T-A95B-NVFP4`.

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with the NVFP4 (MoE) + FP8 block (attention) mixed-precision scheme, exported in compressed-tensors format. 

```python
# NOTE: to use a custom dataset, see examples/custom_dataset_example.py
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    FP8_BLOCK,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Select model and load it.
init_dist()
model_id = "inference-optimization/Qwen3.8-1.0B-A0.6B"  # Qwen/Qwen3.8-2.4T-A95B
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        max_memory={},
        offload_folder="offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Create recipe
recipe = [
    QuantizationModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\..*",
                    r"re:.*linear_attn.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)$",
                ],
                **FP8_BLOCK,
            ),
            "mlp": QuantizationScheme(
                targets=[r"re:.*mlp\..*"],
                **NVFP4,
            ),
        },
        ignore=[
            "re:.*lm_head",
            "re:.*mlp.gate$",
            "re:.*shared_expert_gate.*",
        ],
    ),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset="perfectblend",
    splits="train[:512]",
    recipe=recipe,
    max_seq_length=2048,
    num_calibration_samples=1024,
    pipeline="sequential",
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4-FP8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Evaluation

This model was evaluated on GPQA Diamond, served with vLLM (OpenAI-compatible API). Recovery is computed against the unquantized model.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Qwen/Qwen3.8-2.4T-A95B</th>
      <th>RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-FP8</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Reasoning</b></td>
      <td>GPQA Diamond</td>
      <td>92.6</td>
      <td>93.1</td>
      <td>100.54%</td>
    </tr>
  </tbody>
</table>
