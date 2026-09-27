---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- fp4
- nvfp4
- vllm
- llm-compressor
- compressed-tensors
base_model: Qwen/Qwen3.8-Flash-Next
---

# RedHatAI/Qwen3.8-Flash-Next-NVFP4

## Model Overview
- **Model Architecture:** Qwen4ExpForConditionalGeneration
  - **Input:** Text / Image / Video
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Release Date:** 2026-08-27
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). It was evaluated to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of the Mixture-of-Experts (MoE) experts in [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) to NVFP4 (FP4) data type, ready for inference with vLLM.

This reduces the per-weight precision of the MoE expert parameters from 16 to 4 bits, substantially reducing their memory and disk footprint, while the rest of the model is kept in its original BF16 precision.

Only the weights and activations of the MoE expert linear operators are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```
vllm serve RedHatAI/Qwen3.8-Flash-Next-NVFP4 \
  --tensor-parallel-size 4 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

Adjust the tensor-parallel size and other hardware-specific settings to your deployment — see the [vLLM recipe for Qwen3.8-Flash-Next](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from [open-perfectblend](https://huggingface.co/datasets/mlabonne/open-perfectblend) (1024 samples).

## Evaluation

This model was evaluated on GPQA Diamond, a scientific reasoning benchmark, served with vLLM (OpenAI-compatible API) and compared against the unquantized [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) baseline.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Qwen/Qwen3.8-Flash-Next</th>
      <th>RedHatAI/Qwen3.8-Flash-Next-NVFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Reasoning</b></td>
      <td>GPQA Diamond (0-shot)</td>
      <td>91.7</td>
      <td>90.9</td>
      <td>99.13%</td>
    </tr>
  </tbody>
</table>