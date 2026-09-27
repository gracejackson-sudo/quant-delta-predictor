---
license: other
license_name: "kimi-k3"
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- nvfp4
- fp4
- vllm
- llm-compressor
- compressed-tensors
base_model: moonshotai/Kimi-K3
---

# Kimi-K3-NVFP4

## Model Overview
- **Model Architecture:** KimiK3ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Release Date:** 2026-07-30
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by quantizing the MoE expert weights and activations of [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) to FP4 (NVFP4) data type, ready for inference with vLLM. Attention, shared experts, and other non-quantized layers keep their original precision.

This optimization reduces the number of bits per parameter in the quantized MoE layers from 16 to 4, reducing the disk size and GPU memory requirements of those layers by approximately 75%.

Only the weights and activations of the linear operators in the MoE experts are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```bash
vllm serve RedHatAI/Kimi-K3-NVFP4 \
  --tensor-parallel-size 8 \
  --trust_remote_code \
  --load-format instanttensor \
  --reasoning-parser kimi_k3 \
  --language-model-only  # optional
```

This model requires [vllm-project/vllm#50500](https://github.com/vllm-project/vllm/pull/50500).

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with the NVFP4 scheme, exported in compressed-tensors format.

## Evaluation

This model was evaluated on GPQA Diamond, served with vLLM (OpenAI-compatible API). Recovery is computed against the unquantized model.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>moonshotai/Kimi-K3</th>
      <th>RedHatAI/Kimi-K3-NVFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Reasoning</b></td>
      <td>GPQA Diamond</td>
      <td>93.5</td>
      <td>91.0</td>
      <td>97.33%</td>
    </tr>
  </tbody>
</table>
