---
library_name: transformers
pipeline_tag: text-generation
tags:
- nvfp4
- fp4
- vllm
- llm-compressor
- compressed-tensors
base_model: Qwen/Qwen3.8-2.4T-A95B
---

# Qwen3.8-2.4T-A95B-NVFP4-REAP-50

## Model Overview
- **Model Architecture:** Qwen3_5MoeForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
  - **Expert pruning:** 50% REAP Pruning
- **Release Date:** 2026-08-19
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by applying mixed-precision quantization to [Qwen/Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) with 50% REAP Pruning of the experts: the MoE expert linear layers use FP4 (NVFP4) weights and activations, ready for inference with vLLM. Attention and non-quantized layers keep their original precision.

This optimization reduces the number of bits per parameter in the quantized MoE layers from 16 to 4, and applies 50% REAP Pruning to the experts, reducing the disk size and GPU memory requirements of those layers by approximately 75% plus the sparsity gain.

Only the weights and activations of the linear operators in the MoE experts are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```bash
vllm serve RedHatAI/Qwen3.8-2.4T-A95B-NVFP4-REAP-50 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --reasoning-parser qwen3
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with the NVFP4 scheme and 25% REAP Pruning of the experts, exported in compressed-tensors 