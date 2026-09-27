---
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
- conversational
license: other
license_name: "kimi-k3"
library_name: compressed-tensors
pipeline_tag: image-text-to-text
base_model:
- moonshotai/Kimi-K3
---

# Kimi-K3-FP8-BLOCK

## Model Overview
- **Model Architecture:** KimiK3ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2026-07-27
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3), using FP8 for both weights and activations, ready for inference with vLLM. See the [base model card](https://huggingface.co/moonshotai/Kimi-K3) for full model details and capabilities.

### Model Optimizations

This model was obtained by quantizing the weights and activations of the linear operators in [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) to FP8 — using block-wise scaling for the weights and dynamic group-wise scaling for the activations — producing an 8-bit (FP8) representation that is ready for inference with vLLM.

Only the weights and activations of the linear operators within the transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```
docker run --gpus all \
  --privileged --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e GLOO_SOCKET_IFNAME=$IFACE_NAME \
  -e NCCL_SOCKET_IFNAME=$IFACE_NAME \
  -e VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1 \
  vllm/vllm-openai:kimi-k3 RedHatAI/Kimi-K3-FP8-BLOCK \
  --trust-remote-code \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr $HEAD_IP \
  --no-enable-flashinfer-autotune \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with the `FP8_BLOCK` scheme (model-free post-training quantization, no calibration data required), as shown below.

<details>

```python
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer

from llmcompressor import model_free_ptq

MODEL_ID = "moonshotai/Kimi-K3"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

ignore = [
    "re:.*embed_tokens.*",
    "re:.*self_attn.*",  # no attention because (q_proj|k_proj|v_proj|b_proj|f_a_proj) are all fused, and `b_proj` has a weight non-divisible by 128
    "re:.*block_sparse_moe\.gate.*",
    "re:.*self_attention_res_proj.*",
    "re:.*mlp_res_proj.*",
    "re:.*output_attn_res_proj.*",
    "re:.*lm_head.*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=ignore,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        ignore=ignore,
    ),
    max_workers=7,
    device=[
        f"cuda:{i}"
        for i in range(7)
    ],
)
```

</details>

## License

Both the model weights and code are released under the [Kimi K3 License](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE).
