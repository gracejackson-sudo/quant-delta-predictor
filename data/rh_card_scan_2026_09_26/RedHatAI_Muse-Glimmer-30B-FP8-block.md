---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
base_model: meta-models/Muse-Glimmer-30B
license: apache-2.0
---

# Muse-Glimmer-30B-FP8-block

## Model Overview
- **Model Architecture:** MuseGlimmerForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2026-08-10
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) to FP8 (W8A8 block) data type, ready for inference with vLLM.

This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```
docker run --gpus all \
    --privileged --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:muse-glimmer RedHatAI/Muse-Glimmer-30B-FP8-block \
    --generation-config auto \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser muse_glimmer \
    --reasoning-parser muse_glimmer
```

For detailed instructions including multi-GPU deployment, multimodal inference, etc see the [Muse-Glimmer 30B vLLM usage guide](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B).

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with the FP8_BLOCK scheme in model-free post-training quantization (no calibration data required), exported in compressed-tensors format.

<details>

```python
from llmcompressor import model_free_ptq

MODEL_ID = "meta-models/Muse-Glimmer-30B"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-block"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"],
    max_workers=15,
    device="cuda:0",
)
```

</details>