---
tags:
- fp8
- moe
- vllm
- llm-compressor
- compressed-tensors
library_name: transformers
license: apache-2.0
base_model: thinkingmachines/Inkling
pipeline_tag: image-text-to-text
---

# Inkling-FP8-block

## Model Overview
- **Model Architecture:** InklingForConditionalGeneration
  - **Input:** Text / Image / Audio
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2026-07-16
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling), a 975B total / 41B active parameter multimodal Mixture-of-Experts model that accepts text, image, and audio inputs and generates text outputs.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling) to FP8 data type using block-wise quantization, ready for inference with vLLM.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Weights are quantized statically using 128x128 block-wise FP8 scaling, and activations are quantized dynamically at inference time using per-group scaling (group size 128). Only the weights and activations of the linear (attention and MoE expert) layers within the language backbone are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor). The vision encoder, audio encoder, token embedding and unembedding layers, normalization layers, biases, MoE routing/gating logic, shared experts, and the model's early dense MLP layers are kept in their original precision.

## Deployment

### Use with vLLM

This model can be deployed using [vLLM](https://docs.vllm.ai/en/latest/).

> **Note:** FP8 quantization support for Inkling requires [vllm-project/vllm#48876](https://github.com/vllm-project/vllm/pull/48876).

1. Start the vLLM server:
```
vllm serve RedHatAI/Inkling-FP8-block \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}'
```

> **Tip:** For text-only workloads, pass `--limit-mm-per-prompt '{"image": 0, "audio": 0}'` to skip the vision/audio encoder memory allocation and free up GPU memory for a longer context window.

2. Send requests to the server:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/Inkling-FP8-block"

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```
