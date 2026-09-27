---
language:
- en
base_model:
- deepreinforce-ai/Ornith-1.0-35B
pipeline_tag: text-generation
tags:
- qwen3_5
- moe
- fp8
- vllm
- conversational
- text-generation-inference
license: mit
---

## Model Overview
- **Model Architecture:** Qwen3_5MoeForConditionalGeneration
  - **Input:** Text (+ image/video inputs supported by the base architecture)
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP8
  - **Weight quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use. Similarly to the base model, this quantized version is intended for agentic coding and general assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights and activations of [deepreinforce-ai/Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within the language-model transformer blocks are quantized.
Weights are quantized with 128x128 block scaling (FP8_BLOCK), and activations are quantized dynamically with the matching per-group scheme, which gives the best throughput on NVIDIA Hopper and Blackwell GPUs.
The router/gate projections, shared-expert gates, all normalization layers, the token embeddings, the vision tower, and the short causal-conv1d projections used by the linear-attention (gated deltanet) layers are kept at full precision, since they are either not compute-bound `Linear` layers or are highly sensitive to quantization error.
Additionally, the linear-attention gating projections `in_proj_a`/`in_proj_b` (shape `[32, hidden_size]`) are kept at full precision because their 32-row output dimension is not divisible by the FP8_BLOCK 128x128 block size.
This model was quantized with the `model_free_ptq` pathway of [llm-compressor](https://github.com/vllm-project/llm-compressor), which applies the recipe directly to the safetensors checkpoint without requiring a transformers model definition or a calibration dataset.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Ornith-1.0-35B-FP8-BLOCK"
number_gpus = 1
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)
outputs = llm.generate(prompts, sampling_params)
generated_text = outputs[0].outputs[0].text
print(generated_text)
```

## Creation

<details>
  <summary>Creation details</summary>

  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. This mirrors the `model_free_ptq` recipe used for [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B), the base architecture that Ornith-1.0-35B was post-trained from, with two extra ignore patterns for the non-128-divisible linear-attention gating projections.

  ```python
  from llmcompressor import model_free_ptq

  MODEL_ID = "deepreinforce-ai/Ornith-1.0-35B"
  SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"

  # Apply FP8-Block to the model
  # Once quantized, the model is saved
  # using compressed-tensors to the SAVE_DIR.
  model_free_ptq(
      model_stub=MODEL_ID,
      save_directory=SAVE_DIR,
      scheme="FP8_BLOCK",
      ignore=[
          "lm_head",
          "re:.*mlp.gate$",
          "re:.*mlp.shared_expert_gate.*",
          "re:.*norm.*",
          "re:.*embed_tokens.*",
          "re:.*visual.*",
          "re:.*conv1d.*",
          # in_proj_a/in_proj_b are [32, 2048] gating projections used by the
          # linear-attention (gated deltanet) layers; the 32-row output dim is
          # not divisible by the FP8_BLOCK 128x128 block size, so they must be
          # skipped (in_proj_qkv/in_proj_z/out_proj are all 128-divisible and
          # remain quantized).
          "re:.*linear_attn.in_proj_a.*",
          "re:.*linear_attn.in_proj_b.*",
      ],
      max_workers=15,
      device="cuda:0",
  )
  ```
</details>
