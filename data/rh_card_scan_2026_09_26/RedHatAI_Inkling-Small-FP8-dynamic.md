---
language:
- en
base_model:
- thinkingmachines/Inkling-Small
pipeline_tag: image-text-to-text
tags:
- fp8
- vllm
- conversational
- image-text-to-text
- audio-text-to-text
- moe
- text-generation-inference
license: apache-2.0
license_link: https://www.apache.org/licenses/LICENSE-2.0
---

## Model Overview
- **Model Architecture:** InklingForConditionalGeneration
  - **Input:** Text, Image, Audio
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP8
  - **Weight quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use. Similarly to the base model, this quantized version is intended for assistant-like chat, multimodal understanding, and coding tasks.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing activations and weights of [thinkingmachines/Inkling-Small](https://huggingface.co/thinkingmachines/Inkling-Small) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Inkling-Small-FP8-dynamic"
number_gpus = 4
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

  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below.

  ```python
  from llmcompressor import model_free_ptq

  MODEL_ID = "thinkingmachines/Inkling-Small"
  SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-dynamic"

  model_free_ptq(
      model_stub=MODEL_ID,
      save_directory=SAVE_DIR,
      scheme="FP8_DYNAMIC",
      ignore=[
          "model.llm.unembed",
          "model.llm.embed",
          "re:.*norm.*",
          "re:.*bias$",
          "re:.*\\.attn$",
          "re:.*\\.attn\\..*",
          "re:.*sconv$",
          "re:.*gate.*",
          "re:.*global_scale$",
          "re:model\\.visual\\..*",
          "re:model\\.audio\\..*",
      ],
      max_workers=2,
  )
  ```
</details>
