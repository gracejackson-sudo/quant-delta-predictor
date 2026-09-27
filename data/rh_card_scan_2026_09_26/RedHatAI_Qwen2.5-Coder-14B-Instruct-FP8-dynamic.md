---
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/blob/main/LICENSE
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- code
- codeqwen
- chat
- qwen
- qwen-coder
- fp8
- llm-compressor
- compressed-tensors
- vllm
base_model:
- Qwen/Qwen2.5-Coder-14B-Instruct
---
## Model Overview
- **Model Architecture:** Qwen2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 11/28/2024
- **Version:** 1.0
- **Model Developers:** Red Hat

Quantized version of [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen2.5-Coder-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Qwen2.5-Coder-14B-Instruct-FP8-dynamic
```

2. Send requests to the server:

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/Qwen2.5-Coder-14B-Instruct-FP8-dynamic"

messages = [
    {"role": "user", "content": "Write a quick sort algorithm."},
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```

## Creation

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 

<details>
  <summary>Model Creation Code</summary>

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_stub = "Qwen/Qwen2.5-Coder-14B-Instruct"
model_name = model_stub.split("/")[-1]

model = AutoModelForCausalLM.from_pretrained(model_stub, dtype="auto")

tokenizer = AutoTokenizer.from_pretrained(model_stub)

# Configure the quantization algorithm and scheme
recipe = QuantizationModifier(
    ignore=["lm_head"],
    targets="Linear",
    scheme="FP8_dynamic",
)

# Apply quantization
oneshot(
    model=model,
    recipe=recipe,
)

# Save to disk in compressed-tensors format
save_path = model_name + "-FP8-dynamic"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")
```
</details>

