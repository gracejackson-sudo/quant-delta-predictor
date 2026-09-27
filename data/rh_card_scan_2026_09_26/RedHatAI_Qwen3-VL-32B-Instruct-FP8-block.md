---
license: apache-2.0
pipeline_tag: text-generation
tags:
- fp8
- quantized
- llm-compressor
- compressed-tensors
- red hat
base_model:
- Qwen/Qwen3-VL-32B-Instruct
---



# Qwen3-VL-32B-Instruct-FP8-block

## Model Overview
- **Model Architecture:** Qwen3VLForConditionalGeneration
  - **Input:** Text, Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 



## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Qwen3-VL-32B-Instruct-FP8-block --tensor_parallel_size 2
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

model = "RedHatAI/Qwen3-VL-32B-Instruct-FP8-block"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```





## Creation

This model was quantized using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as shown below.

<details>
  <summary>Creation details</summary>

```python
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: Requires a minimum of transformers 4.57.0

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

# Load model.
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with channel-wise quantization
#   * quantize the activations to fp8 with dynamic token activations
# NOTE: only datafree quantization is supported for Qwen3-VL-MoE currently
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-BLOCK"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```
</details>


## Evaluation


The model was evaluated on the OpenLLMv1 leaderboard task, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>
  
  **ChartQA**
  ```
  lm_eval \
    --model vllm-vlm \
    --model_args pretrained="RedHatAI/Qwen3-VL-32B-Instruct-FP8-block",dtype=auto,add_bos_token=False,max_model_len=262144,tensor_parallel_size=2,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True,max_images=10 \
    --tasks chartqa \
    --apply_chat_template \
    --batch_size auto
  ```


  **MMLU**  
  ```
  lm_eval \
    --model vllm-vlm \
    --model_args pretrained="RedHatAI/Qwen3-VL-32B-Instruct-FP8-block",dtype=auto,add_bos_token=False,max_model_len=262144,tensor_parallel_size=2,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True,max_images=10 \
    --tasks mmlu \
    --apply_chat_template \
    --batch_size auto
  ```
</details>


# Accuracy Comparison

## ChartQA Results

| Model | Accuracy | Recovery (%) |
|-------|----------|--------------|
| Qwen/Qwen3-VL-32B-Instruct | 61.52 | 100.00 |
| Qwen/Qwen3-VL-32B-Instruct-FP8 | 86.92 | 141.32 |
| RedHatAI/Qwen3-VL-32B-Instruct-FP8-block | 86.60 | 140.82 |
| RedHatAI/Qwen3-VL-32B-Instruct-FP8-dynamic | 86.68 | 140.95 |

## MMLU Results

| Model | Accuracy | Recovery (%) |
|-------|----------|--------------|
| Qwen/Qwen3-VL-32B-Instruct | 78.03 | 100.00 |
| Qwen/Qwen3-VL-32B-Instruct-FP8 | 77.80 | 99.71 |
| RedHatAI/Qwen3-VL-32B-Instruct-FP8-block | 77.72 | 99.60 |
| RedHatAI/Qwen3-VL-32B-Instruct-FP8-dynamic | 77.89 | 99.82 |