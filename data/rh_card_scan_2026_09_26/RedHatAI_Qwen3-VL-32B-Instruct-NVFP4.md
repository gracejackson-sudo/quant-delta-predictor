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

# Qwen3-VL-32B-Instruct-NVFP4

## Model Overview
- **Model Architecture:** Qwen3VLForConditionalGeneration
  - **Input:** Text, Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 



## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Qwen3-VL-32B-Instruct-NVFP4 --tensor_parallel_size 2
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

model = "RedHatAI/Qwen3-VL-32B-Instruct-NVFP4"

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
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"

# Load model.
model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_function(example):
    messgages = []
    for message in example["messages"]:
        messgages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messgages,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {
        key: (
            torch.tensor(value)
            if key != "pixel_values"
            else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        )
        for key, value in batch[0].items()
    }


# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with group-wise quantization
#   * quantize the activations to fp4 with dynamic group activations
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
    ],
)

# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    dataset=ds,
    data_collator=data_collator,
)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
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
    --model_args pretrained="RedHatAI/Qwen3-VL-32B-Instruct-NVFP4",dtype=auto,add_bos_token=False,max_model_len=262144,tensor_parallel_size=2,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True,max_images=10 \
    --tasks chartqa \
    --apply_chat_template \
    --batch_size auto
  ```


  **MMLU**  
  ```
  lm_eval \
    --model vllm-vlm \
    --model_args pretrained="RedHatAI/Qwen3-VL-32B-Instruct-NVFP4",dtype=auto,add_bos_token=False,max_model_len=262144,tensor_parallel_size=2,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True,max_images=10 \
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
| RedHatAI/Qwen3-VL-32B-Instruct-NVFP4 | 86.56 | 140.7 |

## MMLU Results

| Model | Accuracy | Recovery (%) |
|-------|----------|--------------|
| Qwen/Qwen3-VL-32B-Instruct | 78.03 | 100.00 |
| Qwen/Qwen3-VL-32B-Instruct-FP8 | 77.80 | 99.71 |
| RedHatAI/Qwen3-VL-32B-Instruct-FP8-block | 77.72 | 99.60 |
| RedHatAI/Qwen3-VL-32B-Instruct-FP8-dynamic | 77.89 | 99.82 |
| RedHatAI/Qwen3-VL-32B-Instruct-NVFP4 | 76.27 | 97.74 |