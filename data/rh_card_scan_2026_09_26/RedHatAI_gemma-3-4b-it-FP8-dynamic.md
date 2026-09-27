---
tags:
- vllm
- vision
- fp8
license: apache-2.0
license_link: >-
  https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
- en
base_model: google/gemma-3-4b-it
library_name: transformers
---

# gemma-3-4b-it-FP8-Dynamic

## Model Overview
- **Model Architecture:** gemma-3-4b-it
  - **Input:** Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2/24/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it).

### Model Optimizations

This model was obtained by quantizing the weights of [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) to FP8 data type, ready for inference with vLLM >= 0.5.2.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from transformers import AutoProcessor

# Define model name once
model_name = "RedHatAI/gemma-3-4b-it-FP8-dynamic"

# Load image and processor
image = ImageAsset("cherry_blossom").pil_image.convert("RGB")
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

# Build multimodal prompt
chat = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is the content of this image?"}]},
    {"role": "assistant", "content": []}
]
prompt = processor.apply_chat_template(chat, add_generation_prompt=True)

# Initialize model
llm = LLM(model=model_name, trust_remote_code=True)

# Run inference
inputs = {"prompt": prompt, "multi_modal_data": {"image": [image]}}
outputs = llm.generate(inputs, SamplingParams(temperature=0.2, max_tokens=64))

# Display result
print("RESPONSE:", outputs[0].outputs[0].text)

```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below as part a multimodal announcement blog.

<details>
  <summary>Model Creation Code</summary>
  
```python
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
model_id = google/gemma-3-4b-it
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Recipe
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        sequential_targets=["Gemma3DecoderLayer"],
        ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
    ),
]

SAVE_DIR=f"{model_id.split('/')[1]}-FP8-Dynamic"

# Perform oneshot
oneshot(
    model=model,
    recipe=recipe,
    trust_remote_code_model=True,
    output_dir=SAVE_DIR
)


```
</details>

## Evaluation

The model was evaluated using [lm_evaluation_harness](https://github.com/neuralmagic/lm-evaluation-harness) for OpenLLM v1 text benchmark. The evaluations were conducted using the following commands:

<details>
<summary>Evaluation Commands</summary>

### OpenLLM v1
```
lm_eval \
  --model vllm \
  --model_args pretrained="<model_name>",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=<n>,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True,enforce_eager=True \
  --tasks openllm \
  --batch_size auto
```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>google/gemma-3-4b-it</th>
      <th>RedHatAI/gemma-3-4b-it-FP8-Dynamic</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC Challenge</td>
      <td>56.57%</td>
      <td>57.08%</td>
      <td>100.90%</td>
    </tr>
    <tr>
      <td>GSM8K</td>
      <td>76.12%</td>
      <td>75.51%</td>
      <td>99.20%</td>
    </tr>
    <tr>
      <td>Hellaswag</td>
      <td>74.96%</td>
      <td>74.92%</td>
      <td>99.95%</td>
    </tr>
    <tr>
      <td>MMLU</td>
      <td>58.38%</td>
      <td>57.98%</td>
      <td>99.32%</td>
    </tr>
    <tr>
      <td>Truthfulqa (mc2)</td>
      <td>51.87%</td>
      <td>51.62%</td>
      <td>99.52%</td>
    </tr>
    <tr>
      <td>Winogrande</td>
      <td>70.32%</td>
      <td>71.03%</td>
      <td>101.01%%%%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>64.70%</b></td>
      <td><b>64.69%</b></td>
      <td><b>99.98%</b></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Vision Evals</b></td>
      <td>MMMU (val)</td>
      <td>39.89%/td>	
      <td>38.33%</td>
      <td>96.09%</td>
    </tr>
    <tr>
      <td>ChartQA</td>
      <td>50.76%</td>
      <td>51.60%</td>
      <td>101.65%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>45.33%</b></td>
      <td><b>44.97%</b></td>
      <td><b>98.87%</b></td>
    </tr>
  </tbody>
</table>

