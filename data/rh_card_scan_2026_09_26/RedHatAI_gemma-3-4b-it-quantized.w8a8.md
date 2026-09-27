---
tags:
- vllm
- vision
- w8a8
license: gemma
base_model: google/gemma-3-4b-it
library_name: transformers
---

# gemma-3-4b-it-quantized.w8a8

## Model Overview
- **Model Architecture:** google/gemma-3-4b-it
  - **Input:** Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
  - **Activation quantization:** INT8
- **Release Date:** 6/4/2025
- **Version:** 1.0
- **Model Developers:** RedHatAI

Quantized version of [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it).

### Model Optimizations

This model was obtained by quantizing the weights of [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) to INT8 data type, ready for inference with vLLM >= 0.8.0.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from transformers import AutoProcessor

# Define model name once
model_name = "RedHatAI/gemma-3-4b-it-quantized.w8a8"

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

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below:

<details>
  <summary>Model Creation Code</summary>
  
```python
import base64
from io import BytesIO
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot


# Load model.
model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "neuralmagic/calibration"
DATASET_SPLIT = {"LLM": "train[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)

dampening_frac=0.05

def data_collator(batch):
    assert len(batch) == 1, "Only batch size of 1 is supported for calibration"
    item = batch[0]
    collated = {}
    import torch


    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            collated[key] = value.unsqueeze(0)
        elif isinstance(value, list) and isinstance(value[0][0], int):
            # Handle tokenized inputs like input_ids, attention_mask
            collated[key] = torch.tensor(value)
        elif isinstance(value, list) and isinstance(value[0][0], float):
            # Handle possible float sequences
            collated[key] = torch.tensor(value)
        elif isinstance(value, list) and isinstance(value[0][0], torch.Tensor):
            # Handle batched image data (e.g., pixel_values as [C, H, W])
            collated[key] = torch.stack(value)  # -> [1, C, H, W]
        elif isinstance(value, torch.Tensor):
            collated[key] = value
        else:
            print(f"[WARN] Unrecognized type in collator for key={key}, type={type(value)}")
    
    return collated
   


# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        ignore=["re:.*lm_head.*", "re:.*embed_tokens.*", "re:vision_tower.*", "re:multi_modal_projector.*"],
        sequential_update=True,
        sequential_targets=["Gemma3DecoderLayer"],
        dampening_frac=dampening_frac,
    )
]

SAVE_DIR=f"{model_id.split('/')[1]}-quantized.w8a8"

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
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
      <th>RedHatAI/gemma-3-4b-it-quantized.w8a8</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC Challenge</td>
      <td>56.57%</td>
      <td>56.31%</td>
      <td>99.55%</td>
    </tr>
    <tr>
      <td>GSM8K</td>
      <td>76.12%</td>
      <td>72.93%</td>
      <td>95.82%</td>
    </tr>
    <tr>
      <td>Hellaswag</td>
      <td>74.96%</td>
      <td>74.35%</td>
      <td>99.19%</td>
    </tr>
    <tr>
      <td>MMLU</td>
      <td>58.38%</td>
      <td>57.58%</td>
      <td>98.63%</td>
    </tr>
    <tr>
      <td>Truthfulqa (mc2)</td>
      <td>51.87%</td>
      <td>51.60%</td>
      <td>99.49%</td>
    </tr>
    <tr>
      <td>Winogrande</td>
      <td>70.32%</td>
      <td>71.11%</td>
      <td>101.12%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>64.70%</b></td>
      <td><b>63.98%</b></td>
      <td><b>98.89%</b></td>
    </tr>
    <tr>
      <td rowspan="3"><b>Vision Evals</b></td>
      <td>MMMU (val)</td>
      <td>39.89%</td>	
      <td>40.44%</td>
      <td>101.38%</td>
    </tr>
    <tr>
      <td>ChartQA</td>
      <td>50.76%</td>
      <td>49.80%</td>
      <td>98.11%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>45.33%</b></td>
      <td><b>45.12%</b></td>
      <td><b>99.74%</b></td>
    </tr>
  </tbody>
</table>
