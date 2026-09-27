---
tags:
- fp4
- vllm
language:
- en
- de
- fr
- it
- pt
- hi
- es
- th
pipeline_tag: text-generation
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4
description: Qwen3-VL delivers comprehensive upgrades across the board superior text understanding & generation, deeper visual perception & reasoning, extended context length, enhanced spatial and video dynamics comprehension, and stronger agent interaction capabilities.
readme: https://huggingface.co/RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4/main/README.md
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
provider: Alibaba Cloud
validated_on:
  - RHOAI 3.3
  - RHAIIS 3.3
  - vLLM 0.13.0
base_model: Qwen/Qwen3-VL-235B-A22B-Instruct
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Qwen3-VL-235B-A22B-Instruct-NVFP4
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Qwen/Qwen3-VL-235B-A22B-Instruct
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 10/29/2025
- **Version:** 1.0
- **Model Developers:** RedHatAI
- **ModelCar Storage URI**: oci://registry.redhat.io/rhai/modelcar-qwen3-vl-235b-a22b-instruct-nvfp4:3.0
- **Validated on vLLM:** 0.13.0
- **Validated on RHAIIS:** 3.3
- **Validated on RHOAI:** 3.3

This model is a quantized version of [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) to FP4 data type, ready for inference with vLLM>=0.9.1
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights and activations of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4"
number_gpus = 1

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [LLM Compressor with calibration samples from UltraChat](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/llama3_example.py), as presented in the code snipet below.

<details>
  
```python
import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# NOTE: Requires a minimum of transformers 4.57.0

MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"


# Load model.
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = replace_modules_for_calibration(model)

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

This model was evaluated on the well-known OpenLLM v1, OpenLLM v2 and HumanEval_64 benchmarks using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness). The Reasoning evals were done using [ligheval](https://github.com/neuralmagic/lighteval).

### Accuracy
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Qwen/Qwen3-VL-235B-A22B-Instruct</th>
      <th>RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4 (this model)</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM -->
    <tr>
      <td rowspan="7"><b>OpenLLM</b></td>
      <td>arc_challenge</td>
      <td>72.95</td>
      <td>71.59</td>
      <td>98.13</td>
    </tr>
    <tr>
      <td>gsm8k</td>
      <td>90.37</td>
      <td>88.25</td>
      <td>97.65</td>
    </tr>
    <tr>
      <td>hellaswag</td>
      <td>87.94</td>
      <td>86.80</td>
      <td>98.70</td>
    </tr>
    <tr>
      <td>mmlu</td>
      <td>87.12</td>
      <td>86.22</td>
      <td>98.97</td>
    </tr>
    <tr>
      <td>truthfulqa_mc2</td>
      <td>63.31</td>
      <td>62.37</td>
      <td>98.52</td>
    </tr>
    <tr>
      <td>winogrande</td>
      <td>81.93</td>
      <td>80.43</td>
      <td>98.17</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>80.60</b></td>
      <td><b>79.28</b></td>
      <td><b>98.35</b></td>
    </tr>
    <!-- Vision -->
    <tr>
      <td rowspan="7"><b>Vision</b></td>
      <td>mmmu_val</td>
      <td>63.56</td>
      <td>62.11</td>
      <td>97.71</td>
    </tr>
    <tr>
      <td>chartqa</td>
      <td>90.52</td>
      <td>89.00</td>
      <td>98.32</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>77.04</b></td>
      <td><b>75.56</b></td>
      <td><b>98.08</b></td>
    </tr>
  </tbody>
</table>



### Reproduction

The results were obtained using the following commands:

<details>

#### OpenLLM 
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks openllm \
  --batch_size auto
```

#### Vision
```
python3 -m lmms_eval \
    --model vllm \
    --model_args model=RedHatAI/Qwen3-VL-235B-A22B-Instruct-NVFP4,tensor_parallel_size=4,max_model_len=20000 \
    --tasks chartqa,mmmu_val \
    --batch_size 1
```

</details>