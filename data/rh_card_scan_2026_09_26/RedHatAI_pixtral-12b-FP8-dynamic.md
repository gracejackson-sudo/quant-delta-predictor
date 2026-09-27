---
tags:
- fp8
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
library_name: vllm
base_model:
- mistral-community/pixtral-12b
- mistralai/Pixtral-12B-2409
---

# pixtral-12b-FP8-dynamic

## Model Overview
- **Model Architecture:** Pixtral (Llava)
  - **Input:** Text/Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. Similar to [mistralai/Pixtral-12B-2409](https://huggingface.co/mistralai/Pixtral-12B-2409), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 11/1/2024
- **Version:** 1.0
- **License(s):** [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)
- **Model Developers:** Neural Magic

Quantized version of [mistral-community/pixtral-12b](https://huggingface.co/mistral-community/pixtral-12b).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [mistral-community/pixtral-12b](https://huggingface.co/mistral-community/pixtral-12b) to FP8 data type, ready for inference with vLLM built from source.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the FP8 representations of the quantized weights and activations. Activations are also quantized on a per-token dynamic basis.
[LLM Compressor](https://github.com/vllm-project/llm-compressor) is used for quantization.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams

# Initialize the LLM
model_name = "neuralmagic/pixtral-12b-FP8-dynamic"
llm = LLM(model=model_name, max_model_len=10000)

# Create the prompt
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]

# Set up sampling parameters
sampling_params = SamplingParams(temperature=0.2, max_tokens=100)

# Generate the response
outputs = llm.chat(messages, sampling_params=sampling_params)

# Print the generated text
for output in outputs:
    print(output.outputs[0].text)
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

```
vllm serve neuralmagic/pixtral-12b-FP8-dynamic
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor/blob/f90013702b15bd1690e4e2fe9ed434921b6a6199/examples/quantization_w8a8_fp8/llama3.2_vision_example.py), as presented in the code snipet below.

```python
from transformers import AutoProcessor, LlavaForConditionalGeneration

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot, wrap_hf_model_class

MODEL_ID = "mistral-community/pixtral-12b"

# Load model.
model_class = wrap_hf_model_class(LlavaForConditionalGeneration)
model = model_class.from_pretrained(MODEL_ID, device_map="auto", torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["re:.*lm_head", "re:multi_modal_projector.*", "re:vision_model.*"],
)

# Apply quantization and save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
oneshot(model=model, recipe=recipe, output_dir=SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
input_ids = processor(text="Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=20)
print(processor.decode(output[0]))
print("==========================================")
```

## Evaluation

### Multimodal Benchmarks

|                   | pixtral-12b | pixtral-12b-FP8-dynamic |
|:-------------------:|:-------------:|:----------:|
| **MMMU** *(CoT)*      | 49.44       | 51.11     | 
| **Mathvista** *(CoT)*   | 58.1        | 59.4     |
| **ChartQA** *(CoT)*    | 82.64        | 82.68     | 
| **DocVQA** *(ANLS)*        | 89.36       | 89.35     | 

### Text Benchmarks

|                   | pixtral-12b | pixtral-12b-FP8-dynamic |
|:-------------------:|:-------------:|:----------:|
| **MMLU** *(5-shot)*   |  69.27       | 68.96     |
| **Math** *(0-shot)*         |  43.82       | 43.27     |
| **Human Eval** *(Pass@1)*    |   77.80      | 76.4     |

### Reproduction

TBD
