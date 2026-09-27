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
license: llama3.2
base_model: meta-llama/Llama-3.2-90B-Vision-Instruct
---

# Llama-3.2-90B-Vision-Instruct-FP8-dynamic

## Model Overview
- **Model Architecture:** Meta-Llama-3.2
  - **Input:** Text/Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. Similarly to [Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 9/25/2024
- **Version:** 1.0
- **License(s):** [llama3.2](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct/blob/main/LICENSE.txt)
- **Model Developers:** Neural Magic

Quantized version of [Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct) to FP8 data type, ready for inference with vLLM built from source.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the FP8 representations of the quantized weights and activations. Activations are also quantized on a per-token dynamic basis.
[LLM Compressor](https://github.com/vllm-project/llm-compressor) is used for quantization.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

# Initialize the LLM
model_name = "neuralmagic/Llama-3.2-90B-Vision-Instruct-FP8-dynamic"
llm = LLM(model=model_name, max_num_seqs=1, enforce_eager=True, tensor_parallel_size=4)

# Load the image
image = ImageAsset("cherry_blossom").pil_image.convert("RGB")

# Create the prompt
question = "If I had to write a haiku for this one, it would be: "
prompt = f"<|image|><|begin_of_text|>{question}"

# Set up sampling parameters
sampling_params = SamplingParams(temperature=0.2, max_tokens=30)

# Generate the response
inputs = {
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    },
}
outputs = llm.generate(inputs, sampling_params=sampling_params)

# Print the generated text
print(outputs[0].outputs[0].text)
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

```
vllm serve neuralmagic/Llama-3.2-90B-Vision-Instruct-FP8-dynamic --enforce-eager --max-num-seqs 16 --tensor-parallel-size 4
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor/blob/f90013702b15bd1690e4e2fe9ed434921b6a6199/examples/quantization_w8a8_fp8/llama3.2_vision_example.py), as presented in the code snipet below.

```python
from transformers import AutoProcessor, MllamaForConditionalGeneration

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot, wrap_hf_model_class

MODEL_ID = "meta-llama/Llama-3.2-90B-Vision-Instruct"

# Load model.
model_class = wrap_hf_model_class(MllamaForConditionalGeneration)
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

TBD

### Reproduction

TBD
