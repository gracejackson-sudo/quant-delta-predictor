---
tags:
- w4a16
- int4
- vllm
- vision
license: apache-2.0
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
  - en
base_model: microsoft/Phi-3-vision-128k-instruct
library_name: transformers
---

# Phi-3-vision-128k-instruct-W4A16-G128

## Model Overview
- **Model Architecture:** Phi-3-vision-128k-instruct
  - **Input:** Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** FP16
- **Release Date:** 1/31/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct).

### Model Optimizations

This model was obtained by quantizing the weights of [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) to INT4 data type, ready for inference with vLLM >= 0.5.2.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="neuralmagic/Phi-3-vision-128k-instruct-W4A16-G128",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
)

# prepare inputs
question = "What is the content of this image?"
inputs = {
    "prompt": f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n",
    "multi_modal_data": {
        "image": ImageAsset("cherry_blossom").pil_image.convert("RGB")
    },
}

# generate response
print("========== SAMPLE GENERATION ==============")
outputs = llm.generate(inputs, SamplingParams(temperature=0.2, max_tokens=64))
print(f"PROMPT  : {outputs[0].prompt}")
print(f"RESPONSE: {outputs[0].outputs[0].text}")
print("==========================================")
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below as part a multimodal announcement blog.

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot

# Load model.
model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    _attn_implementation="eager",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
processor.chat_template = processor.tokenizer.chat_template

# Calibration dataset arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = "test[:512]"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

# Apply chat template and tokenize inputs.
def preprocess_and_tokenize(example):
    messages = [{"role": "user", "content": "<|image_1|>\nWhat does the image show?"}]
    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )
    images = example["image"]

    return processor(
        text=text,
        images=images,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )

ds = ds.map(preprocess_and_tokenize, writer_batch_size=1, remove_columns=ds.column_names)

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Recipe
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    sequential_targets=["Phi3DecoderLayer"],
    ignore=["lm_head", "re:model.vision_embed_tokens.*"],
)

# Perform oneshot
SAVE_DIR = model_id.split("/")[1] + "-W4A16-G128"

oneshot(
    model=model,
    processor=processor,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    output_dir=SAVE_DIR
)
```

## License

The model is licensed under the [MIT license](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/resolve/main/LICENSE).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
