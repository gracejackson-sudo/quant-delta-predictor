---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- int4
- vllm
- llm-compressor
- compressed-tensors
base_model: meta-models/Muse-Glimmer-30B
license: apache-2.0
---

# Muse-Glimmer-30B-INT4

## Model Overview
- **Model Architecture:** MuseGlimmerForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** None
- **Release Date:** 2026-08-11
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B).

### Model Optimizations

This model was obtained by quantizing the weights of [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) to INT4 data type while keeping activations in original precision, ready for inference with vLLM.

This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### vLLM Serving

```
docker run --gpus all \
    --privileged --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:muse-glimmer RedHatAI/Muse-Glimmer-30B-INT4 \
    --generation-config auto \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser muse_glimmer \
    --reasoning-parser muse_glimmer
```

For detailed instructions including multi-GPU deployment, multimodal inference, etc see the [Muse-Glimmer 30B vLLM usage guide](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B).

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from open-perfectblend, using the W4A16 GPTQ scheme, exported in compressed-tensors format.

<details>

```python
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    MuseGlimmerForConditionalGeneration,
)

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.utils import load_context

MODEL_ID = "meta-models/Muse-Glimmer-30B"

# Load model.
with load_context(MuseGlimmerForConditionalGeneration):
    model = MuseGlimmerForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

DATASET_ID = "mlabonne/open-perfectblend"
DATASET_SPLIT = "train"

# Select number of samples. 512 samples is recommended for GPTQ.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}")
ds = ds.shuffle(seed=42)

ROLE_MAP = {"human": "user", "gpt": "assistant"}


def preprocess(example):
    messages = [
        {"role": ROLE_MAP.get(msg["from"], msg["from"]), "content": msg["value"]}
        for msg in example["conversations"]
    ]
    return {
        "text": processor.apply_chat_template(
            messages,
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return processor.tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm and scheme.
recipe = GPTQModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["re:.*vision.*", "lm_head", "re:.*embed_tokens.*"],
)

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = processor.tokenizer(
    "Hello my name is", return_tensors="pt"
).input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=100)
print(processor.tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-INT4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

</details>