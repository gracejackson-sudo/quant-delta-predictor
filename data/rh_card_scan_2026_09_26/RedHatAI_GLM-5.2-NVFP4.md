---
license: mit
base_model:
- zai-org/GLM-5.2
library_name: transformers
tags:
- compressed-tensors
- vLLM
---

# RedHatAI/GLM-5.2-NVFP4

This is a quantized version of `zai-org/GLM-5.2` with MoE layers quantized to NVFP4.

## Usage

This model is intended for deployment with vLLM and requires the following branch: https://github.com/vllm-project/vllm/pull/41276.
You can serve the model using

```bash
vllm serve RedHatAI/GLM-5.2-NVFP4 \
    --tensor-parallel-size 8 \
    --kv_cache_dtype=fp8
```

## Evaluation ##

| Model | GPQA Diamond |
| - | - |
| zai-org/GLM-5.2 | 91.2 |
| RedHatAI/GLM-5.2-NVFP4 | 90.0 | 

## Creation Process

This model was created using [LLM Compressor](https://github.com/vllm-project/llm-compressor). The example script can be found in `examples/quantizing_moe/glm5_example.py` [[Example] GLM5.2 Example](https://github.com/vllm-project/llm-compressor/pull/2869). Quantizing the model with data parallelism and 6xA100 takes about 3 hours.

```python
import torch
from compressed_tensors.offload import init_dist
from compressed_tensors.quantization.quant_scheme import (
    NVFP4,
    QuantizationScheme,
)
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.datasets.utils import get_rank_partition
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

# Load the model
init_dist()
model_id = "zai-org/GLM-5.2"
with load_context():
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto_offload",
        max_memory={},
        offload_folder="/mnt/nvme-data/engine/kylesayrs/offload_folder",
    )
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID, split=get_rank_partition(DATASET_SPLIT, NUM_CALIBRATION_SAMPLES)
)
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
recipe = QuantizationModifier(
    config_groups={
        "mlp": QuantizationScheme(
            targets=[r"re:.*mlp\..*"],
            **NVFP4,
        ),
    },
    ignore=[
        r"re:^model\.layers\.[0-2]\..*"
        r"re:.*mlp\.gate.*",  # not technically necessary
        r"re:.*shared_experts.*",
        r"lm_head",
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    batch_size=4,
    recipe=recipe,
    shuffle_calibration_samples=False,
)

# Save to disk compressed.
# Note: base checkpoint generation_config needs fixing for newer transformers versions
model.generation_config.top_p = None
SAVE_DIR = (
    "/mnt/nvme-data/engine/kylesayrs/"
    + model_id.rstrip("/").split("/")[-1]
    + "-NVFP4"
)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

torch.distributed.destroy_process_group()
```