---
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: any-to-any
base_model: google/gemma-4-12B-it
provider: Google
name: RedHatAI/gemma-4-12B-it-NVFP4
description: NVFP4 quantized variant of gemma-4-12B-it.
readme: https://huggingface.co/RedHatAI/gemma-4-12B-it-NVFP4/blob/main/README.md
tags:
- fp4
- vllm
- llm-compressor
- compressed-tensors
tasks:
- text-to-text
- text-generation
- image-text-to-text
- audio-text-to-text
validated_on:
  - RHOAI 3.5
  - RHAIIS 3.5
  - vLLM 0.24.0
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  gemma-4-12B-it-NVFP4
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
# NVFP4 Quantized RedHatAI/gemma-4-12B-it-NVFP4

This is a preliminary version (and subject to change) of NVFP4 quantized [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) model. 
The model has both weights and activations quantized to NVFP4 format with [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor).

It is compatible and tested against vllm nightly.

# Creation Script

Run this script with this LLM Compressor [PR](https://github.com/vllm-project/llm-compressor/pull/2647) and latest transformers to
quantize the model using GPTQ

<details>
  
```python
import torch
from compressed_tensors.offload import dispatch_model
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL_ID = "google/gemma-4-12B-it"
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 2048

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_function(example):
    messages = []
    for message in example["messages"]:
        messages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messages,
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


recipe = GPTQModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:.*embed_vision.*",
        "re:.*embed_audio.*",
        "re:.*vision_embedder.*",
    ],
)

oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = torch.tensor(
    [[
        2, 105, 2364, 107, 818, 3282, 506, 7217, 563, 3730, 563,
        1547, 106, 107, 105, 4368, 107
    ]]
).to(model.device)
output = model.generate(
    input_ids,
    max_new_tokens=100,
)
print(processor.tokenizer.decode(output[0]))
print("==========================================\n\n")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-GPTQ"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

# Patch config: transformers renames checkpoint keys on load (vision_embedder ->
# embed_vision), but save_pretrained reverts them. The ignore list in config.json
# uses HF names (embed_vision) while safetensors keys use checkpoint names
# (vision_embedder), so vllm can't match them. Add the checkpoint name explicitly.
import json as _json
_cfg_path = SAVE_DIR + "/config.json"
with open(_cfg_path) as _f:
    _cfg = _json.load(_f)
_qcfg = _cfg.get("quantization_config")
if _qcfg:
    _ign = _qcfg.setdefault("ignore", [])
    if "model.vision_embedder.patch_dense" not in _ign:
        _ign.append("model.vision_embedder.patch_dense")
        with open(_cfg_path, "w") as _f:
            _json.dump(_cfg, _f, indent=2)
        print("Patched config.json: added vision_embedder.patch_dense to ignore list")

```
</details>

# Preliminary Evaluations

1) GSM8K Platinum
2) Wikitext PPL
```
lm_eval --model vllm \
  --model_args "pretrained=RedHatAI/gemma-4-12B-it-NVFP4,dtype=auto,max_model_len=$MAX_MODEL_LEN,add_bos_token=True,gpu_memory_utilization=0.85" \
  --tasks gsm8k_platinum --num_fewshot 5 --apply_chat_template --batch_size auto

lm_eval --model vllm \
  --model_args "pretrained=RedHatAI/gemma-4-12B-it-NVFP4,dtype=auto,max_model_len=$MAX_MODEL_LEN,add_bos_token=True,gpu_memory_utilization=0.85" \
  --tasks wikitext --num_fewshot 0 --apply_chat_template --batch_size auto
```

Evals:
```
+---------------+------------------+--------------+---------------+----------+
| model_name    | flexible-extract | strict-match | bits_per_byte | byte_ppl |
+---------------+------------------+--------------+---------------+----------+
| baseline-bf16 | 0.9082           | 0.8958       | 1.9125        | 3.7645   |
| NVFP4-RTN     | 0.8892           | 0.8776       | 2.0945        | 4.2706   |
| NVFP4-iMatrix | 0.8974           | 0.8825       | 1.9855        | 3.9600   | 
| *NVFP4-GPTQ*  | 0.9016           | 0.8867       | 2.0704        | 4.2001   |
+---------------+------------------+--------------+---------------+----------+

Recovery
+---------------+------------------+--------------+---------------+----------+
| *NVFP4-GPTQ*  | 99.27%            | 98.98%      | 92.37%        | 89.6%    |
+---------------+------------------+--------------+---------------+----------+
```