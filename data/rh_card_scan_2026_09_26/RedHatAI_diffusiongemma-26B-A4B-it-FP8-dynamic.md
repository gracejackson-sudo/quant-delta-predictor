---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
base_model: google/diffusiongemma-26B-A4B-it
name: RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic
license: apache-2.0
license_name: apache-2.0
description: FP8-dynamic quantized variant of google/diffusiongemma-26B-A4B-it.
readme: https://huggingface.co/RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic/blob/main/README.md
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
provider: Google
validated_on:
  - RHOAI 3.5
  - RHAIIS 3.5
  - vLLM 0.24.0
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  diffusiongemma-26B-A4B-it-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>

This model is an FP8 quantized version of [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it). The model has both weights and activations quantized to FP8 using [vllm/llm-compressor](https://github.com/vllm-project/llm-compressor) and in the [compressed-tensors](https://github.com/vllm-project/compressed-tensors) format.
It was evaluated on several tasks to assess its quality in comparison to the unquantized model using vLLM.

## Deployment
```bash
VLLM_USE_V2_MODEL_RUNNER=1
vllm serve  RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic \
    --trust-remote-code \
    --max-num-seqs 4 \
    --hf-overrides '{"diffusion_sampler": "entropy_bound", "diffusion_entropy_bound": 0.1}' \
    --default-chat-template-kwargs '{"enable_thinking": true}'
```
## Creation
```python
"""
Quantize DiffusionGemma model to FP8 using LLM Compressor v0.11.0

Model: google/diffusiongemma-26B-A4B-it
- Total parameters: ~25.8B
- Expert parameters: 22.8B (88.4%)
- Non-expert parameters: 3.0B (11.6%)

Note: This will require a local update to transformers to support the model definition.
"""

import torch
from compressed_tensors.offload import dispatch_model
from transformers import AutoProcessor
from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

from llmcompressor import oneshot
from llmcompressor.modeling.diffusion_gemma4 import (  # noqa: F401
    CalibrationDiffusionGemmaTextExperts,
)
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model
MODEL_ID = "google/diffusiongemma-26B-A4B-it"
model = DiffusionGemmaForBlockDiffusion.from_pretrained(
    MODEL_ID, dtype="auto", trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

# CalibrationDiffusionGemmaTextExperts replaces the original
# DiffusionGemmaTextExperts class during calibration to:
# 1. Linearize the 3D expert tensors into individual nn.Linear modules
# 2. Ensure all experts are properly calibrated, even those not activated
#    for certain tokens during calibration

# Configure the quantization scheme
# FP8 Dynamic for all Linear layers
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:.*embed.*",
        "re:.*router",
        "re:.*vision_tower.*",
        "re:.*self_conditioning.*",
    ],
)

oneshot(
    model=model,
    recipe=recipe
)


# Test sample generation
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)

# "The reason the sky is blue is because" + chat template
input_ids = torch.tensor(
    [[
        2, 105, 2364, 107, 818, 3282, 506, 7217, 563, 3730, 563,
        1547, 106, 107, 105, 4368, 107
    ]]
).to(model.device)

output = model.generate(
    input_ids,
    max_new_tokens=100,
    max_denoising_steps=48,
)
print(processor.tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk in compressed-tensors format
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

## Accuracy
The following metrics were generated when serving the quantized model with vLLM on a single B200 GPU.

| Benchmark     | google/diffusiongemma-26B-A4B-it  | RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic | Recovery (%) |
|---------------|-------:|------------:|-------------:|
| AIME 2025     | 0.437 | 0.423 | 96.8% |
| GPQA Diamond  | 0.641 | 0.657 | 102.5% |
| IFEval        | 0.879 | 0.862 | 98.1% |
| GSM8K         | 0.943 | 0.942 | 99.9% |
| MMLU 0-Shot   | 0.539 | 0.505 | 93.7% |
| **Thinking**  |        |        |        |
| AIME 2025     | 0.650 | 0.660 | 101.5% |
| GPQA Diamond  | 0.698 | 0.689 | 98.7% |
| GSM8K         | 0.951 | 0.952 | 100.1% |