---
base_model:
- JetBrains/Mellum2-12B-A2.5B-Thinking
tags:
- mellum
- fp8
- vllm
- compressed-tensors
name: RedHatAI/Mellum2-12B-A2.5B-Thinking-FP8-Dynamic
---

# FP8 Quantized RedHatAI/Mellum2-12B-A2.5B-Thinking-FP8-Dynamic

This is a preliminary version (and subject to change) of the FP8 quantized [JetBrains/Mellum2-12B-A2.5B-Thinking](https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Thinking) model.
The model has both weights and activations quantized to FP8 with [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor).

It is compatible and tested against vllm main. Deploy it with: `vllm serve RedHatAI/Mellum2-12B-A2.5B-Thinking-FP8-Dynamic`

## Creation

```python
import json, os

from compressed_tensors.offload import dispatch_model
from transformers import Qwen3MoeForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Thinking"

# Load model.
model = Qwen3MoeForCausalLM.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_model(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)


config_path = os.path.join(SAVE_DIR, "config.json")

with open(config_path, "r") as f:
    config = json.load(f)

config["architectures"] = ["MellumForCausalLM"]
config["model_type"] = "mellum"

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
```
