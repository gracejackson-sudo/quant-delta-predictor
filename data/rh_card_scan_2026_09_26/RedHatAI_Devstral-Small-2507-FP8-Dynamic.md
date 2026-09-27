---
language:
- en
base_model:
- mistralai/Devstral-Small-2507
pipeline_tag: text-generation
tags:
- mistral
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP8
- compressed-tensors
license: mit
license_name: mit
name: RedHatAI/Devstral-Small-2507
description: This model was obtained by quantizing weights and activations of Devstral-Small-2507 to FP8 data type.
readme: https://huggingface.co/RedHatAI/Devstral-Small-2507-FP8-Dynamic/main/README.md
tasks:
- text-to-text
provider: mistralai
---

# Devstral-Small-2507-FP8-Dynamic

## Model Overview
- **Model Architecture:** MistralForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP8
  - **Weight quantization:** FP8
- **Release Date:** 08/28/2025
- **Version:** 1.0
- **Model Developers:** Red Hat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing weights and activations of [Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%).
Weight quantization also reduces disk size requirements by approximately 50%.

## Creation
<details>
This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below.

```python
from transformers import AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


MODEL_ID = "mistralai/Devstral-Small-2507"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
)
oneshot(model=model, recipe=recipe)
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
```
</details>

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```bash
vllm serve RedHatAI/Devstral-Small-2507-FP8-Dynamic --tensor-parallel-size 1 --tokenizer_mode mistral
```

## Evaluation

The model was evaluated on popular coding tasks (HumanEval, HumanEval+, MBPP, MBPP+) via [EvalPlus](https://github.com/evalplus/evalplus) and vllm backend (v0.10.1.1).
For evaluations, we run greedy sampling and report pass@1. The command to reproduce evals:
```bash
evalplus.evaluate --model "RedHatAI/Devstral-Small-2507-FP8-Dynamic" \
                  --dataset [humaneval|mbpp] \
                  --base-url http://localhost:8000/v1 \
                  --backend openai --greedy
```


### Accuracy

|                             | Recovery (%) | mistralai/Devstral-Small-2507 | RedHatAI/Devstral-Small-2507-FP8-Dynamic<br>(this model) |
| --------------------------- | :----------: | :------------------: | :--------------------------------------------------: |
| HumanEval                   | 100.67        | 89.0                | 89.6                                                |
| HumanEval+                  | 102.22        | 81.1                | 82.9                                                |
| MBPP                        | 97.29       | 77.5                | 75.4                                                |
| MBPP+                       | 98.03       | 66.1                | 64.8                                                |
| **Average Score**           | **99.68**    | **78.43**            | **78.18**                                            |
