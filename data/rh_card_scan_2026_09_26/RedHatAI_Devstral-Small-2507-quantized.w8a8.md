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
- INT8
- compressed-tensors
license: mit
license_name: mit
name: RedHatAI/Devstral-Small-2507
description: This model was obtained by quantizing weights and activations of Devstral-Small-2507 to INT8 data type.
readme: https://huggingface.co/RedHatAI/Devstral-Small-2507-quantized.w8a8/main/README.md
tasks:
- text-to-text
provider: mistralai
---

# Devstral-Small-2507-quantized.w8a8

## Model Overview
- **Model Architecture:** MistralForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Release Date:** 08/29/2025
- **Version:** 1.0
- **Model Developers:** Red Hat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing weights and activations of [Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%).
Weight quantization also reduces disk size requirements by approximately 50%.


## Creation
<details>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below.

  ```bash
python quantize.py --model_path mistralai/Devstral-Small-2507 --calib_size 512 --dampening_frac 0.05
```

  ```python
import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.messages import (
    SystemMessage, UserMessage
)

def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = os.path.join(repo_id, filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    return system_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.1)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype="auto",
    use_cache=False,
    trust_remote_code=True,
)

ds = load_dataset("garage-bAInd/Open-Platypus", split="train")
ds = ds.shuffle(seed=42).select(range(args.calib_size))

SYSTEM_PROMPT = load_system_prompt(args.model_path, "SYSTEM_PROMPT.txt")
tokenizer = MistralTokenizer.from_hf_hub("mistralai/Devstral-Small-2507")

def tokenize(sample):
    tmp = tokenizer.encode_chat_completion(
        ChatCompletionRequest(
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                UserMessage(content=sample['instruction']),
            ],
        )
    )
    return {'input_ids': tmp.tokens}

ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = [
    SmoothQuantModifier(
      smoothing_strength=0.8,
      mappings=[
          [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
          [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
          [["re:.*down_proj"], "re:.*up_proj"],
      ],
    ),
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W8A8",
        dampening_frac=args.dampening_frac,
    )
]
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=args.calib_size,
    max_seq_length=8192,
)

save_path = args.model_path + "-quantized.w8a8"
model.save_pretrained(save_path)
```
</details>

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```bash
vllm serve RedHatAI/Devstral-Small-2507-quantized.w8a8 --tensor-parallel-size 1 --tokenizer_mode mistral
```

## Evaluation

The model was evaluated on popular coding tasks (HumanEval, HumanEval+, MBPP, MBPP+) via [EvalPlus](https://github.com/evalplus/evalplus) and vllm backend (v0.10.1.1).
For evaluations, we run greedy sampling and report pass@1. The command to reproduce evals:
```bash
evalplus.evaluate --model "RedHatAI/Devstral-Small-2507-quantized.w8a8" \
                  --dataset [humaneval|mbpp] \
                  --base-url http://localhost:8000/v1 \
                  --backend openai --greedy
```


### Accuracy

|                             | Recovery (%) | mistralai/Devstral-Small-2507 | RedHatAI/Devstral-Small-2507-quantized.w8a8<br>(this model) |
| --------------------------- | :----------: | :------------------: | :--------------------------------------------------: |
| HumanEval                   | 100.67        | 89.0                | 89.6                                                |
| HumanEval+                  | 101.48        | 81.1                | 82.3                                                |
| MBPP                        | 98.71       | 77.5                | 76.5                                                |
| MBPP+                       | 102.42       | 66.1                | 67.7                                                |
| **Average Score**           | **100.77**    | **78.43**            | **79.03**                                            |
