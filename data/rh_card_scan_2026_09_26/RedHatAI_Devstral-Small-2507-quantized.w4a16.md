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
- INT4
- compressed-tensors
license: mit
license_name: mit
name: RedHatAI/Devstral-Small-2507
description: This model was obtained by quantizing weights of Devstral-Small-2507 to INT4 data type.
readme: https://huggingface.co/RedHatAI/Devstral-Small-2507-quantized.w4a16/main/README.md
tasks:
- text-to-text
provider: mistralai
---

# Devstral-Small-2507-quantized.w4a16

## Model Overview
- **Model Architecture:** MistralForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT4
  - **Weight quantization:** None
- **Release Date:** 08/29/2025
- **Version:** 1.0
- **Model Developers:** Red Hat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing weights of [Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) to INT4 data type.
This optimization reduces the number of bits used to represent weights from 16 to 4, reducing GPU memory requirements (by approximately 75%).
Weight quantization also reduces disk size requirements by approximately 75%.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```bash
vllm serve RedHatAI/Devstral-Small-2507-quantized.w4a16 --tensor-parallel-size 1 --tokenizer_mode mistral
```

## Creation
<details>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below.

  ```bash
python quantize.py --model_path mistralai/Devstral-Small-2507 --calib_size 1024 --dampening_frac 0.1 --observer mse --sym false --actorder weight
```

  ```python
import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
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

def parse_actorder(value):
    if value.lower() == "false":
        return False
    elif value.lower() == "weight":
        return "weight"
    elif value.lower() == "group":
        return "group"
    else:
        raise argparse.ArgumentTypeError("Invalid value for --actorder.")

def parse_sym(value):
    if value.lower() == "false":
        return False
    elif value.lower() == "true":
        return True
    else:
        raise argparse.ArgumentTypeError(f"Invalid value for --sym. Use false or true, but got {value}")


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.1)
parser.add_argument('--observer', type=str, default="minmax")
parser.add_argument('--sym', type=parse_sym, default=True)
parser.add_argument(
    '--actorder',
    type=parse_actorder,
    default=False,
    help="Specify actorder as 'group' (string) or False (boolean)."
)
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

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=args.sym,
        group_size=128,
        strategy=QuantizationStrategy.GROUP,
        observer=args.observer,
        actorder=args.actorder
    ),
    input_activations=None,
    output_activations=None,
)

recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        dampening_frac=args.dampening_frac,
        config_groups={"group_0": quant_scheme},
    )
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=args.calib_size,
    max_seq_length=8192,
)

save_path = args.model_path + "-quantized.w4a16"
model.save_pretrained(save_path)
```
</details>

## Evaluation

The model was evaluated on popular coding tasks (HumanEval, HumanEval+, MBPP, MBPP+) via [EvalPlus](https://github.com/evalplus/evalplus) and vllm backend (v0.10.1.1).
For evaluations, we run greedy sampling and report pass@1. The command to reproduce evals:
```bash
evalplus.evaluate --model "RedHatAI/Devstral-Small-2507-quantized.w4a16" \
                  --dataset [humaneval|mbpp] \
                  --base-url http://localhost:8000/v1 \
                  --backend openai --greedy
```


### Accuracy

|                             | Recovery (%) | mistralai/Devstral-Small-2507 | RedHatAI/Devstral-Small-2507-quantized.w4a16<br>(this model) |
| --------------------------- | :----------: | :------------------: | :--------------------------------------------------: |
| HumanEval                   | 98.65        | 89.0                | 87.8                                                |
| HumanEval+                  | 100.0        | 81.1                | 81.1                                                |
| MBPP                        | 98.97       | 77.5                | 76.7                                                |
| MBPP+                       | 102.12       | 66.1                | 67.5                                                |
| **Average Score**           | **99.81**    | **78.43**            | **78.28**                                            |
