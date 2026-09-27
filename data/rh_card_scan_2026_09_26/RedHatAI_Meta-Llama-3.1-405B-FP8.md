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
license: llama3.1
base_model: meta-llama/Meta-Llama-3.1-405B
---

# Meta-Llama-3.1-405B-FP8

## Model Overview
- **Model Architecture:** Meta-Llama-3.1
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. Similarly to [Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B), this model serves as a base version.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 8/6/2024
- **Version:** 1.0
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

Quantized version of [Meta-Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B).
It achieves an average score of 82.00 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), recovering 98.7% of dense performance.
<!-- whereas the unquantized model achieves 79.84. -->

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Meta-Llama-3.1-405B](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B) to FP8 data type, ready for inference with vLLM built from source.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-tensor quantization is applied, in which a single linear scaling maps the FP8 representations of the quantized weights and activations.
[LLM Compressor](https://github.com/vllm-project/llm-compressor) is used for quantization with 512 sequences of UltraChat.

<!-- ## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Meta-Llama-3.1-405B-FP8"
number_gpus = 2

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details. -->

## Creation

This model was created by applying [LLM Compressor with calibration samples from UltraChat](https://github.com/vllm-project/llm-compressor/blob/sa/big_model_support/examples/big_model_offloading/big_model_w8a8_calibrate.py), as presented in the code snipet below.

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map,
    custom_offload_device_map,
)

recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
"""

model_stub = "meta-llama/Meta-Llama-3.1-405B"
model_name = model_stub.split("/")[-1]

device_map = calculate_offload_device_map(
    model_stub, reserve_for_hessians=False, num_gpus=8, torch_dtype=torch.float16
)

model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype=torch.float16, device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_stub)

output_dir = f"./{model_name}-FP8"

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

oneshot(
    model=model,
    output_dir=output_dir,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
)
```

## Evaluation

The model was evaluated on MMLU, ARC-Challenge, GSM-8K, Hellaswag, Winogrande and TruthfulQA.
Evaluation was conducted using the Neural Magic fork of [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness/tree/llama_3.1_instruct) (branch llama_3.1_instruct) and the [vLLM](https://docs.vllm.ai/en/stable/) engine.
This version of the lm-evaluation-harness includes versions of ARC-Challenge that matches the prompting style of [Meta-Llama-3.1-evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-evals).
An asterisk indicates that some evaluations are still being collected.

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Meta-Llama-3.1-405B </strong>
   </td>
   <td><strong>Meta-Llama-3.1-405B-FP8(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>*
   </td>
   <td>84.72
   </td>
   <td>*
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (0-shot)
   </td>
   <td>95.99
   </td>
   <td>95.82
   </td>
   <td>99.82%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>88.10
   </td>
   <td>87.94
   </td>
   <td>99.82%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>90.02
   </td>
   <td>89.14
   </td>
   <td>99.02%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>87.61
   </td>
   <td>86.42
   </td>
   <td>98.64%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>49.83
   </td>
   <td>47.93
   </td>
   <td>96.19%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>*</strong>
   </td>
   <td><strong>82.00</strong>
   </td>
   <td><strong>98.70%</strong>
   </td>
  </tr>
</table>

### Reproduction

The results were obtained using the following commands:

#### MMLU
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-FP8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-FP8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks arc_challenge_llama_3.1_instruct \
  --num_fewshot 25 \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-FP8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size auto
```

#### Hellaswag
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-FP8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks hellaswag \
  --num_fewshot 10 \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-FP8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks winogrande \
  --num_fewshot 5 \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-FP8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks truthfulqa_mc \
  --num_fewshot 0 \
  --batch_size auto
```