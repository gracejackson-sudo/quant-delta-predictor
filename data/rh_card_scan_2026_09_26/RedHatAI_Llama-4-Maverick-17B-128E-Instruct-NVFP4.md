---
tags:
- fp4
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
base_model: meta-llama/Llama-4-Maverick-17B-128E-Instruct
---

# Llama-4-Maverick-17B-128E-Instruct-NVFP4

## Model Overview
- **Model Architecture:** Meta-Llama-3.1
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. 
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 10/29/25
- **Version:** 1.0
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** RedHatAI

This model is a quantized version of [Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) to FP4 data type, ready for inference with vLLM>=0.9.1
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.
<details>
<summary>Model Usage Code</summary>
  
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4"
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
</details>

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [LLM Compressor with calibration samples from neuralmagic/calibration dataset](https://github.com/vllm-project/llm-compressor/blob/main/examples/multimodal_vision/llama4_example.py), as presented in the code snipet below.

<details>
<summary>Model Creation Code</summary>

```python
import torch
from datasets import load_dataset
from transformers import Llama4ForConditionalGeneration, Llama4Processor

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
model = Llama4ForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
processor = Llama4Processor.from_pretrained(model_id)
# MoE calibration is now handled automatically by the pipeline.
# The `SequentialLlama4TextMoe` modules (from `llmcompressor.modeling.llama4`)
# will be applied during calibration to enable
# proper expert calibration and vLLM compatibility.
# These replace the original `Llama4TextMoe` class from
# `transformers.models.llama4.modeling_llama4`.

DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")


def preprocess_function(example):
    messgages = []
    for message in example["messages"]:
        messgages.append(
            {
                "role": message["role"],
                "content": [{"type": "text", "text": message["content"]}],
            }
        )

    return processor.apply_chat_template(
        messgages,
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


# Configure the quantization algorithm to run.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:.*self_attn",
        "re:.*router",
        "re:.*vision_model.*",
        "re:.*multi_modal_projector.*",
        "Llama4TextAttention",
    ],
)

# Apply algorithms.
# due to the large size of Llama4, we specify sequential targets such that
# only one MLP is loaded into GPU memory at a time
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    sequential_targets=["Llama4TextMLP"],
    data_collator=data_collator,
)


# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

```
</details>

## Evaluation

This model was evaluated on the well-known OpenLLM v1, OpenLLM v2 and HumanEval_64 benchmarks. All evaluations were conducted using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness).

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Llama-4-Maverick-17B-128E-Instruct</th>
      <th>Llama-4-Maverick-17B-128E-Instruct-NVFP4 (this model)</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM -->
    <tr>
      <td rowspan="8"><b>OpenLLM V1</b></td>
      <td>arc_challenge_llama</td>
      <td>95.97</td>
      <td>95.88</td>
      <td>99.91</td>
    </tr>
    <tr>
      <td>gsm8k_llama</td>
      <td>96.13</td>
      <td>96.06</td>
      <td>99.93</td>
    </tr>
    <tr>
      <td>mmlu_llama</td>
      <td>86.77</td>
      <td>85.49</td>
      <td>98.53</td>
    </tr>
    <tr>
      <td>mmlu_cot_llama</td>
      <td>89.49</td>
      <td>88.72</td>
      <td>99.14</td>
    </tr>
    <tr>
      <td>truthfulqa_mc2</td>
      <td>68.23</td>
      <td>68.42</td>
      <td>100.28</td>
    </tr>
    <tr>
      <td>winogrande</td>
      <td>77.98</td>
      <td>77.74</td>
      <td>99.69</td>
    </tr>
    <tr>
      <td>hellaswag</td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b></b></td>
      <td><b>85.23</b></td>
      <td><b></b></td>
    </tr>
    <!-- Leaderboard (vLLM 0.11.0) -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>BBH</td>
      <td></td>
      <td>69.52</td>
      <td></td>
    </tr>
    <tr>
      <td>MMLU-Pro</td>
      <td></td>
      <td>62.83</td>
      <td></td>
    </tr>
    <tr>
      <td>MuSR</td>
      <td></td>
      <td>45.77</td>
      <td></td>
    </tr>
    <tr>
      <td>IFEval</td>
      <td></td>
      <td>89.45</td>
      <td></td>
    </tr>
    <tr>
      <td>GPQA</td>
      <td></td>
      <td>30.54</td>
      <td></td>
    </tr>
    <tr>
      <td>Math-Hard</td>
      <td></td>
      <td>64.95</td>
      <td></td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td></td>
      <td><b>60.51</b></td>
      <td></td>
    </tr>
    <!-- Coding -->
    <tr>
      <td rowspan="1"><b>Coding</b></td>
      <td>HumanEval_64 (pass@2)</td>
      <td></td>
      <td>88.88</td>
      <td></td>
    </tr>

  </tbody>
</table>


### Reproduction

The results were obtained using the following commands:

<details>
<summary>Model Evaluation Commands</summary>

#### MMLU_LLAMA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks mmlu_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### MMLU_COT_LLAMA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks mmlu_cot_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks arc_challenge_llama \
  --apply_chat_template \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks gsm8k_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### Hellaswag
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks hellaswag \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks winogrande \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks truthfulqa \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval and HumanEval_64
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_64_instruct \
  --batch_size auto
```
</details>
