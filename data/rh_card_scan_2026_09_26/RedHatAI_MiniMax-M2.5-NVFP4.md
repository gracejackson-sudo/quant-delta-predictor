---
library_name: transformers
license: other
license_name: modified-mit
license_link: https://github.com/MiniMax-AI/MiniMax-M2.5/blob/main/LICENSE
pipeline_tag: text-generation
base_model:
- MiniMaxAI/MiniMax-M2.5
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP4
---

# MiniMax-M2.5-NVFP4

## Model Overview
- **Model Architecture:** MiniMaxM2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
- **Intended Use Cases:**
  - Reasoning.
  - Function calling.
  - Subject matter experts via fine-tuning.
  - Multilingual instruction following.
  - Translation.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 03/28/2026
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights and activations of [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) to FP4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/MiniMax-M2.5-NVFP4"
number_gpus = 1
sampling_params = SamplingParams(temperature=1.0, top_p=0.95, top_k=40, min_p=0, max_tokens=256)

messages = [
    {"role": "user", "content": prompt}
]

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


```python
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modeling.minimax_m2 import (  # noqa: F401
    CalibrationMiniMaxM2SparseMoeBlock,
)
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load the model
model_id = "RedHatAI/MiniMax-M2.5-BF16"
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16, config=config,trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
# MoE calibration is handled automatically by the pipeline.
# The `CalibrationMiniMaxM2SparseMoeBlock` modules (from
# `llmcompressor.modeling.minimax_m2`) will be applied during calibration to enable
# proper expert calibration. These replace the original
# `MiniMaxM2SparseMoeBlock` class from
# `transformers.models.minimax_m2.modeling_minimax_m2`.

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
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

moe_ignores = [
    "lm_head",
    "re:.*block_sparse_moe.gate$",
]

# Experts live under `model.layers.*.block_sparse_moe.experts.<idx>.(w1|w2|w3)`.
EXPERT_TARGET_REGEX = [
    "re:.*block_sparse_moe\\.experts\\.\\d+\\.w1$",
    "re:.*block_sparse_moe\\.experts\\.\\d+\\.w2$",
    "re:.*block_sparse_moe\\.experts\\.\\d+\\.w3$",
]

recipe = QuantizationModifier(
    targets=EXPERT_TARGET_REGEX,
    scheme="NVFP4",
    weight_observer="mse",
    ignore= moe_ignores
)


# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    processor=tokenizer,
    recipe=recipe,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    sequential_targets=["MiniMaxM2DecoderLayer"],
)

# Save to disk compressed.
SAVE_DIR = model_id.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```
</details>
 



## Evaluation

The model was evaluated on the ifeval, mmlu_pro and gsm8k_platinum  using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), on reasoning tasks using [lighteval](https://github.com/neuralmagic/lighteval/tree/reasoning).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.


<details>
  <summary>Evaluation details</summary>

  Deploy using vllm to create an OpenAI-compatible API endpoint:

- vLLM:
    ```shell
    vllm serve RedHatAI/MiniMax-M2.5-NVFP4 --max-model-len 262144 --reasoning-parser deepseek_r1
    ```

  **lm-evaluation-harness**
  ```
  lm_eval --model local-chat-completions \
    --tasks mmlu_pro_chat \
    --model_args "model=RedHatAI/MiniMax-M2.5-NVFP4,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks ifeval \
    --model_args "model=RedHatAI/MiniMax-M2.5-NVFP4,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks gsm8k_platinum_cot_llama \
    --model_args "model=RedHatAI/MiniMax-M2.5-NVFP4,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/MiniMax-M2.5-NVFP4
    dtype: auto
    gpu_memory_utilization: 0.9
    max_model_length: 40960
    generation_parameters:
      temperature: 1.0
      top_k: 40
      min_p: 0.0
      top_p: 0.95
      max_new_tokens: 64000
  ```

  ```
  lighteval endpoint litellm lighteval_model_arguments.yaml  \
    "aime25|0,math_500|0,gpqa:diamond|0"
  ```


</details>

### Accuracy

| Benchmark | RedHatAI/MiniMax-M2.5-BF16 | RedHatAI/MiniMax-M2.5-NVFP4 | Recovery (%) |
|-----------|------------------------------------------|-------------------------------------------|--------------|
| GSM8k Platinum (0-shot) | 95.15 | 93.91 | 98.70 |
| IfEval (0-shot) | 92.05 | 89.89 | 97.66 |
| AIME 2025 | 87.50 | 77.08 | 88.10 |
| GPQA diamond | 83.67 | 80.30 | 95.98 |
| Math 500 | 87.33 | 87.73 | 100.46 |
| MMLU Pro Chat | 80.83 | 80.08 | 99.07 |
