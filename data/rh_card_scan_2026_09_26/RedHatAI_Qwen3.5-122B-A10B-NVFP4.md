---
base_model:
- Qwen/Qwen3.5-122B-A10B
tags:
- qwen
- nvfp4
- vllm
- compressed-tensors
name: RedHatAI/Qwen3.5-122B-A10B-NVFP4
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Qwen3.5-122B-A10B-NVFP4
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Qwen3NextForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) to FP4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using [vLLM](https://github.com/vllm-project/vllm).

1. **Text-Only**: Skip the vision encoder to free up memory for additional KV cache:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-NVFP4 --reasoning-parser qwen3 --language-model-only --moe_backend flashinfer_cutlass
```

2. **Multimodal (Text + Image)**: Serve with full vision support:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-NVFP4 --reasoning-parser qwen3 --moe_backend flashinfer_cutlass
```

3. **Tool Call**: Enable tool use support:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-NVFP4 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --moe_backend flashinfer_cutlass
```

4. **Multi-Token Prediction (MTP)**: For speculative decoding:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-NVFP4 --reasoning-parser qwen3 --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' --moe_backend flashinfer_cutlass
```


Send requests to the server:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/Qwen3.5-122B-A10B-NVFP4"

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```
## Creation

This model was quantized using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as shown below.

<details>
  <summary>Creation details</summary>

```python
import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: This example requires transformers >= v5

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"

# Load model.
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# No need to include mtp layers as they are not loaded
# through Qwen3_5MoeForConditionalGeneration
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.select_columns(["messages"])
ds = ds.shuffle(seed=42)


def preprocess_function(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
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
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
    data_collator=data_collator,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

# MTP layers are excluded from the model through Qwen3_5MoeForConditionalGeneration
# Save them as-is from the original checkpoint into the quantized output.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)
```
</details>



## Evaluation

The model was evaluated on the ifeval, mmlu_pro and gsm8k_platinum  using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), on reasoning tasks using [lighteval](https://github.com/neuralmagic/lighteval/tree/reasoning).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.


<details>
  <summary>Evaluation details</summary>

  **lm-evaluation-harness**
  ```
  lm_eval --model local-chat-completions \
    --tasks mmlu_pro_chat \
    --model_args "model=RedHatAI/Qwen3.5-122B-A10B-NVFP4,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=5678"
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks ifeval \
    --model_args "model=RedHatAI/Qwen3.5-122B-A10B-NVFP4,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=5678"
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks gsm8k_platinum_cot_llama \
    --model_args "model=RedHatAI/Qwen3.5-122B-A10B-NVFP4,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=5678"
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    provider: "hosted_vllm"
    model_name: "hosted_vllm/RedHatAI/Qwen3.5-122B-A10B-NVFP4"
    base_url: "http://0.0.0.0:8000/v1"
    api_key: ""
    timeout: 2400
    concurrent_requests: 128
    generation_parameters:
      temperature: 1.0
      max_new_tokens: 64000
      top_p: 0.95
      top_k: 20
      min_p: 0.0
      presence_penalty: 1.5
      repetition_penalty: 1.0
      seed: 5678
  ```

  ```
  lighteval endpoint litellm lighteval_model_arguments.yaml  \
    "aime25|0,math_500|0,gpqa:diamond|0"
  ```


</details>


## Accuracy

| Benchmark | Qwen3.5-122B-A10B | Qwen3.5-122B-A10B-NVFP4 (this model) | Recovery (%) |
|--------|-------------|-------------------|--------------|
| GSM8k Platinum (0-shot) | 95.59 | 95.37 | 99.77 |
| MMLU-Pro (0-shot) | 86.96 | 86.62 | 99.61 |
| IfEval (0-shot) | 93.80 | 93.32 | 99.49 |
| AIME 2025 | 92.92 |  91.66 | 98.65 |
| GPQA diamond | 87.54 | 86.70 | 99.04  |
| Math 500 | 84.73 | 84.80 | 100.08 |
