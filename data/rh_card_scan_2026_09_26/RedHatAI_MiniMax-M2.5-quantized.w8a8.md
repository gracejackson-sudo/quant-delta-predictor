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
- INT8
---

# MiniMax-M2.5-quantized.w8a8

## Model Overview
- **Model Architecture:** MiniMaxM2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
- **Intended Use Cases:**
  - Reasoning.
  - Function calling.
  - Subject matter experts via fine-tuning.
  - Multilingual instruction following.
  - Translation.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 04/29/2026
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights of [MiniMax-M2.5](https://huggingface.co/MiniMaxAI/MiniMax-M2.5) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
A combination of the [SmoothQuant](https://arxiv.org/abs/2211.10438) and [GPTQ](https://arxiv.org/abs/2210.17323) algorithms is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/MiniMax-M2.5-quantized.w8a8"
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
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "RedHatAI/MiniMax-M2.5-BF16"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)

NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load dataset.
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

# Preprocess the data into the format the model is trained with.
def preprocess(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

ds = ds.map(preprocess)

# Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
def tokenize(sample):
    return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
ds = ds.map(tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm to run.
recipe = GPTQModifier( scheme="W8A8", weight_observer="mse", targets= [r"re:.*block_sparse_moe\.experts\.\d+\.w[1-3]$", r"re:.*mlp\.experts\.\d+\.(gate|up|gate_up|down)_proj$" ], ignore=["re:.*self_attn.*", "lm_head"])


# Apply quantization.
oneshot(
    model=model, dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    processor=processor
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + ".w8a8"
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
    vllm serve RedHatAI/MiniMax-M2.5.w8a8 --max-model-len 262144 --reasoning-parser deepseek_r1
    ```

  **lm-evaluation-harness**
  ```
  lm_eval --model local-chat-completions \
    --tasks mmlu_pro_chat \
    --model_args "model=RedHatAI/MiniMax-M2.5.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks ifeval \
    --model_args "model=RedHatAI/MiniMax-M2.5.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks gsm8k_platinum_cot_llama \
    --model_args "model=RedHatAI/MiniMax-M2.5.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=40,min_p=0.0,max_gen_toks=64000
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/MiniMax-M2.5.w8a8
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

| Benchmark | RedHatAI/MiniMax-M2.5-BF16 | RedHatAI/MiniMax-M2.5.w8a8 | Recovery (%) |
|-----------|------------------------------------------|------------------------------------------|--------------|
| GSM8k Platinum (0-shot) | 95.15 | 95.18 | 100.03 |
| IfEval (0-shot) | 92.05 | 90.33 | 98.13 |
| AIME 2025 | 87.50 | 88.33 | 100.95 |
| GPQA diamond | 83.67 | 84.51 | 101.01 |
| Math 500 | 87.33 | 87.13 | 99.77 |
| MMLU Pro Chat | 80.83 | 81.25 | 100.51 |

