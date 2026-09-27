---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE
pipeline_tag: image-text-to-text
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
base_model: Qwen/Qwen3.5-4B
---

# Qwen3.5-4B-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen/Qwen3.5-4B
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
  - **Model size:** 8.0 GB (reduced from 9.3 GB in BF16)
- **Release Date:** 2026-05-11
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B). Evaluation results and reproduction steps are provided below.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) to FP8 data type, ready for inference with vLLM.

This optimization reduces the model weights from 9.3 GB to 8.0 GB on disk (~14% reduction). Activations are quantized dynamically at inference time using per-tensor scaling, requiring no calibration data.

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

1. Initialize vLLM server:

**Multimodal (vision + text):**
```bash
vllm serve RedHatAI/Qwen3.5-4B-FP8-dynamic \
  --reasoning-parser qwen3 \
  --max-model-len 262144
```

**Text-only (lower memory):**
```bash
vllm serve RedHatAI/Qwen3.5-4B-FP8-dynamic \
  --reasoning-parser qwen3 \
  --max-model-len 262144 \
  --language-model-only
```

2. Send requests to the server:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/Qwen3.5-4B-FP8-dynamic"

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

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) using data-free FP8 dynamic quantization, as presented in the code snippet below.

<details>

```python
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

MODEL_ID = "Qwen/Qwen3.5-4B"

IGNORE_LAYERS = [
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*visual.*",
    "re:.*model.visual.*",
    "re:.*linear_attn.*",
]

model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=IGNORE_LAYERS,
)

oneshot(model=model, recipe=recipe)

model.save_pretrained("Qwen3.5-4B-FP8-dynamic", save_compressed=True)
processor.save_pretrained("Qwen3.5-4B-FP8-dynamic")
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir="Qwen3.5-4B-FP8-dynamic")
```

<details>
<summary>Package versions</summary>

- `llm-compressor==0.10.1.dev44+g437f8afe`
- `compressed-tensors==0.14.1a20260325`
- `transformers==5.3.0`
- `vllm==0.18.1`
- `lm-eval` — `neuralmagic/lm-evaluation-harness@741f1d8` (branch: `mmlu-pro-chat-variant`)
- `lighteval` — `neuralmagic/lighteval@6f0f351` (branch: `eldar-fix-litellm`)

</details>

</details>

## Evaluation

This model was evaluated on GSM8k-Platinum, MMLU-Pro, IFEval, Math 500, AIME 2025, and GPQA Diamond using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [lighteval](https://github.com/huggingface/lighteval), with inference served via vLLM.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Qwen/Qwen3.5-4B</th>
      <th>RedHatAI/Qwen3.5-4B-FP8-dynamic</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><b>Instruction Following</b></td>
      <td>GSM8k-Platinum (0-shot)</td>
      <td>94.2%</td>
      <td>94.5%</td>
      <td>100.3%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (0-shot)</td>
      <td>79.3%</td>
      <td>79.1%</td>
      <td>99.9%</td>
    </tr>
    <tr>
      <td>IFEval — prompt strict (0-shot)</td>
      <td>88.0%</td>
      <td>88.4%</td>
      <td>100.5%</td>
    </tr>
    <tr>
      <td>IFEval — instruction strict (0-shot)</td>
      <td>91.2%</td>
      <td>91.6%</td>
      <td>100.4%</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Reasoning</b></td>
      <td>Math 500 (0-shot)</td>
      <td>84.6%</td>
      <td>84.7%</td>
      <td>100.2%</td>
    </tr>
    <tr>
      <td>AIME 2025 (0-shot)</td>
      <td>85.0%</td>
      <td>85.0%</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (0-shot)</td>
      <td>76.8%</td>
      <td>76.3%</td>
      <td>99.3%</td>
    </tr>
  </tbody>
</table>

### Reproduction

The results were obtained using the following commands. GSM8k-Platinum, MMLU-Pro, IFEval, Math 500, and GPQA Diamond were each run 3 times with different seeds and results averaged. AIME 2025 was run 8 times. The vLLM server was started with `--language-model-only` for all evaluations.

<details>

#### GSM8k-Platinum (lm-eval, 0-shot, 3 repetitions)
```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Qwen3.5-4B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=100,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_gsm8k_platinum.json \
  --seed <SEED> \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,max_gen_toks=65536,seed=<SEED>"
```
Seeds used: 42, 1234, 4158

#### MMLU-Pro (lm-eval, 0-shot, 3 repetitions)
```bash
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=RedHatAI/Qwen3.5-4B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=100,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_mmlu_pro.json \
  --seed <SEED> \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,max_gen_toks=65536,seed=<SEED>"
```
Seeds used: 42, 1234, 4158

#### IFEval (lm-eval, 0-shot, 3 repetitions)
```bash
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=RedHatAI/Qwen3.5-4B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=100,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_ifeval.json \
  --seed <SEED> \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,presence_penalty=1.5,repetition_penalty=1.0,max_gen_toks=65536,seed=<SEED>"
```
Seeds used: 42, 1234, 4158

#### Math 500 (lighteval, 0-shot, 3 repetitions)
```bash
lighteval endpoint litellm \
  "model_name=hosted_vllm/RedHatAI/Qwen3.5-4B-FP8-dynamic,provider=hosted_vllm,base_url=http://0.0.0.0:8000/v1,timeout=3600,concurrent_requests=100,generation_parameters={temperature:1.0,max_new_tokens:65536,top_p:0.95,top_k:20,min_p:0.0,presence_penalty:1.5,repetition_penalty:1.0,seed:<SEED>}" \
  "math_500@k=1@n=1|0" \
  --output-dir results_math500 \
  --save-details
```
Seeds used: 42, 1234, 4158

#### AIME 2025 (lighteval, 0-shot, 8 repetitions)
```bash
lighteval endpoint litellm \
  "model_name=hosted_vllm/RedHatAI/Qwen3.5-4B-FP8-dynamic,provider=hosted_vllm,base_url=http://0.0.0.0:8000/v1,timeout=3600,concurrent_requests=100,generation_parameters={temperature:1.0,max_new_tokens:65536,top_p:0.95,top_k:20,min_p:0.0,presence_penalty:1.5,repetition_penalty:1.0,seed:<SEED>}" \
  "aime25@k=1@n=1|0" \
  --output-dir results_aime25 \
  --save-details
```
Seeds used: 42, 1234, 1356, 3344, 4158, 5322, 5678, 9843

#### GPQA Diamond (lighteval, 0-shot, 3 repetitions)
```bash
lighteval endpoint litellm \
  "model_name=hosted_vllm/RedHatAI/Qwen3.5-4B-FP8-dynamic,provider=hosted_vllm,base_url=http://0.0.0.0:8000/v1,timeout=3600,concurrent_requests=100,generation_parameters={temperature:1.0,max_new_tokens:65536,top_p:0.95,top_k:20,min_p:0.0,presence_penalty:1.5,repetition_penalty:1.0,seed:<SEED>}" \
  "gpqa:diamond@k=1@n=1|0" \
  --output-dir results_gpqa_diamond \
  --save-details
```
Seeds used: 42, 1234, 4158

</details>
