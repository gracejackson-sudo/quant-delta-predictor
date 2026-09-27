---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/main/LICENSE
pipeline_tag: image-text-to-text
validated_on:
  - RHOAI 3.4
  - RHAIIS 3.4
  - vLLM 0.18.0
name: RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic
description: This model was obtained by quantizing the weights and activations of Qwen/Qwen3.5-122B-A10B to FP8 data type, ready for inference with vLLM.
readme: https://huggingface.co/RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic/blob/main/README.md
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
- qwen3_5_moe
base_model: Qwen/Qwen3.5-122B-A10B
provider: Alibaba Cloud
---

# Qwen3.5-122B-A10B-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen/Qwen3.5-122B-A10B
  - **Input:** Text, Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2026-03-27
- **Version:** 1.0
- **Model Developers:** RedHatAI
- **ModelCar Storage URI:** oci://registry.redhat.io/rhai/modelcar-qwen3-5-122b-a10b-fp8-dynamic:3.0
- **Validated on vLLM:** 0.18.0
- **Validated on RHAIIS:** 3.4
- **Validated on RHOAI:** 3.4


This model is a quantized version of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) to FP8 data type, ready for inference with vLLM.

This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor). Layers such as the visual encoder, linear attention (Gated DeltaNet), MoE router gates, shared experts, and token embeddings are kept in original precision.

## Deployment

### Use with vLLM

This model can be deployed efficiently using [vLLM](https://github.com/vllm-project/vllm).

1. **Text-Only**: Skip the vision encoder to free up memory for additional KV cache:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic --reasoning-parser qwen3 --language-model-only
```

2. **Multimodal (Text + Image)**: Serve with full vision support:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic --reasoning-parser qwen3
```

3. **Tool Call**: Enable tool use support:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

4. **Multi-Token Prediction (MTP)**: For speculative decoding:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic --reasoning-parser qwen3 --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
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

model = "RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic"

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

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with FP8 dynamic quantization, as presented in the code snippet below.

<details>

```python
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.5-122B-A10B"

# Load model.
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with channel-wise quantization
#   * quantize the activations to fp8 with dynamic per-token quantization
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*mlp\\.shared_expert$",
        "re:.*linear_attn.*",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-dynamic"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

</details>

## Evaluation

This model was evaluated on GSM8K-Platinum, MMLU-Pro, IFEval, Math 500, GPQA Diamond, AIME 25, and LiveCodeBench v6 using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [lighteval](https://github.com/huggingface/lighteval), served with [vLLM](https://github.com/vllm-project/vllm) using `--language-model-only`.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Qwen/Qwen3.5-122B-A10B</th>
      <th>RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><b>Reasoning</b></td>
      <td>GSM8K-Platinum (0-shot)</td>
      <td>95.59</td>
      <td>95.70</td>
      <td>100.1%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (0-shot)</td>
      <td>86.96</td>
      <td>86.96</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>Math 500 (0-shot)</td>
      <td>85.20</td>
      <td>85.13</td>
      <td>99.9%</td>
    </tr>
    <tr>
      <td>AIME 25 (0-shot)</td>
      <td>92.08</td>
      <td>92.50</td>
      <td>100.4%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (0-shot)</td>
      <td>87.37</td>
      <td>86.03</td>
      <td>98.5%</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Instruction Following</b></td>
      <td>IFEval prompt-level strict (0-shot)</td>
      <td>92.08</td>
      <td>91.13</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>IFEval inst-level strict (0-shot)</td>
      <td>93.80</td>
      <td>93.69</td>
      <td>99.9%</td>
    </tr>
    <tr>
      <td><b>Coding</b></td>
      <td>LiveCodeBench v6 (0-shot)</td>
      <td>81.33</td>
      <td>81.90</td>
      <td>100.7%</td>
    </tr>
  </tbody>
</table>

### Reproduction

The results were obtained using the following commands:

<details>

The model was served with vLLM using the following command:

```
vllm serve RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic --reasoning-parser qwen3 --language-model-only --max-model-len 96000
```

Each benchmark was run 3 times with different seeds (42, 1234, 4158), except AIME 25 which used 8 seeds (42, 1234, 4158, 5322, 1356, 9843, 3344, 5678). Scores are averaged across all seeds.

#### lm-eval benchmarks

##### GSM8K-Platinum (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=2400" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results.json \
  --seed 42 \
  --gen_kwargs "do_sample=true,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=42"
```

##### IFEval (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=2400" \
  --apply_chat_template \
  --output_path results.json \
  --seed 42 \
  --gen_kwargs "do_sample=true,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=42"
```

##### MMLU-Pro (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results.json \
  --seed 42 \
  --gen_kwargs "do_sample=true,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=42"
```

#### lighteval benchmarks

**litellm_config.yaml:**
```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/RedHatAI/Qwen3.5-122B-A10B-FP8-dynamic"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 3600
  max_model_length: 96000
  concurrent_requests: 64
  generation_parameters:
    temperature: 1.0
    max_new_tokens: 64000
    top_p: 0.95
    top_k: 20
    min_p: 0.0
    presence_penalty: 1.5
    repetition_penalty: 1.0
    seed: 0
```

##### Math 500, GPQA Diamond, LiveCodeBench v6 (0-shot)
```
lighteval endpoint litellm litellm_config.yaml \
  "math_500|0,gpqa:diamond|0,lcb:codegeneration_v6|0" \
  --output-dir results \
  --save-details
```

##### AIME 25 (0-shot)
```
lighteval endpoint litellm litellm_config.yaml \
  "aime25|0" \
  --output-dir results \
  --save-details
```

</details>
