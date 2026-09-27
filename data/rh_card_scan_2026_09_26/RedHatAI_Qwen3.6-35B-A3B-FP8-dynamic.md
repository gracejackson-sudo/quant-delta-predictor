---
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
- qwen3_6_moe
base_model: Qwen/Qwen3.6-35B-A3B
name: RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic
license: apache-2.0
license_name: apache-2.0
description: 35B Qwen 3.6 MOE model with improved reasoning, tool calling, and VL capabilities, quantized to FP8. 
readme: https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic/main/README.md
license_link: https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic/blob/main/LICENSE
provider: Alibaba Cloud
tool_calling_supported: true
required_cli_args: ['--reasoning-parser qwen3', '--enable-prefix-caching']
default-chat-template-kwargs: '{"enable_thinking": true}'
chat_template_file_name: None
chat_template_path: None
tool_call_parser: qwen3_coder
validated_tasks:
- tool-calling
validated_on:
  - RHOAI 3.4
  - RHAIIS 3.4
  - vLLM 0.18.0
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Qwen3.6-35B-A3B-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Qwen/Qwen3.6-35B-A3B
  - **Input:** Text, Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2026-06-25
- **Version:** 1.0
- **Model Developers:** RedHatAI
- **ModelCar Storage URI:** oci://registry.redhat.io/rhai/modelcar-qwen3-5-35b-a3b-fp8-dynamic:3.0
- **Validated on vLLM:** 0.18.0
- **Validated on RHAIIS:** 3.4
- **Validated on RHOAI:** 3.4


This model is a quantized version of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) to FP8 data type, ready for inference with vLLM.

This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor). Layers such as the visual encoder, linear attention (Gated DeltaNet), MoE router gates, shared experts, and token embeddings are kept in original precision.

## Deployment

### Use with vLLM

This model can be deployed efficiently using [vLLM](https://github.com/vllm-project/vllm).

1. **Text-Only**: Skip the vision encoder to free up memory for additional KV cache:

```
vllm serve RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic --reasoning-parser qwen3 --language-model-only
```

2. **Multimodal (Text + Image)**: Serve with full vision support:

```
vllm serve RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic --reasoning-parser qwen3
```

3. **Tool Call**: Enable tool use support:

```
vllm serve RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

4. **Multi-Token Prediction (MTP)**: For speculative decoding:

```
vllm serve RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic --reasoning-parser qwen3 --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
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

model = "RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic"

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
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

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

| Benchmark | Qwen/Qwen3.6-35B-A3B | RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic | Recovery (%) |
|-----------|-------------------|-------------------------------|--------------|
| GSM8k Platinum  | 95.73 | 95.62 | 99.88 |
| IfEval  | 93.09 | 92.69 | 99.57 |
| AIME 2025 | 92.92 | 92.50 | 99.55 |
| GPQA diamond | 84.51 | 84.85 | 100.40 |
| Math 500 | 84.80 | 84.73 | 99.92 |
| Lcb Codegeneration V6 | 77.33 | 78.86 | 101.97 |
| MMLU Pro Chat | 85.32 | 85.36 | 100.05 |
| BFCLv4 Overall | 57.83 | 57.94 | 100.19 |
| BFCLv4 Single Turn | 53.81  | 53.33 | 99.11  |
| BFCLv4 Multi-Turn | 62.250 | 60.880  | 97.80  |
| BFCLv4 Agentic | 49.910  | 51.310  | 102.81 |
| SWEBench Verified | 54.8 | 52 | 94.89 |

### Reproduction

The results were obtained using the following commands:

<details>

The model was served with vLLM using the following command:

```
vllm serve RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic --reasoning-parser qwen3 --language-model-only --max-model-len 96000
```

Each benchmark was run 3 times with different seeds (42, 1234, 4158), except AIME 25 which used 8 seeds (42, 1234, 4158, 5322, 1356, 9843, 3344, 5678). Scores are averaged across all seeds.

#### lm-eval benchmarks

##### GSM8K-Platinum (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=2400" \
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
  --model_args "model=RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=2400" \
  --apply_chat_template \
  --output_path results.json \
  --seed 42 \
  --gen_kwargs "do_sample=true,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=42"
```

##### MMLU-Pro (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
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
  model_name: "hosted_vllm/RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 2400
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

#### BFCLv4

BFCL requires the model to be registered in the leaderboard codebase before running evaluation.

**Step 1 — Register the model in `bfcl_eval/constants/model_config.py`**

Add the following entry to `api_inference_model_map`:

```python
"Qwen3.6-35B-A3B-FP8-dynamic": ModelConfig(
    model_name="Qwen3.6-35B-A3B-FP8-dynamic",
    display_name="Qwen3.6-35B-A3B-FP8-dynamic (FC)",
    url="https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic",
    org="Google",
    license="Apache 2.0",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,
),
```

**Step 2 — Add the key to `bfcl_eval/constants/supported_models.py`**

Add `"Qwen3.6-35B-A3B-FP8-dynamic"` to the `SUPPORTED_MODELS` list.

**Step 3 — Start the vLLM server** (use the command at the top of this section; the `--served-model-name` flag ensures BFCL can find the model by its registered slug).

**Step 4 — Generate responses and evaluate**
```
bfcl generate --model Qwen3.6-35B-A3B-FP8-dynamic --test-category all
bfcl evaluate --model Qwen3.6-35B-A3B-FP8-dynamic --test-category all
```

</details>
