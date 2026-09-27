---
library_name: transformers
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- fp8
- vllm
- llm-compressor
- compressed-tensors
- qwen3.5
- code
- agent
- sft
- omnicoder
- tesslate
base_model: Tesslate/OmniCoder-9B
---

# OmniCoder-9B-FP8-Dynamic

## Model Overview
- **Model Architecture:** Qwen3_5ForConditionalGeneration
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2026-03-20
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Tesslate/OmniCoder-9B](https://huggingface.co/Tesslate/OmniCoder-9B). See the Evaluation section below for accuracy relative to the unquantized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Tesslate/OmniCoder-9B](https://huggingface.co/Tesslate/OmniCoder-9B) to FP8 data type, ready for inference with vLLM.

This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

1. Initialize vLLM server:

```
vllm serve RedHatAI/OmniCoder-9B-FP8-Dynamic --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only --max-model-len 262144
```

2. Send requests to the server:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/OmniCoder-9B-FP8-Dynamic"

messages = [
    {"role": "user", "content": "Explain the difference between a mutex and a semaphore."},
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.6,
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with FP8 dynamic (W8A8) quantization and exported in compressed-tensors format.

<details>

```python
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Tesslate/OmniCoder-9B"

# Load model and processor.
model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Configure FP8 dynamic quantization:
#   * weights: FP8 with per-channel static scales
#   * activations: FP8 with dynamic per-token scales
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "lm_head",
        "re:.*model.embed_tokens.*",
        "re:.*visual.*",
        "re:.*conv1d.*",
    ],
)

# Apply quantization (no calibration data required).
oneshot(model=model, recipe=recipe)

# Save in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-Dynamic"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

</details>

## Evaluation

This model was evaluated on GSM8K-Platinum, MMLU-Pro, IFEval, Math 500, GPQA Diamond, AIME 25, and LiveCodeBench v6 using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [lighteval](https://github.com/huggingface/lighteval), served with [vLLM](https://github.com/vllm-project/vllm). SWE-Bench Verified was evaluated using [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) and the [SWE-bench](https://github.com/SWE-bench/SWE-bench) harness.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Tesslate/OmniCoder-9B</th>
      <th>RedHatAI/OmniCoder-9B-FP8-Dynamic</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5"><b>Reasoning</b></td>
      <td>GSM8K-Platinum (0-shot)</td>
      <td>94.27</td>
      <td>93.19</td>
      <td>98.9%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (0-shot)</td>
      <td>82.42</td>
      <td>81.69</td>
      <td>99.1%</td>
    </tr>
    <tr>
      <td>Math 500 (0-shot)</td>
      <td>83.20</td>
      <td>84.47</td>
      <td>101.5%</td>
    </tr>
    <tr>
      <td>AIME 25 (0-shot)</td>
      <td>77.08</td>
      <td>74.17</td>
      <td>96.2%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (0-shot)</td>
      <td>81.99</td>
      <td>81.48</td>
      <td>99.4%</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Instruction Following</b></td>
      <td>IFEval prompt-level strict (0-shot)</td>
      <td>74.92</td>
      <td>69.19</td>
      <td>92.4%</td>
    </tr>
    <tr>
      <td>IFEval inst-level strict (0-shot)</td>
      <td>76.42</td>
      <td>70.70</td>
      <td>92.5%</td>
    </tr>
    <tr>
      <td rowspan="2"><b>Coding</b></td>
      <td>LiveCodeBench v6 (0-shot)</td>
      <td>54.10</td>
      <td>54.86</td>
      <td>101.4%</td>
    </tr>
    <tr>
      <td>SWE-Bench Verified (resolve rate)</td>
      <td>28.20</td>
      <td>30.20</td>
      <td>107.1%</td>
    </tr>
  </tbody>
</table>

### Reproduction

The results were obtained using the following commands:

<details>

The model was served with vLLM using the following command:

```
vllm serve RedHatAI/OmniCoder-9B-FP8-Dynamic --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only --max-model-len 262144
```

Each benchmark was run multiple times with different random seeds. Most tasks used 3 seeds (42, 1234, 4158). AIME 25 used 8 seeds (42, 1234, 4158, 5322, 1356, 9843, 3344, 5678). Scores are averaged across all seeds.

#### lm-eval benchmarks

##### IFEval (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=RedHatAI/OmniCoder-9B-FP8-Dynamic,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=2400" \
  --apply_chat_template \
  --output_path results_ifeval.json \
  --seed 42 \
  --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=0.0,seed=42"
```

##### MMLU-Pro (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=RedHatAI/OmniCoder-9B-FP8-Dynamic,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_mmlu_pro.json \
  --seed 42 \
  --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=0.0,seed=42"
```

##### GSM8K-Platinum (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/OmniCoder-9B-FP8-Dynamic,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_gsm8k_platinum.json \
  --seed 42 \
  --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=0.0,seed=42"
```

#### lighteval benchmarks

**litellm_config.yaml**

```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/RedHatAI/OmniCoder-9B-FP8-Dynamic"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 2400
  concurrent_requests: 256
  generation_parameters:
    temperature: 0.6
    max_new_tokens: 50000
    top_p: 0.95
    presence_penalty: 0.0
    top_k: 20
    seed: 42
```

##### Math 500, GPQA Diamond, LiveCodeBench v6 (0-shot)
```
lighteval endpoint litellm litellm_config.yaml \
  "math_500|0,gpqa:diamond|0,lcb:codegeneration_v6|0" \
  --output-dir results_lighteval \
  --save-details
```

##### AIME 25 (0-shot)
```
lighteval endpoint litellm litellm_config.yaml \
  "aime25|0" \
  --output-dir results_aime25 \
  --save-details
```

#### SWE-Bench Verified

SWE-Bench Verified was evaluated with [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) for agent rollouts against the vLLM server, and scored with the [SWE-bench](https://github.com/SWE-bench/SWE-bench) evaluation harness.

**registry.yaml**

```yaml
{
  "RedHatAI/OmniCoder-9B-FP8-Dynamic": {
    "max_tokens": 262144,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "litellm_provider": "hosted_vllm",
    "mode": "chat"
  }
}
```

Set the model endpoint in `swebench.yaml`:

```yaml
model:
  model_name: "hosted_vllm/RedHatAI/OmniCoder-9B-FP8-Dynamic"
  model_kwargs:
    api_base: "http://0.0.0.0:8100/v1"
    api_key: ""
    temperature: 0.2
    top_p: 0.95
    presence_penalty: 0.0
    top_k: 20
    max_new_tokens: 240000
```

Run agent rollouts:

```
LITELLM_MODEL_REGISTRY_PATH=registry.yaml \
mini-extra swebench \
  --subset inference-optimization/SWE-bench_Verified \
  --split test \
  --config swebench.yaml \
  --workers 64 \
  --output verified_swe_instances
```

Score predictions with the SWE-bench harness:

```
python -m swebench.harness.run_evaluation \
  --dataset_name inference-optimization/SWE-bench_Verified \
  --predictions_path ./verified_swe_instances/preds.json \
  --max_workers 8 \
  --run_id validate-verified_swe_instances \
  --cache_level instance
```

</details>
