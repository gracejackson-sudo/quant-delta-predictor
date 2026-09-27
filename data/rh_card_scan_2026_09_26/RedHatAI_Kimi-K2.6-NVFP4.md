---
base_model: moonshotai/Kimi-K2.6
license: other
license_name: modified-mit
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- kimi
- fp4
- nvfp4
- vllm
- llm-compressor
- compressed-tensors
name: RedHatAI/Kimi-K2.6-NVFP4
---

# RedHatAI/Kimi-K2.6-NVFP4

## Model Overview
- **Model Architecture:** moonshotai/Kimi-K2.6 (`KimiK25ForConditionalGeneration`)
- **Input:** Text, image, and video
- **Output:** Text
- **Weight Quantization:** NVFP4 (FP4 tensor-group quantization)
- **Activation Quantization:** NVFP4 (FP4 tensor-group quantization)
- **Release Date:** 2026-04-30
- **Model Developers:** RedHatAI

This model is a quantized variant of [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6), exported in compressed-tensors format for vLLM deployment and evaluated on instruction-following, reasoning, function-calling, and agentic coding workloads.

### Model Optimizations

This checkpoint applies NVFP4 quantization to transformer linear layers with group-wise FP4 weights and activations, using FP8 scale tensors. The format is optimized for efficient low-precision serving while preserving strong benchmark quality on Kimi-K2.6 evaluations.

The model is exported in compressed-tensors format and is intended for OpenAI-compatible inference with vLLM.

## Creation

This model was quantized with [LLM Compressor](https://github.com/vllm-project/llm-compressor) and exported as compressed-tensors. The script below is a representative reference script aligned with `recipe.yaml` and the published quantization configuration.

<details>
<summary><b>Reference quantization script (NVFP4)</b></summary>

```python
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer
from llmcompressor import model_free_ptq

MODEL_ID = "moonshotai/Kimi-K2.6"
SAVE_DIR = "Kimi-K2.6-NVFP4"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
    "re:.*self_attn.*",
    "re:.*kv_a_proj_with_mqa$",
    "re:.*q_a_proj$",
    "re:.*vision_tower.*",
    "re:.*embed_tokens$",
    "re:.*norm$",
    "re:.*mm_projector.*",
    "re:.*vision.*",
]

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="NVFP4",
    ignore=ignore,
    converter=CompressedTensorsDequantizer(
        MODEL_ID,
        quant_config_key="text_config.quantization_config",
        ignore=ignore,
    ),
    max_workers=2,
    device="cuda:0",
)
```

</details>

## Deployment

### Use with vLLM

```bash
vllm serve RedHatAI/Kimi-K2.6-NVFP4 \
  --trust-remote-code \
  --mm-encoder-tp-mode data \
  --tool-call-parser kimi_k2 \
  --reasoning-parser kimi_k2 \
  --enable-auto-tool-choice
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="RedHatAI/Kimi-K2.6-NVFP4",
    messages=[{"role": "user", "content": "Explain how transformers use attention."}],
)

print(resp.choices[0].message.content)
```

## Evaluation

We evaluated this model with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [lighteval](https://github.com/huggingface/lighteval), BFCL v4, and SWE-Bench Lite served through a vLLM (`0.22.1`) OpenAI-compatible endpoint.

| Category | Benchmark | Score |
| --- | --- | ---: |
| Reasoning and instruction following | AIME25 (pass@1, avg@8) | 96.25% |
| Reasoning and instruction following | GPQA Diamond (pass@1, avg@3) | 91.08% |
| Reasoning and instruction following | MATH-500 (pass@1, avg@3) | 93.13% |
| Reasoning and instruction following | MMLU-Pro Chat (custom-extract, avg@3) | 86.75% |
| Reasoning and instruction following | GSM8K Platinum CoT (strict-match, avg@3) | 92.50% |
| Reasoning and instruction following | GSM8K Platinum CoT (flexible-extract, avg@3) | 96.94% |
| Reasoning and instruction following | IFEval (prompt-level strict, avg@3) | 94.02% |
| Reasoning and instruction following | IFEval (instruction-level strict, avg@3) | 95.96% |
| Agentic function calling (accuracy) | BFCL v4 non_live | 86.44% |
| Agentic function calling (accuracy) | BFCL v4 live | 78.31% |
| Agentic function calling (accuracy) | BFCL v4 multi_turn | 63.75% |
| Agentic function calling (accuracy) | BFCL v4 memory | 63.23% |
| Agentic function calling (accuracy) | BFCL v4 web_search | 13.00% |
| Agentic coding | SWE-Bench Lite (dev) | 21.74% |

BFCL rows report category accuracy. SWE-Bench follows the official harness score style. For run transparency: 5 of 23 tasks were resolved, and 19 instances produced non-empty graded patches.

### Recovery vs. base model (`moonshotai/Kimi-K2.6`)

| Benchmark | Base model (`moonshotai/Kimi-K2.6`) | This model | Recovery |
| --- | ---: | ---: | ---: |
| AIME25 (pass@1, avg@8) | 90.00% | 96.25% | 106.94% |
| GPQA Diamond (pass@1, avg@3) | 84.51% | 91.08% | 107.77% |
| MATH-500 (pass@1, avg@3) | 93.53% | 93.13% | 99.57% |
| MMLU-Pro Chat (custom-extract, avg@3) | 86.70% | 86.75% | 100.06% |
| GSM8K Platinum CoT (strict-match, avg@3) | 93.80% | 92.50% | 98.61% |
| GSM8K Platinum CoT (flexible-extract, avg@3) | 96.33% | 96.94% | 100.63% |
| IFEval (prompt-level strict, avg@3) | 94.82% | 94.02% | 99.16% |
| IFEval (instruction-level strict, avg@3) | 96.52% | 95.96% | 99.42% |

## Reproduction

Representative commands used to produce and aggregate these runs:

### vLLM + lm-eval (example)

```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Kimi-K2.6-NVFP4,max_length=40960,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_gsm8k_platinum.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=1234"
```

### lighteval config used

```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/RedHatAI/Kimi-K2.6-NVFP4"
  base_url: "http://127.0.0.1:8000/v1"
  api_key: "EMPTY"
  timeout: 3600
  max_model_length: 40960
  concurrent_requests: 8
  generation_parameters:
    temperature: 1.0
    max_new_tokens: 65536
    top_p: 0.95
    seed: 1234
    top_k: 20
    presence_penalty: 1.5
```

```bash
lighteval endpoint litellm litellm_config.yaml \
  "aime25@1@8|0,math_500@1@3|0,gpqa:diamond@1@3|0" \
  --output-dir results_lighteval \
  --save-details
```

### BFCL v4 and SWE-Bench Lite scripts

```bash
# BFCL categories: non_live, live, multi_turn, memory, web_search
./scripts/bfcl/run_bfcl_local.sh kimi_nvfp4 non_live
./scripts/bfcl/run_bfcl_local.sh kimi_nvfp4 live
./scripts/bfcl/run_bfcl_local.sh kimi_nvfp4 multi_turn
./scripts/bfcl/run_bfcl_local.sh kimi_nvfp4 memory
./scripts/bfcl/run_bfcl_local.sh kimi_nvfp4 web_search
```

```bash
# SWE-Bench Lite dev (full split)
SWEBENCH_SUBSET=lite SWEBENCH_SPLIT=dev SWEBENCH_SLICE= \
  ./scripts/swebench/run_swebench_lite_local.sh kimi_nvfp4

# Official SWE-bench resolved-rate evaluation
/home/shubhra/environments/mini-swe-agent/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Lite \
  --split dev \
  --predictions_path /home/shubhra/kimik2.6_evals/results/swebench_resolved_eval/kimi_nvfp4_lite_dev_preds_merged.json \
  --max_workers 4 \
  --run_id kimi_nvfp4_lite_dev_20260701_resolved
```

Most lm-eval/lighteval tasks were run with 3 seeds and then averaged; AIME25 was run with 8 seeds. BFCL v4 and SWE-Bench Lite numbers come from the aggregated run artifacts listed below.

## Every Eval Ever Artifacts

- `every_eval_ever/aime25.json`
- `every_eval_ever/gpqa_diamond.json`
- `every_eval_ever/gsm8k_platinum_cot_llama.json`
- `every_eval_ever/ifeval.json`
- `every_eval_ever/math_500.json`
- `every_eval_ever/mmlu_pro_chat.json`
- `every_eval_ever/bfcl_v4.json`
- `every_eval_ever/swebench_lite_dev.json`
