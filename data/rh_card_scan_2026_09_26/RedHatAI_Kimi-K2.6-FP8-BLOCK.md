---
base_model: moonshotai/Kimi-K2.6
license: other
license_name: modified-mit
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- kimi
- fp8
- vllm
- llm-compressor
- compressed-tensors
name: RedHatAI/Kimi-K2.6-FP8-BLOCK
---

# RedHatAI/Kimi-K2.6-FP8-BLOCK

## Model Overview
- **Model Architecture:** moonshotai/Kimi-K2.6 (`KimiK25ForConditionalGeneration`)
- **Input:** Text, image, and video
- **Output:** Text
- **Weight Quantization:** FP8 (block-wise scaling)
- **Activation Quantization:** FP8 (dynamic grouped scaling)
- **Release Date:** 2026-04-29
- **Model Developers:** RedHatAI

This model is a quantized variant of [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6), exported in compressed-tensors format for vLLM deployment and evaluated on instruction-following, reasoning, function-calling, and agentic coding workloads.

### Model Optimizations

This checkpoint applies FP8 block quantization to transformer linear layers and FP8 dynamic quantization to activations. The resulting representation is optimized for high-throughput serving while maintaining strong benchmark retention on Kimi-K2.6 evaluation suites.

The model is exported in compressed-tensors format and is intended for OpenAI-compatible inference with vLLM.

## Creation

This model was quantized with [LLM Compressor](https://github.com/vllm-project/llm-compressor) and exported as compressed-tensors. The script below is a representative reference script aligned with the published quantization configuration.

<details>
<summary><b>Reference quantization script (FP8 block)</b></summary>

```python
from compressed_tensors.entrypoints.convert import CompressedTensorsDequantizer
from llmcompressor import model_free_ptq

MODEL_ID = "moonshotai/Kimi-K2.6"
SAVE_DIR = "Kimi-K2.6-FP8-BLOCK"

ignore = [
    "re:.*mlp.gate$",
    "re:.*lm_head",
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
    scheme="FP8_BLOCK",
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
vllm serve RedHatAI/Kimi-K2.6-FP8-BLOCK \
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
    model="RedHatAI/Kimi-K2.6-FP8-BLOCK",
    messages=[{"role": "user", "content": "Explain how transformers use attention."}],
)

print(resp.choices[0].message.content)
```

## Evaluation

We evaluated this model with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [lighteval](https://github.com/huggingface/lighteval), BFCL v4, and SWE-Bench Lite served through a vLLM (`0.22.1`) OpenAI-compatible endpoint.

| Category | Benchmark | Score |
| --- | --- | ---: |
| Reasoning and instruction following | AIME25 (pass@1, avg@8) | 96.25% |
| Reasoning and instruction following | GPQA Diamond (pass@1, avg@3) | 89.39% |
| Reasoning and instruction following | MATH-500 (pass@1, avg@3) | 94.27% |
| Reasoning and instruction following | MMLU-Pro Chat (custom-extract, avg@3) | 86.55% |
| Reasoning and instruction following | GSM8K Platinum CoT (strict-match, avg@3) | 93.13% |
| Reasoning and instruction following | GSM8K Platinum CoT (flexible-extract, avg@3) | 96.31% |
| Reasoning and instruction following | IFEval (prompt-level strict, avg@3) | 95.63% |
| Reasoning and instruction following | IFEval (instruction-level strict, avg@3) | 96.92% |
| Agentic function calling (accuracy) | BFCL v4 non_live | 86.58% |
| Agentic function calling (accuracy) | BFCL v4 live | 77.79% |
| Agentic function calling (accuracy) | BFCL v4 multi_turn | 61.62% |
| Agentic function calling (accuracy) | BFCL v4 memory | 61.72% |
| Agentic function calling (accuracy) | BFCL v4 web_search | 14.00% |
| Agentic coding | SWE-Bench Lite (dev) | 34.78% |

BFCL rows report category accuracy. SWE-Bench follows the official harness score style. For run transparency: 8 of 23 tasks were resolved, and 19 instances produced non-empty graded patches.

### Recovery vs. base model (`moonshotai/Kimi-K2.6`)

| Benchmark | Base model (`moonshotai/Kimi-K2.6`) | This model | Recovery |
| --- | ---: | ---: | ---: |
| AIME25 (pass@1, avg@8) | 90.00% | 96.25% | 106.94% |
| GPQA Diamond (pass@1, avg@3) | 84.51% | 89.39% | 105.77% |
| MATH-500 (pass@1, avg@3) | 93.53% | 94.27% | 100.79% |
| MMLU-Pro Chat (custom-extract, avg@3) | 86.70% | 86.55% | 99.83% |
| GSM8K Platinum CoT (strict-match, avg@3) | 93.80% | 93.13% | 99.29% |
| GSM8K Platinum CoT (flexible-extract, avg@3) | 96.33% | 96.31% | 99.98% |
| IFEval (prompt-level strict, avg@3) | 94.82% | 95.63% | 100.85% |
| IFEval (instruction-level strict, avg@3) | 96.52% | 96.92% | 100.41% |

## Reproduction

Representative commands used to produce and aggregate these runs:

### vLLM + lm-eval (example)

```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Kimi-K2.6-FP8-BLOCK,max_length=40960,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
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
  model_name: "hosted_vllm/RedHatAI/Kimi-K2.6-FP8-BLOCK"
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
./scripts/bfcl/run_bfcl_local.sh kimi_fp8 non_live
./scripts/bfcl/run_bfcl_local.sh kimi_fp8 live
./scripts/bfcl/run_bfcl_local.sh kimi_fp8 multi_turn
./scripts/bfcl/run_bfcl_local.sh kimi_fp8 memory
./scripts/bfcl/run_bfcl_local.sh kimi_fp8 web_search
```

```bash
# SWE-Bench Lite dev (full split)
SWEBENCH_SUBSET=lite SWEBENCH_SPLIT=dev SWEBENCH_SLICE= \
  ./scripts/swebench/run_swebench_lite_local.sh kimi_fp8

# Official SWE-bench resolved-rate evaluation
/home/shubhra/environments/mini-swe-agent/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-Bench_Lite \
  --split dev \
  --predictions_path /home/shubhra/kimik2.6_evals/results/swebench_resolved_eval/kimi_fp8_lite_dev_preds_merged.json \
  --max_workers 4 \
  --run_id kimi_fp8_lite_dev_20260701_resolved
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
