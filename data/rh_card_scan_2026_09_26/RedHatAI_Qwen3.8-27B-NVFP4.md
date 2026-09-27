---
tags:
- fp4
- fp8
- vllm
- llm-compressor
- compressed-tensors
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-NVFP4

## Model Overview
- **Model Architecture:** Qwen3_5ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4 and FP8
  - **Activation quantization:** FP4 and FP8
- **Release Date:** 2026-09-21
- **Version:** 2.0
- **Model Developers:** RedHatAI

This model is an updated quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), using a mixed-precision FP4/FP8 scheme with an unquantized language-model head and updated quantization scales. See [Evaluation](#evaluation) for accuracy results.

### Model Optimizations

This model was produced by applying mixed-precision quantization to [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). MLP projections are quantized to FP4, attention projections and the final MLP layers are quantized to FP8, and the KV cache is quantized to FP8, while the language-model head is kept in full precision to preserve output quality. The quantization scales were updated by calibrating on a 512-sample subset of the perfectblend dataset with a recipe that combines AWQ and GPTQ.

Only the weights and activations of the linear operators within the transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor). The checkpoint is ~24.7 GB on disk (versus ~54 GB in BF16), reducing disk size and GPU memory requirements by roughly 70%.

## Deployment

### vLLM Serving

```
vllm serve RedHatAI/Qwen3.8-27B-NVFP4 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
```

For optimal peformance, consider using the DSpark draft model [RedHatAI/Qwen3.8-27B-speculator.dspark](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) for speculative decoding, shown below.

```
vllm serve RedHatAI/Qwen3.8-27B-NVFP4 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --speculative-config '{"model":"RedHatAI/Qwen3.8-27B-speculator.dspark","num_speculative_tokens":8,"method":"dspark"}'
```

See <b>Performance Evaluation</b> below for further details.

## Creation

This model was created by applying [LLM Compressor](https://github.com/vllm-project/llm-compressor) with calibration samples from perfectblend, as presented in the code snippet below.

<details>

```python
from compressed_tensors.quantization.quant_scheme import (
    FP8_DYNAMIC,
    NVFP4,
    QuantizationScheme,
)
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.transform.awq import AWQModifier
from llmcompressor.utils import load_context

MODEL_ID = "Qwen/Qwen3.8-27B"

# Load model.
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID)
processor = AutoProcessor.from_pretrained(MODEL_ID)


recipe = [
    AWQModifier(duo_scaling="both"),
    GPTQModifier(
        config_groups={
            "attention": QuantizationScheme(
                targets=[
                    r"re:.*self_attn\.(q|k|v|o)_proj$",
                    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
                    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\..*(gate|up|down)_proj$",
                ],
                **FP8_DYNAMIC,
            ),
            "mlp": QuantizationScheme(
                targets=[r"re:.*mlp\..*(gate|up|down)_proj$"],
                **NVFP4,
            ),
        },
        ignore=[
            "re:visual.*",
            "re:model.visual.*",
            "re:.*lm_head",
        ],
        kv_cache_scheme={
            "num_bits": 8,
            "type": "float",
            "symmetric": True,
            "strategy": "tensor",
            "dynamic": False,
            "observer": "static_minmax",
        },
    ),
]

# Apply quantization.
oneshot(
    model=model,
    processor=processor,
    recipe=recipe,
    dataset="perfectblend",
    splits="train[:512]",
    max_seq_length=4096,
    num_calibration_samples=512,
    moe_calibrate_all_experts=True,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4-GPTQ-AWQ"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```

</details>

## Evaluation

This model was evaluated on GSM8K Platinum, MATH-500, AIME 2025, GPQA Diamond, and IFEval using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness) and [lighteval](https://github.com/neuralmagic/lighteval), and on SWE Bench using [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai), served with vLLM (OpenAI-compatible API). Evaluations were run on 1x B200 GPU.

### Accuracy

Recovery vs. BF16 baseline
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>Qwen/Qwen3.8-27B</th>
      <th>RedHatAI/Qwen3.8-27B-NVFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><b>Reasoning</b></td>
      <td>GSM8K Platinum</td>
      <td>96.25%</td>
      <td>96.72%</td>
      <td>100.49%</td>
    </tr>
    <tr>
      <td>MATH-500</td>
      <td>83.67%</td>
      <td>84.27%</td>
      <td>100.72%</td>
    </tr>
    <tr>
      <td>AIME 2025</td>
      <td>96.67%</td>
      <td>95.00%</td>
      <td>98.27%</td>
    </tr>
    <tr>
      <td>GPQA Diamond</td>
      <td>89.56%</td>
      <td>89.22%</td>
      <td>99.62%</td>
    </tr>
    <tr>
      <td><b>Instruction Following</b></td>
      <td>IFEval</td>
      <td>91.19%</td>
      <td>91.99%</td>
      <td>100.88%</td>
    </tr>
    <tr>
      <td><b>Agentic - Coding</b></td>
      <td>SWE Bench</td>
      <td>78.8%</td>
      <td>78.0%</td>
      <td>98.98%</td>
    </tr>
  </tbody>
</table>
NVFP4 build comparison
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>RedHatAI/Qwen3.8-27B-NVFP4</th>
      <th>unsloth/Qwen3.8-27B-NVFP4</th>
      <th>Inferact/Qwen3.8-27B-NVFP4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><b>Reasoning</b></td>
      <td>GSM8K Platinum</td>
      <td>96.72%</td>
      <td>95.42%</td>
      <td>93.77%</td>
    </tr>
    <tr>
      <td>MATH-500</td>
      <td>84.27%</td>
      <td>85.67%</td>
      <td>82.47%</td>
    </tr>
    <tr>
      <td>AIME 2025</td>
      <td>95.00%</td>
      <td>93.75%</td>
      <td>91.66%</td>
    </tr>
    <tr>
      <td>GPQA Diamond</td>
      <td>89.22%</td>
      <td>89.39%</td>
      <td>87.04%</td>
    </tr>
    <tr>
      <td><b>Instruction Following</b></td>
      <td>IFEval</td>
      <td>91.99%</td>
      <td>91.81%</td>
      <td>91.50%</td>
    </tr>
  </tbody>
</table>

## Reproduction

The results were obtained using the following commands. Each benchmark was run multiple times with different random seeds — 3 repetitions for GSM8K Platinum, MATH-500, GPQA Diamond, and IFEval, and 8 repetitions for AIME 2025 — and the reported score is the mean across seeds.
<details>
  
GSM8K Platinum & IFEval (lm-eval, 0-shot)

Run once per seed:

```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Qwen3.8-27B-NVFP4,max_length=69632,base_url=http://127.0.0.1:3235/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_gsm8k_platinum.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,max_gen_toks=32000,seed=1234"
```
```
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=RedHatAI/Qwen3.8-27B-NVFP4,max_length=69632,base_url=http://127.0.0.1:3235/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_ifeval.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=20,max_gen_toks=32000,seed=1234"
```
MATH-500, AIME 2025, GPQA Diamond (lighteval, 0-shot)

litellm_config.yaml:
```
model_parameters:
  provider: hosted_vllm
  model_name: hosted_vllm/RedHatAI/Qwen3.8-27B-NVFP4
  base_url: http://127.0.0.1:3235/v1
  api_key: ''
  timeout: 3600
  concurrent_requests: 32
  generation_parameters:
    temperature: 1.0
    max_new_tokens: 65536
    top_p: 0.95
    top_k: 20
    seed: 1234
```
Run once per seed (changing seed in the config each time):

```
lighteval endpoint litellm litellm_config.yaml 'math_500@1@3|0' --output-dir results/ --save-details
lighteval endpoint litellm litellm_config.yaml 'aime25@1@8|0' --output-dir results/ --save-details
lighteval endpoint litellm litellm_config.yaml 'gpqa:diamond@1@3|0' --output-dir results/ --save-details
```
</details>

## Performance Evaluation

Each plot sweeps request load for the `math_reasoning` and `HumanEval` benchmark datasets. The x-axis shows per-user interactivity in tokens per second, where higher values mean a snappier response for an individual request. The y-axis shows total server throughput in tokens per second, where higher values mean the system is serving more aggregate load. Each colored line represents a different model and speculator configuration. 

The plots demonstrate the benefit of not just quantizing the LLM, but combining it with a trained speculator model, [RedHatAI/Qwen3.8-27B-speculator.dspark](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark). Each sweep was done using TP=1,DP=4 on B200s.


![Screenshot 2026-09-24 at 11.25.37 PM](https://cdn-uploads.huggingface.co/production/uploads/650a157669739cd310382f88/BaIb0FXBeHwgeqiNq8rSu.png)
![Screenshot 2026-09-24 at 11.25.25 PM](https://cdn-uploads.huggingface.co/production/uploads/650a157669739cd310382f88/c0peOu-Lb9E4LJEKFbbFR.png)
