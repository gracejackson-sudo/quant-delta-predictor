---
language:
- en
- zh
library_name: transformers
license: other
license_name: glm-5.3
pipeline_tag: text-generation
tags:
- nvfp4
- fp4
- vllm
- llm-compressor
- compressed-tensors
base_model: zai-org/GLM-5.3
---

# GLM-5.3-NVFP4

## Model Overview
- **Model Architecture:** GlmMoeDsaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Format:** compressed-tensors
- **Base model:** [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)

This is an NVFP4 checkpoint derived from [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3).
It is optimized for Blackwell inference with NVFP4 weights and activations while retaining higher
precision where needed for model quality.

### Model Optimizations

This model uses NVFP4 FP4 (E2M1) weights and activations and two-level scaling. Layers outside of routed experts are left unquantized. 
The checkpoint uses [LLM Compressor](https://github.com/vllm-project/llm-compressor)'s
compressed-tensors NVFP4 format.

## Deployment

### vLLM Serving

```
vllm serve RedHatAI/GLM-5.3-NVFP4 \
  --served-model-name GLM-5.3-NVFP4 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser glm45 \
  --chat-template-content-format string \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --trust-remote-code
```


For optimal peformance, consider using the DSpark draft model [RedHatAI/GLM-5.3-speculator.dspark](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) for speculative decoding, shown below.

```
vllm serve RedHatAI/GLM-5.3-NVFP4 \
  --served-model-name GLM-5.3-NVFP4 \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser glm45 \
  --chat-template-content-format string \
  --speculative-config '{"model":"RedHatAI/GLM-5.3-speculator.dspark","num_speculative_tokens":8,"method":"dspark"}' \
  --trust-remote-code
```

## Creation

The source model was compressed using an LLM-Compressor with experimental layerwise decompression/compression along with an NVFP4 scheme using `perfectblend` calibration samples. 

## Evaluation

This model was evaluated on GSM8K Platinum, IFEval, MATH-500, AIME 2025, and GPQA Diamond using
[lighteval](https://github.com/huggingface/lighteval). Evaluation was run on a vLLM server hosted on 4xB200. 
Generation used temperature 1.0, the `glm45` reasoning parser, FP8 E4M3 KV cache, and the same
serving stack for the NVFP4 and local FP8 baseline. GSM8K, IFEval, MATH-500, and GPQA used three
seeds; AIME used eight seeds. Recovery is relative to the [base model](https://huggingface.co/zai-org/GLM-5.3) run in the same evaluation harness.

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>RedHatAI/GLM-5.3-NVFP4</th>
      <th>zai-org/GLM-5.3 (FP8)</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Instruction Following</b></td>
      <td>IFEval (prompt-level strict)</td>
      <td>85.70%</td>
      <td>89.77%</td>
      <td>95.5%</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Reasoning</b></td>
      <td>GSM8K Platinum (strict-match)</td>
      <td>96.69%</td>
      <td>97.52%</td>
      <td>99.1%</td>
    </tr>
    <tr>
      <td>MATH-500 (pass@1)</td>
      <td>96.27%</td>
      <td>95.07%</td>
      <td>101.3%</td>
    </tr>
    <tr>
      <td>AIME 2025 (pass@1)</td>
      <td>89.58%</td>
      <td>92.50%</td>
      <td>96.8%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (pass@1)</td>
      <td>90.40%</td>
      <td>92.42%</td>
      <td>97.8%</td>
    </tr>
  </tbody>
</table>

## Performance Evaluation

Performance evaluation for RedHatAI/GLM-5.3-NVFP4 with speculative decoding. 

In the graph, interactivity is the median per-request output-token rate, aggregate throughput is total completed output tokens divided by measurement duration, and each label is completed requests divided by measurement duration. The `math_reasoning` dataset was used for this benchmarking.


![glm53_math_reasoning](https://cdn-uploads.huggingface.co/production/uploads/650a157669739cd310382f88/yDDJe_95-w5NWjCaFxumL.png)
