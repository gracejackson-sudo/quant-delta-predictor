---
license: other
license_name: nvidia-open-model-license
pipeline_tag: text-generation
provider: NVIDIA
tool_calling_supported: true
required_cli_args: ['--reasoning-parser nemotron_v3', '--tool-call-parser qwen3_coder']
chat_template_file_name: None
chat_template_path: None
tool_call_parser: qwen3_coder
validated_tasks:
- tool-calling
tasks:
- text-to-text
- text-generation
- reasoning
- tool-calling
base_model:
- nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16
name: RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block
description: This model was obtained by quantizing the weights and activations of nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16 to FP8 data type.
readme: https://huggingface.co/RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block/blob/main/README.md
tags:
- fp8
- quantized
- llm-compressor
- compressed-tensors
- red hat
validated_on:
- RHOAI 3.5
- RHAIIS 3.5
- vLLM 0.24.0
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">  NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block
 <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" /></h1>
 
## Model Overview
- **Model Architecture:** NemotronHForCausalLM
  - **Input:** Text
  - **Output:** Text
  - **Total Parameters:** 550B
  - **Active Parameters:** 55B
- **Model Optimizations:**
  - **Weight quantization:** FP8 (per-channel)
  - **Activation quantization:** FP8 (dynamic per-token)
- **Intended Use Cases:**
  - Reasoning and complex problem solving.
  - Mathematics and science.
  - Code generation.
  - Instruction following.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 06/04/2025
- **Version:** 1.0
- **Model Developers:** Red Hat

Quantized version of [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformer blocks are quantized.
Weights are quantized with FP8 per-block quantization, while activations are quantized with FP8 dynamic per-token quantization.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend.

**Install dependencies:**
```bash
uv pip install git+https://github.com/vllm-project/vllm.git
uv pip install llmcompressor
```

**Launch the vLLM server:**
```bash
vllm serve RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block \
  --host 0.0.0.0 --port 8088 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --mamba-ssm-cache-dtype float16 \
  --mamba-backend flashinfer \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --speculative-config '{"method": "nemotron_h_mtp", "num_speculative_tokens": 5}' \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' \
  --trust-remote-code
```

**Send requests:**
```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8088/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block"

messages = [
    {"role": "user", "content": "Solve for x: 2x + 5 = 13"},
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

```python
from llmcompressor import model_free_ptq

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16"
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-block"

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*gate$",
        "lm_head",
        "model.embed_tokens",
        "re:.*mixer.conv1d.*",
        "re:.*norm_f*",
        "re:.*bias$",
        "re:.*embed_tokens$",
        "backbone.embeddings"
    ],
    max_workers=15,
    device="cuda:0",
)
```


## Evaluation

The model was evaluated on reasoning tasks using [lighteval](https://github.com/huggingface/lighteval).
[vLLM](https://docs.vllm.ai/en/stable/) was used as the serving backend for all evaluations.

**Install dependencies:**
```bash
uv pip install git+https://github.com/vllm-project/vllm.git
uv pip install lighteval==0.13.0
uv pip install "litellm[caching]>=1.66.0"
```

**Launch the vLLM server:**
```bash
vllm serve RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block \
  --host 0.0.0.0 --port 8088 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --reasoning-parser nemotron_v3 \
  --mamba-ssm-cache-dtype float16 \
  --mamba-backend flashinfer \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-cache-philox-rounds 5 \
  --speculative-config '{"method": "nemotron_h_mtp", "num_speculative_tokens": 5}' \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' \
  --trust-remote-code
```

**AIME 2025:**
```bash
lighteval endpoint litellm \
  "model_name=hosted_vllm/RedHatAI__NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block,provider=hosted_vllm,base_url=http://127.0.0.1:8088/v1,timeout=3600,concurrent_requests=32,generation_parameters={temperature:1.0,top_p:0.95,max_new_tokens:32768}" \
  "aime25|0" \
  --output-dir results --save-details
```

**GPQA Diamond:**
```bash
lighteval endpoint litellm \
  "model_name=hosted_vllm/RedHatAI__NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block,provider=hosted_vllm,base_url=http://127.0.0.1:8088/v1,timeout=3600,concurrent_requests=32,generation_parameters={temperature:1.0,top_p:0.95,max_new_tokens:32768}" \
  "gpqa:diamond|0" \
  --output-dir results --save-details
```

### Accuracy

<table>
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16</th>
      <th>nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4</th>
      <th>RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic</th>
      <th>RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block<br>(this model)</th>
      <th>RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-quantized.w4a16</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AIME 2025 (pass@1)</td>
      <td>90.00</td>
      <td>90.00 (100.0%)</td>
      <td>93.33 (103.7%)</td>
      <td>86.67 (96.3%)</td>
      <td>86.67 (96.3%)</td>
    </tr>
    <tr>
      <td>GPQA Diamond (pass@1)</td>
      <td>78.79</td>
      <td>84.85 (107.7%)</td>
      <td>82.32 (104.5%)</td>
      <td>81.31 (103.2%)</td>
      <td>81.82 (103.8%)</td>
    </tr>
    <tr>
      <td><strong>Average</strong></td>
      <td><strong>84.39</strong></td>
      <td><strong>87.42 (103.6%)</strong></td>
      <td><strong>87.83 (104.1%)</strong></td>
      <td><strong>83.99 (99.5%)</strong></td>
      <td><strong>84.24 (99.8%)</strong></td>
    </tr>
  </tbody>
</table>

### Extended Evaluation

The model was evaluated across instruct, reasoning, and coding tasks. Scores are averaged over multiple seeds (3 seeds for most benchmarks, 8 for AIME 2025). Recovery is computed as the ratio of the quantized model score to the base model score.

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16</th>
      <th>RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block<br>(this model)</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Instruct</td>
      <td>MMLU-CoT (5-shot)</td>
      <td>91.09</td>
      <td>90.98</td>
      <td>99.88%</td>
    </tr>
    <tr>
      <td>Instruct</td>
      <td>GSM8K Platinum (5-shot)</td>
      <td>98.59</td>
      <td>98.68</td>
      <td>100.09%</td>
    </tr>
    <tr>
      <td>Instruct</td>
      <td>IFEval (0-shot)</td>
      <td>91.00</td>
      <td>90.70</td>
      <td>99.67%</td>
    </tr>
    <tr>
      <td>Instruct</td>
      <td>MATH-500</td>
      <td>83.60</td>
      <td>82.80</td>
      <td>99.04%</td>
    </tr>
    <tr>
      <td>Reasoning</td>
      <td>AIME 2025</td>
      <td>64.17</td>
      <td>65.42</td>
      <td>101.95%</td>
    </tr>
    <tr>
      <td>Reasoning</td>
      <td>MATH-500</td>
      <td>85.80</td>
      <td>86.00</td>
      <td>100.23%</td>
    </tr>
    <tr>
      <td>Reasoning</td>
      <td>GSM8K Platinum (0-shot)</td>
      <td>96.77</td>
      <td>97.08</td>
      <td>100.32%</td>
    </tr>
    <tr>
      <td>Reasoning</td>
      <td>IFEval (0-shot)</td>
      <td>94.58</td>
      <td>94.82</td>
      <td>100.25%</td>
    </tr>
    <tr>
      <td>Coding</td>
      <td>LCB CodeGen v6</td>
      <td>51.81</td>
      <td>52.57</td>
      <td>101.47%</td>
    </tr>
  </tbody>
</table>

### Tool Calling Evaluation

The model was evaluated on tool calling tasks using the [Berkeley Function-Calling Leaderboard v4 (BFCLv4)](https://gorilla.cs.berkeley.edu/leaderboard.html).
[vLLM](https://docs.vllm.ai/en/stable/) was used as the serving backend for all evaluations.

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16</th>
      <th>RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-block<br>(this model)</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Overall</strong></td>
      <td><strong>BFCLv4 Overall Acc</strong></td>
      <td><strong>55.44</strong></td>
      <td><strong>54.84</strong></td>
      <td><strong>98.92%</strong></td>
    </tr>
    <tr>
      <td>Single Turn</td>
      <td>Non-Live Acc</td>
      <td>45.00</td>
      <td>44.31</td>
      <td>98.47%</td>
    </tr>
    <tr>
      <td>Single Turn</td>
      <td>Live Acc</td>
      <td>71.65</td>
      <td>72.32</td>
      <td>100.94%</td>
    </tr>
    <tr>
      <td>Multi-Turn</td>
      <td>Multi-Turn Acc</td>
      <td>42.12</td>
      <td>42.75</td>
      <td>101.50%</td>
    </tr>
    <tr>
      <td>Agentic</td>
      <td>Agentic Acc</td>
      <td>57.64</td>
      <td>55.53</td>
      <td>96.34%</td>
    </tr>
  </tbody>
</table>
