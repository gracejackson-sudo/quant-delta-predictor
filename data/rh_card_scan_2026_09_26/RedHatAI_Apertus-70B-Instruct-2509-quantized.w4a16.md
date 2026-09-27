---
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
tags:
  - multilingual
  - compliant
  - swiss-ai
  - apertus
  - fp8
  - vllm
  - compressed-tensors
  - llm-compressor
base_model:
- swiss-ai/Apertus-70B-Instruct-2509
---
## Model Overview
- **Model Architecture:** ApertusForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Release Date:** 9/22/2025
- **Version:** 1.0
- **Model Developers:** Red Hat

Quantized version of [swiss-ai/Apertus-70B-2509](https://huggingface.co/swiss-ai/Apertus-70B-2509).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [swiss-ai/Apertus-70B-2509](https://huggingface.co/swiss-ai/Apertus-70B-2509) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Apertus-70B-Instruct-2509-quantized.w4a16
```

2. Send requests to the server:

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/Apertus-70B-Instruct-2509-quantized.w4a16"

messages = [
    {"role": "user", "content": "Give me a short introduction to large language model."},
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```

## Creation

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 

<details>
  <summary>Model Creation Code</summary>

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_stub = "swiss-ai/Apertus-70B-Instruct-2509"
model_name = model_stub.split("/")[-1]

model = AutoModelForCausalLM.from_pretrained(model_stub, dtype="auto")

tokenizer = AutoTokenizer.from_pretrained(model_stub)

# Configure the quantization algorithm and scheme
recipe = QuantizationModifier(
    ignore=["lm_head"],
    targets="Linear",
    scheme="FP8_dynamic",
)

# Apply quantization
oneshot(
    model=model,
    recipe=recipe,
)

# Save to disk in compressed-tensors format
save_path = model_name + "-quantized.w4a16"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")
```
</details>

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard), using the following command:

<details>
<summary>Evaluation Commands</summary>
  
OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Apertus-70B-Instruct-2509-quantized.w4a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,gpu_memory_utilization=0.2,enable_chunked_prefill=True \
  --tasks openllm \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>swiss-ai/Apertus-70B-Instruct-2509</th>
      <th>RedHatAI/Apertus-70B-Instruct-2509-quantized.w4a16</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>70.82</td>
      <td>70.65</td>
      <td>99.8</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>73.69</td>
      <td>73.45</td>
      <td>99.7</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>86.23</td>
      <td>85.67</td>
      <td>99.4</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>69.21</td>
      <td>68.25</td>
      <td>98.6</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>60.31</td>
      <td>60.55</td>
      <td>100.4</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>80.74</td>
      <td>80.03</td>
      <td>99.1</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>73.50</b></td>
      <td><b>73.10</b></td>
      <td><b>99.5</b></td>
    </tr>
  </tbody>
</table>


