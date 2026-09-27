---
license: apache-2.0
pipeline_tag: text-generation
tags:
- fp8
- quantized
- llm-compressor
- compressed-tensors
- red hat
- image
- video
- multimodal
base_model:
- Qwen/Qwen3-VL-235B-A22B-Instruct
---


# Qwen3-VL-235B-A22B-Instruct-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen3VLMoeForConditionalGeneration
  - **Input:** Text/Image/Video
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 09/28/202510
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-VL-235B-A22B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic --tensor_parallel_size 4
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

model = "RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
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

<details>
  <summary>Creation details</summary>

```python
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-VL-235B-A22B-Instruct"

# Load model.
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = replace_modules_for_calibration(model)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per-channel quantization
#   * quantize the activations to fp8 with dynamic token activations
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
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


The model was evaluated on the OpenLLMv1 leaderboard task, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), on reasoning tasks using [lighteval](https://github.com/huggingface/lighteval) and on vision tasks using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>
  
  **lm-evaluation-harness**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=4,gpu_memory_utilization=0.8,enable_chunked_prefill=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --output_path output_dir \
    --show_config
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic
    dtype: auto
    gpu_memory_utilization: 0.9
    generation_parameters:
      temperature: 0.6
      min_p: 0.0
      top_p: 0.95
      top_k: 20
      max_new_tokens: 32768
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|aime25|0 \
  ```

 **lmms-eval**
```
python3 -m lmms_eval \
    --model vllm \
    --model_args model=RedHatAI/Qwen3-VL-235B-A22B-Instruct-FP8-dynamic,tensor_parallel_size=4,max_model_len=8192,gpu_memory_utilization=0.9 \
    --tasks mmmu_val, chartqa\
    --batch_size 1
```

</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Qwen3-VL-235B-A22B-Instruct</th>
      <th>Qwen3-VL-235B-A22B-Instruct-FP8-dynamic</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>76.54</td>
      <td>75.94</td>
      <td>99.2</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>90.30</td>
      <td>89.92</td>
      <td>99.6</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>87.81</td>
      <td>87.74</td>
      <td>99.9</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>87.11</td>
      <td>87.23</td>
      <td>100.1</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>63.19</td>
      <td>63.48</td>
      <td>100.5</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>81.61</td>
      <td>82.64</td>
      <td>101.3</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>81.09</b></td>
      <td><b>81.16</b></td>
      <td><b>100.1</b></td>
    </tr>
  </tbody>
  <tbody>
    <!-- Reasoning -->
    <tr>
      <td rowspan="6" ><strong>Reasoning<br>(generation)</strong>
      </td>
      <td>AIME 2025
      </td>
      <td>70.00
      </td>
      <td>80.00
      </td>
      <td>114.3
      </td>
    </tr>
  </tbody>
  <tbody>
    <!-- Multimodal -->
    <tr>
      <td rowspan="6" ><strong>Multi-modal</strong>
      </td>
      <td>ChartQA (relaxed_overall)
      </td>
      <td>90.12
      </td>
      <td>90.04
      </td>
      <td>99.9
      </td>
    </tr>
    <tr>
      <td>MMMU (val)</td>
      <td>63.67</td>
      <td>63.33</td>
      <td>99.5</td>
    </tr>
  </tbody>
</table>