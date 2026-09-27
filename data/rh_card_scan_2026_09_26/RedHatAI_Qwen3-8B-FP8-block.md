---
license: apache-2.0
pipeline_tag: text-generation
tags:
- fp8
- quantized
- llm-compressor
- compressed-tensors
- red hat
base_model:
- Qwen/Qwen3-8B
---


# Qwen3-8B-FP8-block

## Model Overview
- **Model Architecture:** Qwen3ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Qwen3-8B-FP8-block --tensor_parallel_size 1
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

model = "RedHatAI/Qwen3-8B-FP8-block"

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

This model was quantized using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as shown below.

<details>
  <summary>Creation details</summary>

```python
from transformers import AutoProcessor, Qwen3ForCausalLM

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "Qwen/Qwen3-8B"

# Load model.
model = Qwen3ForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = replace_modules_for_calibration(model)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per-block quantization
#   * quantize the activations to fp8 with dynamic token activations
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=["lm_head"],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-block"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
```
</details>


## Evaluation


The model was evaluated on the OpenLLM leaderboard task, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>
  
  **Openllm V1**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-8B-FP8-block",dtype=auto,add_bos_token=True,max_model_len=16384,tensor_parallel_size=1,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Openllm V2**  
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-8B-FP8-block",dtype=auto,add_bos_token=False,max_model_len=16384,tensor_parallel_size=1,gpu_memory_utilization=0.7,disable_log_stats=True,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks leaderboard \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Coding Benchmarks**

  ```
  evalplus.evaluate --model "RedHatAI/Qwen3-8B-FP8-block" \
                    --dataset "humaneval" \
                    --backend vllm \
                    --tp 1 \
                    --greedy

  evalplus.evaluate --model "RedHatAI/Qwen3-8B-FP8-block" \
                  --dataset "mbpp" \
                  --backend vllm \
                  --tp 1 \
                  --greedy

  ```

</details>


### Accuracy
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Qwen/Qwen3-8B</th>
      <th>RedHatAI/Qwen3-8B-FP8-block</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>67.66</td>
      <td>67.92</td>
      <td>100.38</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>87.95</td>
      <td>87.79</td>
      <td>99.83</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>76.78</td>
      <td>76.60</td>
      <td>99.77</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>74.88</td>
      <td>74.70</td>
      <td>99.75</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>54.36</td>
      <td>54.27</td>
      <td>99.85</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>71.11</td>
      <td>71.43</td>
      <td>100.44</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>72.12</b></td>
      <td><b>72.12</b></td>
      <td><b>100.00</b></td>
    </tr>
    <!-- OpenLLM Leaderboard V2 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>48.56</td>
      <td>48.80</td>
      <td>100.49</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>29.23</td>
      <td>29.32</td>
      <td>100.30</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>17.82</td>
      <td>18.05</td>
      <td>101.27</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>25.76</td>
      <td>26.09</td>
      <td>101.30</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>41.01</td>
      <td>41.14</td>
      <td>100.32</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>11.32</td>
      <td>11.33</td>
      <td>100.07</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>28.95</b></td>
      <td><b>29.12</b></td>
      <td><b>100.59</b></td>
    </tr>

<td rowspan="4" ><strong>Coding</strong>
   </td>
   <td>HumanEval pass@1
   </td>
   <td>84.80
   </td>
   <td>85.40
   </td>
   <td>100.71
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>78.70
   </td>
   <td>79.90
   </td>
   <td>101.52
   </td>
  </tr>
  <tr>
   <td>MBPP pass@1
   </td>
   <td>72.80
   </td>
   <td>73.50
   </td>
   <td>100.96
   </td>
  </tr>
  <tr>
   <td>MBPP+ pass@1
   </td>
   <td>62.70
   </td>
   <td>64.80
   </td>
   <td>103.35
   </td>
  </tr>

    
  </tbody>
</table>