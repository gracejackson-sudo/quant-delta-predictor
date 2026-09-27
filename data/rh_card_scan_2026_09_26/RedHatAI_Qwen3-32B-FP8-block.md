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
- Qwen/Qwen3-32B
---


# Qwen3-32B-FP8-block

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

Quantized version of [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve nm-testing/Qwen3-32B-FP8-block --tensor_parallel_size 4
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

model = "nm-testing/Qwen3-32B-FP8-block"

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

MODEL_ID = ""nm-testing/Qwen3-32B-FP8-block""

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
    --model_args pretrained="nm-testing/Qwen3-32B-FP8-block",dtype=auto,add_bos_token=True,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Openllm V2**  
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="nm-testing/Qwen3-32B-FP8-block",dtype=auto,add_bos_token=False,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.7,disable_log_stats=True,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks leaderboard \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Coding Benchmarks**

  ```
  evalplus.evaluate --model "nm-testing/Qwen3-32B-FP8-block" \
                    --dataset "humaneval" \
                    --backend vllm \
                    --tp 4 \
                    --greedy
  evalplus.evaluate --model "nm-testing/Qwen3-32B-FP8-block" \
                  --dataset "mbpp" \
                  --backend vllm \
                  --tp 4 \
                  --greedy
  ```


</details>






### Accuracy
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Qwen/Qwen3-32B</th>
      <th>nm-testing/Qwen3-32B-FP8-block</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>72.95</td>
      <td>72.78</td>
      <td>99.77</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>74.15</td>
      <td>74.53</td>
      <td>100.51</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>84.03</td>
      <td>83.86</td>
      <td>99.80</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>81.99</td>
      <td>81.97</td>
      <td>99.97</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>59.18</td>
      <td>58.72</td>
      <td>99.22</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>76.01</td>
      <td>75.22</td>
      <td>98.96</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>74.72</b></td>
      <td><b>74.51</b></td>
      <td><b>99.72</b></td>
    </tr>
    <!-- OpenLLM Leaderboard V2 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>49.04</td>
      <td>49.28</td>
      <td>100.49</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>35.27</td>
      <td>33.94</td>
      <td>96.21</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>19.94</td>
      <td>17.52</td>
      <td>87.88</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>26.01</td>
      <td>24.41</td>
      <td>93.87</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>40.34</td>
      <td>40.21</td>
      <td>99.67</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>12.38</td>
      <td>12.35</td>
      <td>99.73</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>30.50</b></td>
      <td><b>29.62</b></td>
      <td><b>97.11</b></td>
    </tr>
  
   
   <td rowspan="4" ><strong>Coding</strong>
   </td>
   <td>HumanEval pass@1
   </td>
   <td>90.20
   </td>
   <td>90.20
   </td>
   <td>100.00
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>84.80
   </td>
   <td>84.10
   </td>
   <td>98.35
   </td>
  </tr>
  <tr>
   <td>MBPP pass@1
   </td>
   <td>86.50
   </td>
   <td>86.20
   </td>
   <td>99.65
   </td>
  </tr>
  <tr>
   <td>MBPP+ pass@1
   </td>
   <td>73.00
   </td>
   <td>71.40
   </td>
   <td>97.80
   </td>
  </tr>

    
  </tbody>
</table>