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
- meta-llama/Llama-3.3-70B-Instruct
---


# Llama-3.3-70B-Instruct-FP8-block

## Model Overview
- **Model Architecture:** LlamaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve nm-testing/Llama-3.3-70B-Instruct-FP8-block --tensor_parallel_size 4
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

model = "nm-testing/Llama-3.3-70B-Instruct-FP8-block"


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
from transformers import AutoProcessor, LlamaForCausalLM

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

# Load model.
model = LlamaForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
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


The model was evaluated on the OpenLLMv1 leaderboard task, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), on reasoning tasks using [lighteval](https://github.com/huggingface/lighteval).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>
  
  **Openllm V1**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="nm-testing/Llama-3.3-70B-Instruct-FP8-block",dtype=auto,add_bos_token=True,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Openllm V2**  
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="nm-testing/Llama-3.3-70B-Instruct-FP8-block",dtype=auto,add_bos_token=False,max_model_len=16384,tensor_parallel_size=4,gpu_memory_utilization=0.7,disable_log_stats=True,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks leaderboard \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Coding Benchmarks**

  ```
  evalplus.evaluate --model "nm-testing/Llama-3.3-70B-Instruct-FP8-block" \
                    --dataset "humaneval" \
                    --backend vllm \
                    --tp 4 \
                    --greedy
  evalplus.evaluate --model "nm-testing/Llama-3.3-70B-Instruct-FP8-block" \
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
      <th>meta-llama/Llama-3.3-70B-Instruct</th>
      <th>nm-testing/Llama-3.3-70B-Instruct-FP8-block</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>72.53</td>
      <td>72.61</td>
      <td>100.12</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>76.35</td>
      <td>73.16</td>
      <td>95.83</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>86.65</td>
      <td>86.56</td>
      <td>99.90</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>82.51</td>
      <td>82.38</td>
      <td>99.84</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>62.83</td>
      <td>62.64</td>
      <td>99.69</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>83.50</td>
      <td>83.27</td>
      <td>99.72</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>77.39</b></td>
      <td><b>76.77</b></td>
      <td><b>99.20</b></td>
    </tr>
    <!-- OpenLLM Leaderboard V2 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>92.57</td>
      <td>92.57</td>
      <td>100.00</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>69.03</td>
      <td>68.98</td>
      <td>99.92</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>49.24</td>
      <td>49.47</td>
      <td>100.46</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>32.63</td>
      <td>32.63</td>
      <td>100.00</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>44.31</td>
      <td>43.92</td>
      <td>99.10</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>53.55</td>
      <td>53.56</td>
      <td>100.02</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>56.89</b></td>
      <td><b>56.85</b></td>
      <td><b>99.93</b></td>
    </tr>
   <td rowspan="4" ><strong>Coding</strong>
   </td>
   <td>HumanEval pass@1
   </td>
   <td>82.90
   </td>
   <td>82.90
   </td>
   <td>100.00
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>78.70
   </td>
   <td>77.40
   </td>
   <td>98.35
   </td>
  </tr>
  <tr>
   <td>MBPP pass@1
   </td>
   <td>87.70
   </td>
   <td>87.60
   </td>
   <td>99.89
   </td>
  </tr>
  <tr>
   <td>MBPP+ pass@1
   </td>
   <td>72.80
   </td>
   <td>73.00
   </td>
   <td>100.27
   </td>
  </tr>


    
  </tbody>
</table>