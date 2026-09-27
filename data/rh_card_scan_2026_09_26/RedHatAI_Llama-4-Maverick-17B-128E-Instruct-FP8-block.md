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
- meta-llama/Llama-4-Maverick-17B-128E-Instruct
---



# Llama-4-Maverick-17B-128E-Instruct-block-FP8

## Model Overview
- **Model Architecture:** Llama4ForConditionalGeneration
  - **Input:** Text, Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [meta-llama/Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 



## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Llama-4-Maverick-17B-128E-Instruct-block-FP8 --tensor_parallel_size 8
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

model = "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-block-FP8"

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
from transformers import AutoProcessor, LlamaForCausalLM, AutoModelForImageTextToText

from llmcompressor import oneshot
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

# Load model.
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = replace_modules_for_calibration(model)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per-block quantization
#   * quantize the activations to fp8 with dynamic token activations
ecipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_BLOCK",
    ignore=[
        "re:.*lm_head",
        "re:.*self_attn",
        "re:.*router",
        "re:.*vision_model.*",
        "re:.*multi_modal_projector.*",
        "Llama4TextAttention",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)
dispatch_for_generation(model)


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
    --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-block-FP8",dtype=auto,add_bos_token=True,max_model_len=16384,tensor_parallel_size=8,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Openllm V2**  
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-block-FP8",dtype=auto,add_bos_token=False,max_model_len=16384,tensor_parallel_size=8,gpu_memory_utilization=0.7,disable_log_stats=True,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks leaderboard \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Coding Benchmarks**

  ```
  evalplus.evaluate --model "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-block-FP8" \
                    --dataset "humaneval" \
                    --backend vllm \
                    --tp 8 \
                    --greedy
  evalplus.evaluate --model "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-block-FP8" \
                  --dataset "mbpp" \
                  --backend vllm \
                  --tp 8 \
                  --greedy
  ```


**Multimodal Evaluation**
  ```  
  lm_eval \
  --model vllm-vlm \
  --model_args pretrained="RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8-block",dtype=auto,add_bos_token=False,max_model_len=1000000,tensor_parallel_size=8,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True,max_images=10 \
  --tasks mmlu \
  --apply_chat_template \
  --batch_size auto
  ```


</details>








### Accuracy
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>meta-llama/Llama-4-Maverick-17B-128E-Instruct</th>
      <th>RedHatAI/Llama-4-Maverick-17B-128E-Instruct-block-FP8</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>73.38</td>
      <td>73.38</td>
      <td>100.00</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>93.03</td>
      <td>92.72</td>
      <td>99.67</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>87.39</td>
      <td>87.33</td>
      <td>99.93</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>86.03</td>
      <td>86.15</td>
      <td>100.13</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>62.76</td>
      <td>62.90</td>
      <td>100.23</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>79.56</td>
      <td>79.40</td>
      <td>99.80</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>80.36</b></td>
      <td><b>80.31</b></td>
      <td><b>99.94</b></td>
    </tr>
    <!-- OpenLLM Leaderboard V2 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>89.93</td>
      <td>90.89</td>
      <td>101.07</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>70.53</td>
      <td>71.03</td>
      <td>100.71</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>64.73</td>
      <td>65.26</td>
      <td>100.82</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>31.29</td>
      <td>30.54</td>
      <td>97.59</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>46.56</td>
      <td>46.03</td>
      <td>98.86</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>64.11</td>
      <td>63.95</td>
      <td>99.75</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>61.19</b></td>
      <td><b>61.28</b></td>
      <td><b>100.15</b></td>
    </tr>
    <tr>
      <td rowspan="6" ><strong>Multi-modal</strong>
      </td>
      <td>MMMU (val)
      </td>
      <td>79.08
      </td>
      <td>78.50
      </td>
      <td>99.26
      </td>
    </tr> 
    <!-- <td rowspan="4" ><strong>Coding</strong>
   </td>
   <td>HumanEval pass@1
   </td>
   <td>abc
   </td>
   <td>ijk
   </td>
   <td>xyz
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>abc
   </td>
   <td>ijk
   </td>
   <td>xyz
   </td>
  </tr>
  <tr>
   <td>MBPP pass@1
   </td>
   <td>abc
   </td>
   <td>ijk
   </td>
   <td>xyz
   </td>
  </tr>
  <tr>
   <td>MBPP+ pass@1
   </td>
   <td>abc
   </td>
   <td>ijk
   </td>
   <td>xyz
   </td>
  </tr> -->
  </tbody>
</table>