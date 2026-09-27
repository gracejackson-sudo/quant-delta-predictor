---
tags:
- vllm
- vision
- fp8
license: apache-2.0
license_link: >-
  https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
- en
base_model: Qwen/Qwen2.5-VL-72B-Instruct
library_name: transformers
---

# Qwen2.5-VL-72B-Instruct-quantized-FP8-Dynamic

## Model Overview
- **Model Architecture:** Qwen2.5-VL-72B-Instruct
  - **Input:** Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2/24/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen/Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct) to FP8 data type, ready for inference with vLLM >= 0.5.2.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="neuralmagic/Qwen2.5-VL-72B-Instruct-FP8-Dynamic",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
)

# prepare inputs
question = "What is the content of this image?"
inputs = {
    "prompt": f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n",
    "multi_modal_data": {
        "image": ImageAsset("cherry_blossom").pil_image.convert("RGB")
    },
}

# generate response
print("========== SAMPLE GENERATION ==============")
outputs = llm.generate(inputs, SamplingParams(temperature=0.2, max_tokens=64))
print(f"PROMPT  : {outputs[0].prompt}")
print(f"RESPONSE: {outputs[0].outputs[0].text}")
print("==========================================")
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below as part a multimodal announcement blog.

<details>
  <summary>Model Creation Code</summary>
  
```python
import requests
import torch
from PIL import Image
from transformers import AutoProcessor
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import (
    TraceableQwen2_5_VLForConditionalGeneration,
)
from llmcompressor.modifiers.quantization import QuantizationModifier

# Load model.
model_id = Qwen/Qwen2.5-VL-72B-Instruct
model = TraceableQwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Recipe
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        sequential_targets=["MistralDecoderLayer"],
        ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
    ),
]

SAVE_DIR=f"{model_id.split('/')[1]}-FP8-Dynamic"

# Perform oneshot
oneshot(
    model=model,
    recipe=recipe,
    trust_remote_code_model=True,
    output_dir=SAVE_DIR
)


```
</details>

## Evaluation

The model was evaluated using [mistral-evals](https://github.com/neuralmagic/mistral-evals) for vision-related tasks and using [lm_evaluation_harness](https://github.com/neuralmagic/lm-evaluation-harness) for select text-based benchmarks. The evaluations were conducted using the following commands:

<details>
<summary>Evaluation Commands</summary>
  
### Vision Tasks
- vqav2
- docvqa
- mathvista
- mmmu
- chartqa

```
vllm serve neuralmagic/pixtral-12b-quantized.w8a8 --tensor_parallel_size 1 --max_model_len 25000 --trust_remote_code --max_num_seqs 8 --gpu_memory_utilization 0.9 --dtype float16 --limit_mm_per_prompt image=7

python -m eval.run eval_vllm \
        --model_name neuralmagic/pixtral-12b-quantized.w8a8 \
        --url http://0.0.0.0:8000 \
        --output_dir ~/tmp \
        --eval_name <vision_task_name>
```

### Text-based Tasks
#### MMLU
  
```
lm_eval \
  --model vllm \
  --model_args pretrained="<model_name>",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=<n>,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto \
  --output_path output_dir

```

#### MGSM

```
lm_eval \
  --model vllm \
  --model_args pretrained="<model_name>",dtype=auto,max_model_len=4096,max_gen_toks=2048,max_num_seqs=128,tensor_parallel_size=<n>,gpu_memory_utilization=0.9 \
  --tasks mgsm_cot_native \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto \
  --output_path output_dir

```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Qwen/Qwen2.5-VL-72B-Instruct</th>
      <th>neuralmagic/Qwen2.5-VL-72B-Instruct-FP8-Dynamic</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><b>Vision</b></td>
      <td>MMMU (val, CoT)<br><i>explicit_prompt_relaxed_correctness</i></td>
      <td>64.33</td>
      <td>66.88</td>
      <td>103.96%</td>
    </tr>
    <tr>
      <td>VQAv2 (val)<br><i>vqa_match</i></td>
      <td>81.94</td>
      <td>81.94</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td>DocVQA (val)<br><i>anls</i></td>
      <td>94.71</td>
      <td>94.64</td>
      <td>99.93%</td>
    </tr>
    <tr>
      <td>ChartQA (test, CoT)<br><i>anywhere_in_answer_relaxed_correctness</i></td>
      <td>88.96</td>
      <td>89.04</td>
      <td>100.09%</td>
    </tr>
    <tr>
      <td>Mathvista (testmini, CoT)<br><i>explicit_prompt_relaxed_correctness</i></td>
      <td>78.18</td>
      <td>77.78</td>
      <td>99.49%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>81.62</b></td>
      <td><b>81.86</b></td>
      <td><b>100.29%</b></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Text</b></td>
      <td>MGSM (CoT)</td>
      <td>75.45</td>
      <td>75.29</td>
      <td>99.79%</td>
    </tr>
    <tr>
      <td>MMLU (5-shot)</td>
      <td>86.16</td>
      <td>86.12</td>
      <td>99.95%</td>
    </tr>
  </tbody>
</table>


## Inference Performance


This model achieves up to 1.79x speedup in single-stream deployment and up to 1.84x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>
```
  guidellm --model neuralmagic/Qwen2.5-VL-72B-Instruct-FP8-Dynamic --target "http://localhost:8000/v1" --data-type emulated --data prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>,images=<num_images>,width=<image_width>,height=<image_height> --max seconds 120 --backend aiohttp_server
```

</details>


### Single-stream performance (measured with vLLM version 0.7.2)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th style="text-align: center;" colspan="2" >Document Visual Question Answering<br>1680W x 2240H<br>64/128</th>
      <th style="text-align: center;" colspan="2" >Visual Reasoning <br>640W x 480H<br>128/128</th>
      <th style="text-align: center;" colspan="2" >Image Captioning<br>480W x 360H<br>0/128</th>
    </tr>
    <tr>
      <th>Hardware</th>
      <th>Number of GPUs</th>
      <th>Model</th>
      <th>Average Cost Reduction</th>
      <th>Latency (s)</th>
      <th>Queries Per Dollar</th>
      <th>Latency (s)th>
      <th>Queries Per Dollar</th>
      <th>Latency (s)</th>
      <th>Queries Per Dollar</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">A100</td>
      <td>4</td>
      <td>Qwen/Qwen2.5-VL-72B-Instruct</td>
      <td></td>
      <td>6.4</td>
      <td>78</td>
      <td>4.5</td>
      <td>111</td>
      <td>4.4</td>
      <td>113</td>
    </tr>
    <tr>
      <td>2</td>
      <td>neuralmagic/Qwen2.5-VL-72B-Instruct-quantized.w8a8</td>
      <td>1.85</td>
      <td>7.0</td>
      <td>143</td>
      <td>4.9</td>
      <td>205</td>
      <td>4.8</td>
      <td>211</td>
    </tr>
    <tr>
      <td>1</td>
      <td>neuralmagic/Qwen2.5-VL-72B-Instruct-quantized.w4a16</td>
      <td>3.33</td>
      <td>9.4</td>
      <td>213</td>
      <td>5.1</td>
      <td>396</td>
      <td>4.8</td>
      <td>420</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100</td>
      <td>4</td>
      <td>Qwen/Qwen2.5-VL-72B-Instruct</td>
      <td></td>
      <td>4.3</td>
      <td>68</td>
      <td>3.0</td>
      <td>97</td>
      <td>2.9</td>
      <td>100</td>
    </tr>
    <tr>
      <td>2</td>
      <td>neuralmagic/Qwen2.5-VL-72B-Instruct-FP8-Dynamic</td>
      <td>1.79</td>
      <td>4.6</td>
      <td>122</td>
      <td>3.3</td>
      <td>173</td>
      <td>3.2</td>
      <td>177</td>
    </tr>
    <tr>
      <td>1</td>
      <td>neuralmagic/Qwen2.5-VL-72B-Instruct-quantized.w4a16</td>
      <td>5.66</td>
      <td>4.3</td>
      <td>252</td>
      <td>4.4</td>
      <td>251</td>
      <td>4.2</td>
      <td>259</td>
    </tr>
  </tbody>
</table>

**Use case profiles: Image Size (WxH) / prompt tokens / generation tokens

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).

### Multi-stream asynchronous performance (measured with vLLM version 0.7.2)

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th style="text-align: center;" colspan="2" >Document Visual Question Answering<br>1680W x 2240H<br>64/128</th>
      <th style="text-align: center;" colspan="2" >Visual Reasoning <br>640W x 480H<br>128/128</th>
      <th style="text-align: center;" colspan="2" >Image Captioning<br>480W x 360H<br>0/128</th>
    </tr>
    <tr>
      <th>Hardware</th>
      <th>Model</th>
      <th>Average Cost Reduction</th>
      <th>Maximum throughput (QPS)</th>
      <th>Queries Per Dollar</th>
      <th>Maximum throughput (QPS)</th>
      <th>Queries Per Dollar</th>
      <th>Maximum throughput (QPS)</th>
      <th>Queries Per Dollar</th>
    </tr>
  </thead>
  <tbody style="text-align: center">
    <tr>
      <th rowspan="3" valign="top">A100x4</th>
      <td>Qwen/Qwen2.5-VL-72B-Instruct</td>
      <td></td>
      <td>0.4</td>
      <td>180</td>
      <td>1.1</td>
      <td>539</td>
      <td>1.2</td>
      <td>595</td>
    </tr>
    <tr>
      <td>neuralmagic/Qwen2.5-VL-72B-Instruct-quantized.w8a8</td>
      <td>1.80</td>
      <td>0.6</td>
      <td>289</td>
      <td>2.0</td>
      <td>1020</td>
      <td>2.3</td>
      <td>1133</td>
    </tr>
    <tr>
      <td>neuralmagic/Qwen2.5-VL-72B-Instruct-quantized.w4a16</td>
      <td>2.75</td>
      <td>0.7</td>
      <td>341</td>
      <td>3.2</td>
      <td>1588</td>
      <td>4.1</td>
      <td>2037</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x4</th>
      <td>Qwen/Qwen2.5-VL-72B-Instruct</td>
      <td></td>
      <td>0.5</td>
      <td>134</td>
      <td>1.2</td>
      <td>357</td>
      <td>1.3</td>
      <td>379</td>
    </tr>
    <tr>
      <td>neuralmagic/Qwen2.5-VL-72B-Instruct-FP8-Dynamic</td>
      <td>1.73</td>
      <td>0.9</td>
      <td>247</td>
      <td>2.2</td>
      <td>621</td>
      <td>2.4</td>
      <td>669</td>
    </tr>
    <tr>
      <td>neuralmagic/Qwen2.5-VL-72B-Instruct-quantized.w4a16</td>
      <td>8.27</td>
      <td>3.3</td>
      <td>913</td>
      <td>3.3</td>
      <td>898</td>
      <td>3.6</td>
      <td>991</td>
    </tr>
  </tbody>
</table>

**Use case profiles: Image Size (WxH) / prompt tokens / generation tokens

**QPS: Queries per second.

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).