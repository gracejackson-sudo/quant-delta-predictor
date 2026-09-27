---
tags:
- vllm
- vision
- w4a16
license: apache-2.0
license_link: >-
  https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
- en
base_model: Qwen/Qwen2.5-VL-7B-Instruct
library_name: transformers
---

# Qwen2.5-VL-7B-Instruct-quantized-w4a16

## Model Overview
- **Model Architecture:** Qwen/Qwen2.5-VL-7B-Instruct
  - **Input:** Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** FP16
- **Release Date:** 2/24/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) to INT8 data type, ready for inference with vLLM >= 0.5.2.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w4a16",
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
import base64
from io import BytesIO
import torch
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import (
    TraceableQwen2_5_VLForConditionalGeneration,
)
from compressed_tensors.quantization import QuantizationArgs, QuantizationType, QuantizationStrategy, ActivationOrdering, QuantizationScheme

# Load model.
model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

model = TraceableQwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "lmms-lab/flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42)
dampening_frac=0.01

# Apply chat template and tokenize inputs.
def preprocess_and_tokenize(example):
    # preprocess
    buffered = BytesIO()
    example["image"].save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue())
    encoded_image_text = encoded_image.decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base64_qwen},
                {"type": "text", "text": "What does the image show?"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    # tokenize
    return processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
    )
ds = ds.map(preprocess_and_tokenize, remove_columns=ds["calibration"].column_names)

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "config_group": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=128,
                symmetric=True,
                dynamic=False,
                actorder=ActivationOrdering.WEIGHT,
            ),
        ),
    },
    sequential_targets=["Qwen2_5_VLDecoderLayer"],
    ignore=["lm_head", "re:visual.*"],
    update_size=NUM_CALIBRATION_SAMPLES,
    dampening_frac=dampening_frac
)

SAVE_DIR=f"{model_id.split('/')[1]}-quantized.w4a16"

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
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
      <th>Qwen/Qwen2.5-VL-7B-Instruct</th>
      <th>Qwen2.5-VL-7B-Instruct-quantized.W4A16</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><b>Vision</b></td>
      <td>MMMU (val, CoT)<br><i>explicit_prompt_relaxed_correctness</i></td>
      <td>52.00</td>
      <td>51.11</td>
      <td>98.29%</td>
    </tr>
    <tr>
      <td>VQAv2 (val)<br><i>vqa_match</i></td>
      <td>75.59</td>
      <td>73.90</td>
      <td>97.76%</td>
    </tr>
    <tr>
      <td>DocVQA (val)<br><i>anls</i></td>
      <td>94.27</td>
      <td>94.13</td>
      <td>99.85%</td>
    </tr>
    <tr>
      <td>ChartQA (test, CoT)<br><i>anywhere_in_answer_relaxed_correctness</i></td>
      <td>86.44</td>
      <td>85.64</td>
      <td>99.07%</td>
    </tr>
    <tr>
      <td>Mathvista (testmini, CoT)<br><i>explicit_prompt_relaxed_correctness</i></td>
      <td>69.47</td>
      <td>67.17</td>
      <td>96.69%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>75.95</b></td>
      <td><b>74.79</b></td>
      <td><b>98.47%</b></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Text</b></td>
      <td>MGSM (CoT)</td>
      <td>56.38</td>
      <td>51.89</td>
      <td>92.04%</td>
    </tr>
    <tr>
      <td>MMLU (5-shot)</td>
      <td>71.09</td>
      <td>68.67</td>
      <td>96.59%</td>
    </tr>
  </tbody>
</table>

## Inference Performance


This model achieves up to 2.35x speedup in single-stream deployment and up to 2.02x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>
```
  guidellm --model neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w4a16 --target "http://localhost:8000/v1" --data-type emulated --data prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>,images=<num_images>,width=<image_width>,height=<image_height> --max seconds 120 --backend aiohttp_server
```

</details>

### Single-stream performance (measured with vLLM version 0.7.2)

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
      <th>Latency (s)</th>
      <th>Queries Per Dollar</th>
      <th>Latency (s)th>
      <th>Queries Per Dollar</th>
      <th>Latency (s)</th>
      <th>Queries Per Dollar</th>
    </tr>
  </thead>
  <tbody style="text-align: center">
       <tr>
      <th rowspan="3" valign="top">A6000x1</th>
      <th>Qwen/Qwen2.5-VL-7B-Instruct</th>
      <td></td>
      <td>4.9</td>
      <td>912</td>
      <td>3.2</td>
      <td>1386</td>
      <td>3.1</td>
      <td>1431</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w8a8</th>
      <td>1.50</td>
      <td>3.6</td>
      <td>1248</td>
      <td>2.1</td>
      <td>2163</td>
      <td>2.0</td>
      <td>2237</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w4a16</th>
      <td>2.05</td>
      <td>3.3</td>
      <td>1351</td>
      <td>1.4</td>
      <td>3252</td>
      <td>1.4</td>
      <td>3321</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>Qwen/Qwen2.5-VL-7B-Instruct</th>
      <td></td>
      <td>2.8</td>
      <td>707</td>
      <td>1.7</td>
      <td>1162</td>
      <td>1.7</td>
      <td>1198</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w8a8</th>
      <td>1.24</td>
      <td>2.4</td>
      <td>851</td>
      <td>1.4</td>
      <td>1454</td>
      <td>1.3</td>
      <td>1512</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w4a16</th>
      <td>1.49</td>
      <td>2.2</td>
      <td>912</td>
      <td>1.1</td>
      <td>1791</td>
      <td>1.0</td>
      <td>1950</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>Qwen/Qwen2.5-VL-7B-Instruct</th>
      <td></td>
      <td>2.0</td>
      <td>557</td>
      <td>1.2</td>
      <td>919</td>
      <td>1.2</td>
      <td>941</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-FP8-Dynamic</th>
      <td>1.28</td>
      <td>1.6</td>
      <td>698</td>
      <td>0.9</td>
      <td>1181</td>
      <td>0.9</td>
      <td>1219</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w4a16</th>
      <td>1.28</td>
      <td>1.6</td>
      <td>686</td>
      <td>0.9</td>
      <td>1191</td>
      <td>0.9</td>
      <td>1228</td>
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
      <th rowspan="3" valign="top">A6000x1</th>
      <th>Qwen/Qwen2.5-VL-7B-Instruct</th>
      <td></td>
      <td>0.4</td>
      <td>1837</td>
      <td>1.5</td>
      <td>6846</td>
      <td>1.7</td>
      <td>7638</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w8a8</th>
      <td>1.41</td>
      <td>0.5</td>
      <td>2297</td>
      <td>2.3</td>
      <td>10137</td>
      <td>2.5</td>
      <td>11472</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w4a16</th>
      <td>1.60</td>
      <td>0.4</td>
      <td>1828</td>
      <td>2.7</td>
      <td>12254</td>
      <td>3.4</td>
      <td>15477</td>
    </tr>
     <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>Qwen/Qwen2.5-VL-7B-Instruct</th>
      <td></td>
      <td>0.7</td>
      <td>1347</td>
      <td>2.6</td>
      <td>5221</td>
      <td>3.0</td>
      <td>6122</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w8a8</th>
      <td>1.27</td>
      <td>0.8</td>
      <td>1639</td>
      <td>3.4</td>
      <td>6851</td>
      <td>3.9</td>
      <td>7918</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w4a16</th>
      <td>1.21</td>
      <td>0.7</td>
      <td>1314</td>
      <td>3.0</td>
      <td>5983</td>
      <td>4.6</td>
      <td>9206</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>Qwen/Qwen2.5-VL-7B-Instruct</th>
      <td></td>
      <td>0.9</td>
      <td>969</td>
      <td>3.1</td>
      <td>3358</td>
      <td>3.3</td>
      <td>3615</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-FP8-Dynamic</th>
      <td>1.29</td>
      <td>1.2</td>
      <td>1331</td>
      <td>3.8</td>
      <td>4109</td>
      <td>4.2</td>
      <td>4598</td>
    </tr>
    <tr>
      <th>neuralmagic/Qwen2.5-VL-7B-Instruct-quantized.w4a16</th>
      <td>1.28</td>
      <td>1.2</td>
      <td>1298</td>
      <td>3.8</td>
      <td>4190</td>
      <td>4.2</td>
      <td>4573</td>
    </tr>
  </tbody>
</table>

**Use case profiles: Image Size (WxH) / prompt tokens / generation tokens

**QPS: Queries per second.

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).