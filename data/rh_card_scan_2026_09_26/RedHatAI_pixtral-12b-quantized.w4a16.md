---
tags:
- w4a16
- int4
- vllm
- vision
license: apache-2.0
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
  - en
base_model: mgoin/pixtral-12b
library_name: transformers
---

# pixtral-12b-quantized.w4a16

## Model Overview
- **Model Architecture:** mgoin/pixtral-12b
  - **Input:** Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** FP16
- **Release Date:** 2/24/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [mgoin/pixtral-12b](https://huggingface.co/mgoin/pixtral-12b).

### Model Optimizations

This model was obtained by quantizing the weights of [mgoin/pixtral-12b](https://huggingface.co/mgoin/pixtral-12b) to INT4 data type, ready for inference with vLLM >= 0.5.2.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="neuralmagic/pixtral-12b-quantized.w4a16",
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
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import TraceableLlavaForConditionalGeneration
from compressed_tensors.quantization import QuantizationArgs, QuantizationType, QuantizationStrategy, ActivationOrdering, QuantizationScheme

# Load model.
model_id = "mgoin/pixtral-12b"
model = TraceableLlavaForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype="auto"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048
dampening_frac=0.01

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {
        "input_ids": torch.LongTensor(batch[0]["input_ids"]),
        "attention_mask": torch.tensor(batch[0]["attention_mask"]),
        "pixel_values": torch.tensor(batch[0]["pixel_values"]),
    }

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
    sequential_targets=["MistralDecoderLayer"],
    ignore=["re:.*lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
    update_size=NUM_CALIBRATION_SAMPLES,
    dampening_frac=dampening_frac,
)

SAVE_DIR = f"{model_id.split('/')[1]}-quantized.w4a16"

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=DATASET_ID,
    splits=DATASET_SPLIT,
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
        --model_name neuralmagic/pixtral-12b-quantized.w4a16 \
        --url http://0.0.0.0:8000 \
        --output_dir ~/tmp
        --eval_name <vision_task_name>
```

### Text-based Tasks
#### MMLU
  
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/pixtral-12b-quantized.w4a16 ",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=<n>,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks mmlu \
  --num_fewshot 5
  --batch_size auto \
  --output_path output_dir \

```

#### HumanEval

##### Generation
```
python3 codegen/generate.py \
  --model neuralmagic/pixtral-12b-quantized.w4a16 \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval
```
##### Sanitization
```
python3 evalplus/sanitize.py \
  humaneval/neuralmagic/pixtral-12b-quantized.w4a16_vllm_temp_0.2
```
##### Evaluation
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/neuralmagic/pixtral-12b-quantized.w4a16_vllm_temp_0.2-sanitized
```
</details>


### Accuracy
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>mgoin/pixtral-12b</th>
      <th>neuralmagic/pixtral-12b-quantized.w4a16</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><b>Vision</b></td>
      <td>MMMU (val, CoT)<br><i>explicit_prompt_relaxed_correctness</i><br></td>
      <td>48.00</td>
      <td>44.67</td>
      <td>93.06%</td>
    </tr>
    <tr>
      <td>VQAv2 (val)<br><i>vqa_match</i></td>
      <td>78.71</td>
      <td>77.04</td>
      <td>97.88%</td>
    </tr>
    <tr>
      <td>DocVQA (val)<br><i>anls</i></td>
      <td>89.47</td>
      <td>89.02</td>
      <td>99.50%</td>
    </tr>
    <tr>
      <td>ChartQA (test, CoT)<br><i>anywhere_in_answer_relaxed_correctness</i></td>
      <td>81.68</td>
      <td>82.12</td>
      <td>100.54%</td>
    </tr>
    <tr>
      <td>Mathvista (testmini, CoT)<br><i>explicit_prompt_relaxed_correctness</i></td>
      <td>56.50</td>
      <td>54.40</td>
      <td>96.28%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>70.07</b></td>
      <td><b>69.05</b></td>
      <td><b>98.54%</b></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Text</b></td>
      <td>HumanEval <br><i>pass@1</i></td>
      <td>71.40</td>
      <td>63.80</td>
      <td>89.37%</td>
    </tr>
    <tr>
      <td>MMLU (5-shot)</td>
      <td>68.40</td>
      <td>65.56</td>
      <td>95.86%</td>
    </tr>
  </tbody>
</table>


## Inference Performance


This model achieves up to 2.47x speedup in single-stream deployment and up to 1.74x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>
```
  guidellm --model neuralmagic/pixtral-12b-quantized.w4a16 --target "http://localhost:8000/v1" --data-type emulated --data prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>,images=<num_images>,width=<image_width>,height=<image_height> --max seconds 120 --backend aiohttp_server
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
      <th>Latency (s)<th>
      <th>Queries Per Dollar</th>
      <th>Latency (s)</th>
      <th>Queries Per Dollar</th>
    </tr>
  </thead>
  <tbody style="text-align: center">
    <tr>
      <th rowspan="3" valign="top">A6000x1</th>
      <th>mgoin/pixtral-12b</th>
      <td></td>
      <td>5.7</td>
      <td>796</td>
      <td>4.8</td>
      <td>929</td>
      <td>4.7</td>
      <td>964</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w8a8</th>
      <td>1.55</td>
      <td>3.7</td>
      <td>1220</td>
      <td>3.1</td>
      <td>1437</td>
      <td>3.0</td>
      <td>1511</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w4a16</th>
      <td>2.16</td>
      <td>3.2</td>
      <td>1417</td>
      <td>2.1</td>
      <td>2093</td>
      <td>1.9</td>
      <td>2371</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>mgoin/pixtral-12b</th>
      <td></td>
      <td>3.0</td>
      <td>676</td>
      <td>2.4</td>
      <td>825</td>
      <td>2.3</td>
      <td>859</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w8a8</th>
      <td>1.38</td>
      <td>2.2</td>
      <td>904</td>
      <td>1.7</td>
      <td>1159</td>
      <td>1.7</td>
      <td>1201</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w4a16</th>
      <td>1.83</td>
      <td>1.8</td>
      <td>1096</td>
      <td>1.3</td>
      <td>1557</td>
      <td>1.2</td>
      <td>1702</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>mgoin/pixtral-12b</th>
      <td></td>
      <td>1.8</td>
      <td>595</td>
      <td>1.5</td>
      <td>732</td>
      <td>1.4</td>
      <td>764</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-FP8-Dynamic</th>
      <td>1.35</td>
      <td>1.4</td>
      <td>767</td>
      <td>1.1</td>
      <td>1008</td>
      <td>1.0</td>
      <td>1056</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w4a16</th>
      <td>1.37</td>
      <td>1.4</td>
      <td>787</td>
      <td>1.1</td>
      <td>1018</td>
      <td>1.0</td>
      <td>1065</td>
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
      <th>mgoin/pixtral-12b</th>
      <td></td>
      <td>0.6</td>
      <td>2632</td>
      <td>0.9</td>
      <td>4108</td>
      <td>1.1</td>
      <td>4774</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w8a8</th>
      <td>1.50</td>
      <td>0.9</td>
      <td>3901</td>
      <td>1.4</td>
      <td>6160</td>
      <td>1.6</td>
      <td>7292</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w4a16</th>
      <td>1.41</td>
      <td>0.6</td>
      <td>2890</td>
      <td>1.3</td>
      <td>5758</td>
      <td>1.8</td>
      <td>8312</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>mgoin/pixtral-12b</th>
      <td></td>
      <td>1.1</td>
      <td>2291</td>
      <td>1.8</td>
      <td>3670</td>
      <td>2.1</td>
      <td>4284</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w8a8</th>
      <td>1.38</td>
      <td>1.5</td>
      <td>3096</td>
      <td>2.5</td>
      <td>5076</td>
      <td>3.0</td>
      <td>5965</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w4a16</th>
      <td>1.40</td>
      <td>1.4</td>
      <td>2728</td>
      <td>2.6</td>
      <td>5133</td>
      <td>3.5</td>
      <td>6943</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>BF16</th>
      <td></td>
      <td>2.6</td>
      <td>2877</td>
      <td>4.0</td>
      <td>4372</td>
      <td>4.7</td>
      <td>5095</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-FP8-Dynamic</th>
      <td>1.33</td>
      <td>3.4</td>
      <td>3753</td>
      <td>5.4</td>
      <td>5862</td>
      <td>6.3</td>
      <td>6917</td>
    </tr>
    <tr>
      <th>neuralmagic/pixtral-12b-quantized.w4a16</th>
      <td>1.22</td>
      <td>2.8</td>
      <td>3115</td>
      <td>5.0</td>
      <td>5511</td>
      <td>6.2</td>
      <td>6777</td>
    </tr>
  </tbody>
</table>

**Use case profiles: Image Size (WxH) / prompt tokens / generation tokens

**QPS: Queries per second.

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).