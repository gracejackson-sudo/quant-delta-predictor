---
tags:
- vllm
- vision
- audio
- int8
license: mit
base_model: google/gemma-3n-E2B-it
library_name: transformers
---

# RedHatAI/gemma-3n-E2B-it-quantized.w8a8

## Model Overview
- **Model Architecture:** gemma-3n-E2B-it
  - **Input:** Audio-Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
  - **Activation quantization:** INT8
- **Release Date:** 08/01/2025
- **Version:** 1.0
- **Model Developers:** RedHatAI

Quantized version of [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) to INT8 data type, ready for inference with vLLM >= 0.10.0

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="RedHatAI/gemma-3n-E2B-it-quantized.w8a8",
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

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below.

<details>
  <summary>Model Creation Code</summary>
  
```python
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils import dispatch_for_generation

# Load model.
model_id = "google/gemma-3n-E2B-it"
model = Gemma3nForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Oneshot arguments
DATASET_ID = "flickr30k"
DATASET_SPLIT = {"calibration": "test[:512]"}
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

dampening_frac=0.01

# Recipe
recipe = [
    GPTQModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=[
            "re:.*embed_audio.*",
            "re:.*embed_vision.*",
            "re:.*audio_tower.*",
            "re:.*vision_tower.*",
            "re:.*altup.*",
            "re:.*lm_head.*",
            "re:.*laurel.*",
            "re:model\.language_model\.layers\.\d+\.per_layer_input_gate",
            "re:model\.language_model\.layers\.\d+\.per_layer_projection",
            "model.language_model.per_layer_model_projection",
        ],
        dampening_frac=dampening_frac
    ),
]

SAVE_DIR = f"{model_id.split('/')[1]}-quantized.{recipe[0].scheme}"

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
    # gemma3n has broken weight offloading which is required by the sequential pipeline
    pipeline="basic",
    # gemma3n does not support untying word embeddings
    tie_word_embeddings=True,
    output_dir=SAVE_DIR,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
```
</details>

## Evaluation

The model was evaluated using [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness) for OpenLLM V1 and V2 text-based benchmarks. The evaluations were conducted using the following commands:

<details>
<summary>Evaluation Commands</summary>

### OpenLLM V1
  
```
lm_eval \
  --model vllm \
  --model_args pretrained="<model_name>",dtype=auto,add_bos_token=false,max_model_len=4096,gpu_memory_utilization=0.8,enable_chunked_prefill=True,enforce_eager=True,trust_remote_code=True \
  --tasks openllm \
  --batch_size auto \
  --apply_chat_template \
  --fewshot_as_multiturn

```

### Leaderboard V2

```
lm_eval \
  --model vllm \
  --model_args pretrained="<model_name>",dtype=auto,add_bos_token=false,max_model_len=15000,gpu_memory_utilization=0.5,enable_chunked_prefill=True,enforce_eager=True,trust_remote_code=True \
  --tasks leaderboard \
  --batch_size auto \
  --apply_chat_template \
  --fewshot_as_multiturn

```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>google/gemma-3n-E2B-it</th>
      <th>RedHatAI/gemma-3n-E2B-it-quantized.w8a8</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>arc_challenge</td>
      <td>50.60</td>
      <td>50.60</td>
      <td>100.00%</td>
    </tr>
    <tr>
      <td>gsm8k</td>
      <td>48.07</td>
      <td>51.40</td>
      <td>106.93%</td>
    </tr>
    <tr>
      <td>hellaswag</td>
      <td>67.78</td>
      <td>65.45</td>
      <td>96.56%</td>
    </tr>
    <tr>
      <td>mmlu</td>
      <td>59.92</td>
      <td>60.10</td>
      <td>100.30%</td>
    </tr>
    <tr>
      <td>truthfulqa_mc2</td>
      <td>49.98</td>
      <td>49.62</td>
      <td>99.28%</td>
    </tr>
    <tr>
      <td>winogrande</td>
      <td>65.11</td>
      <td>64.56</td>
      <td>99.15%</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td>56.91</td>
      <td>56.96</td>
      <td><b>100.08%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>Leaderboard</b></td>
      <td>bbh</td>
      <td>53.32</td>
      <td>52.56</td>
      <td>98.57%</td>
    </tr>
    <tr>
      <td>mmlu_pro</td>
      <td>29.76</td>
      <td>29.22</td>
      <td>98.19%</td>
    </tr>
    <tr>
      <td>musr</td>
      <td>34.52</td>
      <td>35.58</td>
      <td>103.07%</td>
    </tr>
    <tr>
      <td>ifeval</td>
      <td>80.22</td>
      <td>81.06</td>
      <td>101.05%</td>
    </tr>
    <tr>
      <td>gpqa</td>
      <td>30.54</td>
      <td>29.11</td>
      <td>95.32%</td>
    </tr>
    <tr>
      <td>math_hard</td>
      <td>34.52</td>
      <td>33.76</td>
      <td>97.80%</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td>43.81</td>
      <td>43.55</td>
      <td><b>99.40%</b></td>
    </tr>
  </tbody>
</table>
