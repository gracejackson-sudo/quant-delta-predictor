---
tags:
- vllm
- vision
- audio
- fp8
license: mit
base_model: google/gemma-3n-E2B-it
library_name: transformers
---

# RedHatAI/gemma-3n-E2B-it-FP8-Dynamic

## Model Overview
- **Model Architecture:** gemma-3n-E2B-it
  - **Input:** Audio-Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 08/01/2025
- **Version:** 1.0
- **Model Developers:** RedHatAI

Quantized version of [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it).

### Model Optimizations

This model was obtained by quantizing the weights of [google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) to FP8 data type, ready for inference with vLLM >= 0.10.0

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="RedHatAI/gemma-3n-E2B-it-FP8-Dynamic",
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
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# Load model.
model_id = "google/gemma-3n-E2B-it"
model = Gemma3nForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Recipe
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
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
    ),
]

SAVE_DIR = f"{model_id.split('/')[1]}-{recipe[0].scheme}"

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    recipe=recipe,
    trust_remote_code_model=True,
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
      <th>FP8 Dynamic</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>arc_challenge</td>
      <td>50.60</td>
      <td>50.09</td>
      <td>99.00%</td>
    </tr>
    <tr>
      <td>gsm8k</td>
      <td>48.07</td>
      <td>54.51</td>
      <td>113.40%</td>
    </tr>
    <tr>
      <td>hellaswag</td>
      <td>67.78</td>
      <td>65.67</td>
      <td>96.89%</td>
    </tr>
    <tr>
      <td>mmlu</td>
      <td>59.92</td>
      <td>60.16</td>
      <td>100.40%</td>
    </tr>
    <tr>
      <td>truthfulqa_mc2</td>
      <td>49.98</td>
      <td>49.48</td>
      <td>99.00%</td>
    </tr>
    <tr>
      <td>winogrande</td>
      <td>65.11</td>
      <td>63.85</td>
      <td>98.06%</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td>56.91</td>
      <td>57.29</td>
      <td><b>100.67%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>Leaderboard</b></td>
      <td>bbh</td>
      <td>53.32</td>
      <td>52.99</td>
      <td>99.38%</td>
    </tr>
    <tr>
      <td>mmlu_pro</td>
      <td>29.76</td>
      <td>29.36</td>
      <td>98.66%</td>
    </tr>
    <tr>
      <td>musr</td>
      <td>34.52</td>
      <td>35.85</td>
      <td>103.85%</td>
    </tr>
    <tr>
      <td>ifeval</td>
      <td>80.22</td>
      <td>80.58</td>
      <td>100.45%</td>
    </tr>
    <tr>
      <td>gpqa</td>
      <td>30.54</td>
      <td>29.36</td>
      <td>96.14%</td>
    </tr>
    <tr>
      <td>math_hard</td>
      <td>34.52</td>
      <td>34.97</td>
      <td>101.30%</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td>43.81</td>
      <td>43.85</td>
      <td><b>100.09%</b></td>
    </tr>
  </tbody>
</table>
