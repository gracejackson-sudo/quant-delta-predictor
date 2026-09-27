---
tags:
- fp8
- vllm
language:
- en
- zh
pipeline_tag: text-generation
base_model: zai-org/GLM-4.6
---

# GLM-4.6-FP8-dynamic

## Model Overview
- **Model Architecture:** zai-org/GLM-4.6
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [zai-org/GLM-4.6](https://huggingface.co/zai-org/GLM-4.6) to FP8 data type, ready for inference with vLLM>=0.11.0

Only the weights and activations of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/GLM-4.6-FP8-dynamic"
number_gpus = 4

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [LLM Compressor with calibration samples from UltraChat](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/llama3_example.py), as presented in the code snipet below.

<details>
  
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "zai-org/GLM-4.6"

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", trust_remote_code=True, device_map=None
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Configure the quantization algorithm and scheme.
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore = [
        "lm_head",
    ]
)

# Apply quantization.
# FP8_DYNAMIC uses data-free quantization, so no calibration dataset needed
oneshot(model=model, recipe=recipe, trust_remote_code_model=True)

# Save to disk in compressed-tensors format.
SAVE_DIR = "./" + MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

```
</details>

## Evaluation

This model was evaluated on the well-known text benchmarks using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness). The Reasoning evals were done using [ligheval](https://github.com/neuralmagic/lighteval).

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>zai-org/GLM-4.6-FP8</th>
      <th>RedHatAI/GLM-4.6-FP8-dynamic (this model)</th>
      <th>Recovery</th>
    </tr>
  </thead>
<tbody>
<!-- OpenLLM V1 -->
<tr>
  <td rowspan="2"><b>Leaderboard</b></td>
  <td>MMLU Pro</td>
  <td>50.65%</td>
  <td>50.25%</td>
  <td>99.21%</td>
</tr>
<tr>
  <td>IFEVAL</td>
  <td>91.97</td>
  <td>92.69%</td>
  <td>100.78%</td>
</tr>
<tr>
  <td rowspan="6"><b>Reasoning</b></td>
  <td>AIME25</td>
  <td>96.67%</td>
  <td>93.33%</td>
  <td>96.54%<td>
</tr>
<tr>
  <td>Math-500 (0-shot)</td>
  <td>88.80%</td>
  <td>90.40%</td>
  <td>101.80%</%</td>
</tr>
<tr>
  <td>GPQA (Diamond, 0-shot)</td>
  <td>81.82%</td>
  <td>77.78%</td>
  <td>95.06%</td>
</tr>
</tbody>
</table>



### Reproduction

The results were obtained using the following commands:

<details>

#### Leaderboard

```
lm_eval --model local-chat-completions \
  --tasks mmlu_pro  \
  --model_args "model=RedHatAI/GLM-4.6-FP8-dynamic,max_length=90000,base_url=http://0.0.0.0:3758/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path ./ \
  --seed 42 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,max_gen_toks=64000"


lm_eval --model local-chat-completions \
  --tasks leaderboard_ifeval  \
  --model_args "model=RedHatAI/GLM-4.6-FP8-dynamic,max_length=90000,base_url=http://0.0.0.0:3758/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 5 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path ./ \
  --seed 42 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,max_gen_toks=64000"
```


#### Reasoning
```
litellm_config.yaml:

model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/redhatai-glm-4.6-FP8-dynamic"
  base_url: "http://0.0.0.0:3759/v1"
  api_key: ""
  timeout: 3600
  concurrent_requests: 128
  generation_parameters:
    temperature: 1.0
    max_new_tokens: 131072
    top_p: 0.95
    seed: 0

lighteval endpoint litellm litellm_config.yaml \
  "aime25|0,math_500|0,gpqa:diamond|0" \
  --output-dir ./ \
  --save-details
```

</details>