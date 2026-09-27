---
license: mit
tags:
- qwen
- qwq
- fp8
- vllm
base_model: Qwen/QwQ-32B
library_name: transformers
---

# QwQ-32B-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 3/6/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B).


### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-channel scheme, whereas quantizations are quantized using a symmetric per-token scheme.
[LLM Compressor](https://github.com/vllm-project/llm-compressor) is used for quantization.


## Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

number_gpus = 1
model_name = "neuralmagic/QwQ-32B-FP8-dynamic"

tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])
llm = LLM(model=model_name, tensor_parallel_size=number_gpus, trust_remote_code=True)

messages_list = [
    [{"role": "user", "content": "Who are you? Please respond in pirate speak!"}],
]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
import os

# Load model
model_stub = "Qwen/QwQ-32B"
model_name = model_stub.split("/")[-1]

model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_stub)

# Configure the quantization algorithm and scheme
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

# Apply quantization
oneshot(
    model=model,
    recipe=recipe,
)

# Save to disk in compressed-tensors format
save_path = model_name + "-FP8-dynamic
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")
```


### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Qwen/QwQ-32B</th>
      <th>neuralmagic/QwQ-32B-FP8-dynamic</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td rowspan="4"><b>Reasoning</b></td>
<td>AIME 2024 (pass@1)</td>
<td>78.66</td>
<td>79.40</td>
<td>100.94%</td>
</tr>
<tr>
<td>MATH-500 (pass@1)</td>
<td>97.39</td>
<td>97.44</td>
<td>100.05%</td>
</tr>
<tr>
<td>GPQA Diamond (pass@1)</td>
<td>64.72</td>
<td>63.21</td>
<td>97.66%</td>
</tr>
<tr>
<td><b>Average Score</b></td>
<td><b>80.25</b></td>
<td><b>80.05</b></td>
<td><b>99.75%</b></td>
</tr>
    
  </tbody>
</table>
