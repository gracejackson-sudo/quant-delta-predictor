---
tags:
- fp8
- vllm
license: other
license_name: deepseek-license
license_link: https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/LICENSE-MODEL
---

# DeepSeek-Coder-V2-Lite-Instruct-FP8

## Model Overview
- **Model Architecture:** DeepSeek-Coder-V2-Lite-Instruct
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Meta-Llama-3-7B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-7B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/18/2024
- **Version:** 1.0
- **License(s):** [deepseek-license](https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/LICENSE-MODEL)
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct).
<!-- It achieves an average score of 73.19 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 73.48. -->
It achieves an average score of 79.60 on the [HumanEval+](https://github.com/openai/human-eval?tab=readme-ov-file) benchmark, whereas the unquantized model achieves 79.33.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [DeepSeek-Coder-V2-Lite-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) to FP8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-tensor quantization is applied, in which a single linear scaling maps the FP8 representations of the quantized weights and activations.
[AutoFP8](https://github.com/neuralmagic/AutoFP8) is used for quantization with 512 sequences of UltraChat.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_id, trust_remote_code=True, max_model_len=4096)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [AutoFP8 with calibration samples from ultrachat](https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py) with expert gates kept at original precision, as presented in the code snipet below.
Although AutoFP8 was used for this particular model, Neural Magic is transitioning to using [llm-compressor](https://github.com/vllm-project/llm-compressor) which supports several quantization schemes and models not supported by AutoFP8.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
quantized_model_dir = "DeepSeek-Coder-V2-Lite-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True, model_max_length=4096)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("mgoin/ultrachat_2k", split="train_sft").select(range(512))
examples = [tokenizer.apply_chat_template(batch["messages"], tokenize=False) for batch in ds]
examples = tokenizer(examples, padding=True, truncation=True, return_tensors="pt").to("cuda")

quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="static"
    ignore_patterns=["re:.*lm_head"],
)

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
```

## Evaluation

The model was evaluated on the [HumanEval+](https://github.com/openai/human-eval?tab=readme-ov-file) benchmark with the [Neural Magic fork](https://github.com/neuralmagic/evalplus) of the [EvalPlus implementation of HumanEval+](https://github.com/evalplus/evalplus) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
python codegen/generate.py --model neuralmagic/DeepSeek-Coder-V2-Lite-Instruct-FP8 --temperature 0.2 --n_samples 50 --resume --root ~ --dataset humaneval
python evalplus/sanitize.py ~/humaneval/neuralmagic--DeepSeek-Coder-V2-Lite-Instruct-FP8_vllm_temp_0.2
evalplus.evaluate --dataset humaneval --samples ~/humaneval/neuralmagic--DeepSeek-Coder-V2-Lite-Instruct-FP8_vllm_temp_0.2-sanitized
```

### Accuracy

#### HumanEval+ evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>DeepSeek-Coder-V2-Lite-Instruct</strong>
   </td>
   <td><strong>DeepSeek-Coder-V2-Lite-Instruct-FP8(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>base pass@1
   </td>
   <td>80.8
   </td>
   <td>79.3
   </td>
   <td>98.14%
   </td>
  </tr>
  <tr>
   <td>base pass@10
   </td>
   <td>83.4
   </td>
   <td>84.6
   </td>
   <td>101.4%
   </td>
  </tr>
  <tr>
   <td>base+extra pass@1
   </td>
   <td>75.8
   </td>
   <td>74.9
   </td>
   <td>98.81%
   </td>
  </tr>
  <tr>
   <td>base+extra pass@10
   </td>
   <td>77.3
   </td>
   <td>79.6
   </td>
   <td>102.9%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>79.33</strong>
   </td>
   <td><strong>79.60</strong>
   </td>
   <td><strong>100.3%</strong>
   </td>
  </tr>
</table>