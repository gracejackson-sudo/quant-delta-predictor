---
tags:
- fp8
- vllm
license: other
license_name: deepseek-license
license_link: https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/LICENSE-MODEL
---

# DeepSeek-Coder-V2-Instruct-FP8

## Model Overview
- **Model Architecture:** DeepSeek-Coder-V2-Instruct
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Meta-Llama-3-7B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-7B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/22/2024
- **Version:** 1.0
- **License(s):** [deepseek-license](https://github.com/deepseek-ai/DeepSeek-Coder-V2/blob/main/LICENSE-MODEL)
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-Coder-V2-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct).
<!-- It achieves an average score of 73.19 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 73.48. -->
It achieves an average score of 88.98 on the [HumanEval+](https://github.com/openai/human-eval?tab=readme-ov-file) benchmark, whereas the unquantized model achieves 87.63.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [DeepSeek-Coder-V2-Instruct](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct) to FP8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%. In particular, this model can now be loaded and evaluated with only 4xH100 GPUs, as opposed to 8.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-tensor quantization is applied, in which a single linear scaling maps the FP8 representations of the quantized weights and activations.
[AutoFP8](https://github.com/neuralmagic/AutoFP8) is used for quantization with 512 sequences of UltraChat.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 4
model_name = "neuralmagic/DeepSeek-Coder-V2-Instruct-FP8"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])

messages_list = [
    [{"role": "user", "content": "Who are you? Please respond in pirate speak!"}],
]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [AutoFP8 with calibration samples from ultrachat](https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py) with expert gates kept at original precision, as presented in the code snipet below.
Notably, a custom device map had to be used, as the model was being incorrectly loaded otherwise.
Although AutoFP8 was used for this particular model, Neural Magic is transitioning to using [llm-compressor](https://github.com/vllm-project/llm-compressor) which supports several quantization schemes and models not supported by AutoFP8.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

pretrained_model_dir = "deepseek-ai/DeepSeek-Coder-V2-Instruct"
quantized_model_dir = "DeepSeek-Coder-V2-Instruct-FP8"

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

device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
}
for i in range(1, 60):
    device_map[f"model.layers.{i}"] = i//8

device_map["model.norm"] = 7
device_map["lm_head"] = 7

model = AutoFP8ForCausalLM.from_pretrained(
    pretrained_model_dir, quantize_config=quantize_config, device_map = device_map
)
model.quantize(examples)
model.save_quantized(quantized_model_dir)
```

## Evaluation

The model was evaluated on the [HumanEval+](https://github.com/openai/human-eval?tab=readme-ov-file) benchmark with the [Neural Magic fork](https://github.com/neuralmagic/evalplus) of the [EvalPlus implementation of HumanEval+](https://github.com/evalplus/evalplus) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
python codegen/generate.py --model neuralmagic/DeepSeek-Coder-V2-Instruct-FP8 --temperature 0.2 --n_samples 50 --resume --root ~ --dataset humaneval
python evalplus/sanitize.py ~/humaneval/neuralmagic--DeepSeek-Coder-V2-Instruct-FP8_vllm_temp_0.2
evalplus.evaluate --dataset humaneval --samples ~/humaneval/neuralmagic--DeepSeek-Coder-V2-Instruct-FP8_vllm_temp_0.2-sanitized
```

### Accuracy

#### HumanEval+ evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>DeepSeek-Coder-V2-Instruct</strong>
   </td>
   <td><strong>DeepSeek-Coder-V2-Instruct-FP8(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>base pass@1
   </td>
   <td>88.2
   </td>
   <td>87.6
   </td>
   <td>99.32%
   </td>
  </tr>
  <tr>
   <td>base pass@10
   </td>
   <td>92.3
   </td>
   <td>94.7
   </td>
   <td>102.60%
   </td>
  </tr>
  <tr>
   <td>base+extra pass@1
   </td>
   <td>83.3
   </td>
   <td>83.2
   </td>
   <td>99.88%
   </td>
  </tr>
  <tr>
   <td>base+extra pass@10
   </td>
   <td>86.7
   </td>
   <td>90.4
   </td>
   <td>104.27%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>87.63</strong>
   </td>
   <td><strong>88.98</strong>
   </td>
   <td><strong>101.5%</strong>
   </td>
  </tr>
</table>