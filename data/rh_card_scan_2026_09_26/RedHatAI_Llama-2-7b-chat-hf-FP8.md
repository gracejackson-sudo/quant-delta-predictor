---
tags:
- fp8
- vllm
license: llama2
---

# Llama-2-7b-chat-hf-FP8

## Model Overview
- **Model Architecture:** Llama-2-7b-chat-hf
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 6/26/2024
- **Version:** 1.0
- **License(s):** [llama2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/LICENSE.txt)
- **Model Developers:** Neural Magic

Quantized version of [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
It achieves an average score of 53.28 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 53.43.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) to FP8 data type, ready for inference with vLLM >= 0.5.0.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-tensor quantization is applied, in which a single linear scaling maps the FP8 representations of the quantized weights and activations.
[AutoFP8](https://github.com/neuralmagic/AutoFP8) is used for quantization with 10 repeats of every token in random order.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Llama-2-7b-chat-hf-FP8"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False)

llm = LLM(model=model_id)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [AutoFP8 with calibration samples from ultrachat](https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py), as presented in the code snipet below.
Although AutoFP8 was used for this particular model, Neural Magic is transitioning to using [llm-compressor](https://github.com/vllm-project/llm-compressor) which supports several quantization schemes and models not supported by AutoFP8.

```python
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import torch

from auto_fp8 import AutoFP8ForCausalLM, BaseQuantizeConfig

MODEL_DIR = "meta-llama/Llama-2-7b-chat-hf"
final_model_dir = MODEL_DIR.split("/")[-1]

CONTEXT_LENGTH = 4096
NUM_SAMPLES = 512
NUM_REPEATS = 10

pretrained_model_dir = MODEL_DIR
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True, model_max_length=CONTEXT_LENGTH)
tokenizer.pad_token = tokenizer.eos_token

tokenizer_num_tokens = len(list(tokenizer.get_vocab().values()))
total_token_samples = NUM_REPEATS * tokenizer_num_tokens
num_random_samp = -(-total_token_samples // CONTEXT_LENGTH)

input_ids = np.tile(np.arange(tokenizer_num_tokens), NUM_REPEATS + 1)[:num_random_samp * CONTEXT_LENGTH]
np.random.shuffle(input_ids)
input_ids = input_ids.reshape(num_random_samp, CONTEXT_LENGTH)
input_ids = torch.tensor(input_ids, dtype=torch.int64).to("cuda")

quantize_config = BaseQuantizeConfig(
    quant_method="fp8",
    activation_scheme="static",
)

examples = input_ids

model = AutoFP8ForCausalLM.from_pretrained(pretrained_model_dir, quantize_config=quantize_config)

model.quantize(examples)

quantized_model_dir = f"{final_model_dir}-FP8"
model.save_quantized(quantized_model_dir)
```

## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-2-7b-chat-hf-FP8",dtype=auto,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Llama-2-7b-chat-hf</strong>
   </td>
   <td><strong>Llama-2-7b-chat-hf-FP8(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>47.39
   </td>
   <td>47.33
   </td>
   <td>99.87%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>53.33
   </td>
   <td>53.58
   </td>
   <td>100.4%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>23.28
   </td>
   <td>22.82
   </td>
   <td>98.02%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>78.64
   </td>
   <td>78.31
   </td>
   <td>99.58%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>72.38
   </td>
   <td>72.22
   </td>
   <td>99.77%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>45.58
   </td>
   <td>45.73
   </td>
   <td>100.3%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>53.43</strong>
   </td>
   <td><strong>53.28</strong>
   </td>
   <td><strong>99.72%</strong>
   </td>
  </tr>
</table>