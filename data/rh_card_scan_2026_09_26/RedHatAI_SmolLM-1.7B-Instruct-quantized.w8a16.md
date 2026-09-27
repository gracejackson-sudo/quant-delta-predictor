---
library_name: transformers
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- int8
- vllm
base_model: HuggingFaceTB/SmolLM-1.7B-Instruct
---

# SmolLM-1.7B-Instruct-quantized.w8a16

## Model Overview
- **Model Architecture:** Llama
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [SmolLM-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 8/16/2024
- **Version:** 1.0
- **License(s):** [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Model Developers:** Neural Magic

Quantized version of [SmolLM-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct).
It achieves an average score of 41.83 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 41.76.

### Model Optimizations

This model was obtained by quantizing the weights of [SmolLM-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct) to INT8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights of the linear operators within transformers blocks are quantized. Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the INT8 and floating point representations of the quantized weights.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.
GPTQ used a 1% damping factor and 1,024 sequences of 2,048 random tokens.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/SmolLM-1.7B-Instruct-quantized.w8a16"

sampling_params = SamplingParams(temperature=0.6, top_p=0.92, max_tokens=100)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "List the steps to bake a chocolate cake from scratch."},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_id)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as presented in the code snipet below.

```python
from transformers import AutoTokenizer
from datasets import Dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import random

model_id = "HuggingFaceTB/SmolLM-1.7B-Instruct"

num_samples = 1024
max_seq_len = 2048

tokenizer = AutoTokenizer.from_pretrained(model_id)

max_token_id = len(tokenizer.get_vocab()) - 1
input_ids = [[random.randint(0, max_token_id) for _ in range(max_seq_len)] for _ in range(num_samples)]
attention_mask = num_samples * [max_seq_len * [1]]
ds = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})

recipe = GPTQModifier(
  targets="Linear",
  scheme="W8A16",
  ignore=["lm_head"],
  dampening_frac=0.01,
)

model = SparseAutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="auto",
)

oneshot(
  model=model,
  dataset=ds,
  recipe=recipe,
  max_seq_length=max_seq_len,
  num_calibration_samples=num_samples,
)
model.save_pretrained("SmolLM-1.7B-Instruct-quantized.w8a16")
```

## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/SmolLM-1.7B-Instruct-quantized.w8a16",dtype=auto,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>SmolLM-1.7B-Instruct-quantized</strong>
   </td>
   <td><strong>SmolLM-1.7B-Instruct-quantized.w8a16 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>28.10
   </td>
   <td>28.42
   </td>
   <td>101.1%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>49.06
   </td>
   <td>49.32
   </td>
   <td>100.5%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>4.93
   </td>
   <td>4.93
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>66.96
   </td>
   <td>66.89
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>61.01
   </td>
   <td>61.17
   </td>
   <td>100.3%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>40.28
   </td>
   <td>40.25
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>41.76</strong>
   </td>
   <td><strong>41.83</strong>
   </td>
   <td><strong>100.2%</strong>
   </td>
  </tr>
</table>