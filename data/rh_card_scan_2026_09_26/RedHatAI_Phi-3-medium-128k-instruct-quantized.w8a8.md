---
language:
- en
pipeline_tag: text-generation
license: mit
---

# Phi-3-medium-128k-instruct-quantized.w8a8

## Model Overview
- **Model Architecture:** Phi-3
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/11/2024
- **Version:** 1.0
- **License(s):** [MIT](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md)
- **Model Developers:** Neural Magic

Quantized version of [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct), a 14 billion-parameter open model trained using the Phi-3 datasets.
It achieves an average score of 73.90 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 74.10.

### Model Optimizations

This model was obtained by quantizing the weights of [Phi-3-medium-128k-instruct](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between INT8 and floating point representations for each output channel dimension.
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between INT8 and floating point representations.
Linear scaling factors are computed via by minimizing the mean squarred error (MSE).
The [SmoothQuant](https://arxiv.org/abs/2211.10438) algorithm is used to alleviate outliers in the activations, whereas rhe [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization.
Both algorithms are implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.
GPTQ used a 1% damping factor and 512 sequences sequences taken from Neural Magic's [LLM compression calibration dataset](https://huggingface.co/datasets/neuralmagic/LLM_compression_calibration).


## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below (using 2 GPUs).

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Phi-3-medium-128k-instruct-quantized.w8a8"
number_gpus = 2

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, trust_remote_code=True, max_model_len=8196, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.


## Creation

This model was created by using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as presented in the code snipet below.

```python
from transformers import AutoTokenizer
from datasets import Dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import random

model_id = "microsoft/Phi-3-medium-128k-instruct"

num_samples = 512
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_fn(example):
  return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.shuffle().select(range(num_samples))
ds = ds.map(preprocess_fn)

recipe = [
  SmoothQuantModifier(
    smoothing_strength=0.8,
    mappings=[
      [["re:.*qkv_proj"], "re:.*input_layernorm"],
      [["re:.*gate_up_proj"], "re:.*post_attention_layernorm"],
    ],
  ),
  GPTQModifier(
    sequential=True,
    targets="Linear",
    scheme="W8A8",
    ignore=["lm_head"],
    dampening_frac=0.01,
    observer="mse",
  )
]

model = SparseAutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="auto",
  trust_remote_code=True,
)

oneshot(
  model=model,
  dataset=ds,
  recipe=recipe,
  max_seq_length=max_seq_len,
  num_calibration_samples=num_samples,
)

model.save_pretrained("Phi-3-medium-128k-instruct-quantized.w8a8")
```



## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command (using 2 GPUs):
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Phi-3-medium-128k-instruct-quantized.w8a8",dtype=auto,gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Phi-3-medium-128k-instruct </strong>
   </td>
   <td><strong>Phi-3-medium-128k-instruct-quantized.w8a8 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>76.69
   </td>
   <td>76.74
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>69.45
   </td>
   <td>69.37
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>85.22
   </td>
   <td>84.15
   </td>
   <td>98.7%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>85.10
   </td>
   <td>84.76
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>73.56
   </td>
   <td>73.80
   </td>
   <td>100.3%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>54.57
   </td>
   <td>54.57
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>74.10</strong>
   </td>
   <td><strong>73.90</strong>
   </td>
   <td><strong>99.7%</strong>
   </td>
  </tr>
</table>