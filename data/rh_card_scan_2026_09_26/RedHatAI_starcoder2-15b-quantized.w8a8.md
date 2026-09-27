---
pipeline_tag: text-generation
datasets:
- bigcode/the-stack-v2-train
license: bigcode-openrail-m
library_name: transformers
tags:
- code
model-index:
- name: starcoder2-15b-quantized.w8a8
  results:
  - task:
      type: text-generation
    dataset:
      name: HumanEval+
      type: humanevalplus
    metrics:
    - type: pass@1
      value: 38.1
  - task:
      type: text-generation
    dataset:
      name: HumanEval
      type: humaneval
    metrics:
    - type: pass@1
      value: 44.6
---

# starcoder2-3b-quantized.w8a8

## Model Overview
- **Model Architecture:** StarCoder2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use. Similarly to [starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b), this model is intended for code generation and is _not_ an instruction model. Commands like "Write a function that computes the square root." do not work well.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 8/1/2024
- **Version:** 1.0
- **License(s):** bigcode-openrail-m
- **Model Developers:** Neural Magic

Quantized version of [starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b).
It achieves a HumanEval pass@1 of 44.6, whereas the unquantized model achieves 44.8 when evaluated under the same conditions.

### Model Optimizations

This model was obtained by quantizing the weights of [starcoder2-15b](https://huggingface.co/bigcode/starcoder2-15b) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between INT8 and floating point representations for each output channel dimension.
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between INT8 and floating point representations.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.
GPTQ used a 1% damping factor and 256 sequences of 8,192 random tokens.


## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/starcoder2-15b-quantized.w8a8"
number_gpus = 1

sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompts = ["def print_hello_world():"]

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

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

model_id = "bigcode/starcoder2-15b"

num_samples = 256
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_id)

max_token_id = len(tokenizer.get_vocab()) - 1
input_ids = [[random.randint(0, max_token_id) for _ in range(max_seq_len)] for _ in range(num_samples)]
attention_mask = num_samples * [max_seq_len * [1]]
ds = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})

recipe = GPTQModifier(
  targets="Linear",
  scheme="W8A8",
  ignore=["lm_head"],
  dampening_frac=0.01,
)

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
model.save_pretrained("starcoder2-15b-quantized.w8a8")
```


## Evaluation

The model was evaluated on the [HumanEval](https://arxiv.org/abs/2107.03374) and [HumanEval+](https://arxiv.org/abs/2305.01210) benchmarks, using the generation configuration from [Big Code Models Leaderboard](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard).
We used Neural Magic's fork of [evalplus](https://github.com/neuralmagic/evalplus) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following commands:

```
python codegen/generate.py \
  --model neuralmagic/starcoder2-15b-quantized.w8a8 \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --dataset humaneval \
  -- root "."

python3 evalplus/sanitize.py humaneval/neuralmagic--starcoder2-15b-quantized.w8a8_vllm_temp_0.2

evalplus.evaluate --dataset humaneval --samples humaneval/neuralmagic--starcoder2-15b-quantized.w8a8_vllm_temp_0.2-sanitized
```

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>starcoder2-15b</strong>
   </td>
   <td><strong>starcoder2-15b-quantized.w8a8 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>HumanEval pass@1
   </td>
   <td>44.8
   </td>
   <td>44.6
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td>HumanEval pass@10
   </td>
   <td>62.7
   </td>
   <td>63.3
   </td>
   <td>101.0%
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>38.6
   </td>
   <td>38.1
   </td>
   <td>98.7%
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@10
   </td>
   <td>54.9
   </td>
   <td>55.5
   </td>
   <td>101.1%
   </td>
  </tr>
  <tr>
</table>