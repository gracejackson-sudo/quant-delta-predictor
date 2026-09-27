---
library_name: vllm
language:
- ar
- de
- en
- es
- fr
- hi
- id
- it
- pt
- th
- tl
- vi
base_model:
- meta-llama/Llama-4-Maverick-17B-128E-Instruct
pipeline_tag: image-text-to-text
tags:
- facebook
- meta
- pytorch
- llama
- llama4
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
license: other
license_name: llama4
---

# Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16


## Model Overview
- **Model Architecture:** Llama4ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Release Date:** 06/12/2025
- **Version:** 1.0
- **Model Developers:** Red Hat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing weights of [Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) to INT4 data type.
This optimization reduces the number of bits used to represent weights from 16 to 4, reducing GPU memory requirements by approximately 75%.
Weight quantization also reduces disk size requirements by approximately 75%.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.


## Deployment

This model can be deployed efficiently on vLLM.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16"
number_gpus = 8

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Give me a short introduction to large language model."

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompt, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

<details><summary>Creation details</summary>

This model was created by applying a development version [llm-compressor](https://github.com/vllm-project/llm-compressor).
More details will be added as the the code is merged on main.

</details>

## Evaluation

The model was evaluated on the OpenLLM v1 leaderboard task, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
More evaluations are under way.

<details>
  <summary>Evaluation details</summary>

  **OpenLLM v1**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8,gpu_memory_utilization=0.7,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --batch_size auto 
  ```

</details>

### Accuracy

|                                                | Recovery (%)  | meta-llama/Llama-4-Maverick-17B-128E-Instruct | RedHatAI/Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16<br>(this model) |
| ---------------------------------------------- | :-----------: | :-------------------------------------------: | :-----------------------------------------------------------------: |
| ARC-Challenge<br>25-shot                       | 96.6          | 73.55                                         | 71.08                                                               |
| GSM8k<br>5-shot                                | 99.7          | 93.18                                         | 92.87                                                               |
| HellaSwag<br>10-shot                           | 99.6          | 87.27                                         | 86.95                                                               |
| MMLU<br>5-shot                                 | 99.8          | 85.98                                         | 85.78                                                               |
| TruthfulQA<br>0-shot                           | 100.0         | 62.81                                         | 62.85                                                               |
| WinoGrande<br>5-shot                           | 100.5         | 78.53                                         | 78.93                                                               |
| **OpenLLM v1<br>Average Score**                | **99.4**      | **80.22**                                     | **79.74**                                                               |

