---
license: mit
library_name: vllm
base_model:
- deepseek-ai/DeepSeek-R1
pipeline_tag: text-generation
tags:
- deepseek
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
- GPTQ
---

# DeepSeek-R1-quantized.w4a16

## Model Overview
- **Model Architecture:** DeepseekV3ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** None
  - **Weight quantization:** INT4
- **Release Date:** 04/15/2025
- **Version:** 1.0
- **Model Developers:** Red Hat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing weights of [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) to INT4 data type.
This optimization reduces the number of bits used to represent weights from 8 to 4, reducing GPU memory requirements (by approximately 50%).
Weight quantization also reduces disk size requirements by approximately 50%.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/DeepSeek-R1-quantized.w4a16"
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

We created this model using **MoE-Quant**, a library developed jointly with **ISTA** and tailored for the quantization of very large Mixture-of-Experts (MoE) models.  

For more details, please refer to the [MoE-Quant repository](https://github.com/IST-DASLab/MoE-Quant).

## Evaluation

The model was evaluated on the OpenLLM leaderboard task (v1) via [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and on popular reasoning tasks (AIME 2024, MATH-500, GPQA-Diamond) via [LightEval](https://github.com/huggingface/open-r1).
For reasoning evaluations, we estimate pass@1 based on 10 runs with different seeds.

<details>
  <summary>Evaluation details</summary>

  **OpenLLM v1**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/DeepSeek-R1-quantized.w4a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --batch_size auto
  ```

  **Reasoning Benchmarks**
  ```
  export MODEL_ARGS="pretrained=RedHatAI/DeepSeek-R1-quantized.w4a16,dtype=bfloat16,max_model_length=38768,gpu_memory_utilization=0.8,tensor_parallel_size=8,add_special_tokens=false,generation_parameters={\"max_new_tokens\":32768,\"temperature\":0.6,\"top_p\":0.95,\"seed\":42}"
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  lighteval vllm $MODEL_ARGS "custom|aime24|0|0,custom|math_500|0|0,custom|gpqa:diamond|0|0" \
      --custom-tasks src/open_r1/evaluate.py \
      --use-chat-template \
      --output-dir $OUTPUT_DIR
  ```

</details>

### Accuracy

|                             | Recovery (%) | deepseek/DeepSeek-R1 | RedHatAI/DeepSeek-R1-quantized.w4a16<br>(this model) |
| --------------------------- | :----------: | :------------------: | :--------------------------------------------------: |
| ARC-Challenge<br>25-shot    | 100.00          | 72.53                | 72.53                                                |
| GSM8k<br>5-shot             | 99.76        | 95.91                | 95.68                                                |
| HellaSwag<br>10-shot        | 100.07       | 89.30                | 89.36                                                |
| MMLU<br>5-shot              | 99.74        | 87.22                | 86.99                                                |
| TruthfulQA<br>0-shot        | 100.83       | 59.28                | 59.77                                                |
| WinoGrande<br>5-shot        | 101.65       | 82.00                | 83.35                                                |
| **OpenLLM v1<br>Average Score** | **100.30**        | **81.04**                | **81.28**                                            |
| AIME 2024<br>pass@1         | 98.30         | 78.33                | 77.00                                                |
| MATH-500<br>pass@1          | 99.84        | 97.24                | 97.08                                                |
| GPQA Diamond<br>pass@1      | 98.01        | 73.38                | 71.92                                                |
| **Reasoning<br>Average Score**  | **98.81**        | **82.99**                | **82.00**                                                |

