---
language:
- multilingual
- ar
- zh
- cs
- da
- nl
- en
- fi
- fr
- de
- he
- hu
- it
- ja
- ko
- 'no'
- pl
- pt
- ru
- es
- sv
- th
- tr
- uk
license: mit
license_name: mit
license_link: https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/LICENSE
name: RedHatAI/Phi-4-mini-instruct-FP8-dynamic
base_model:
- microsoft/Phi-4-mini-instruct
provider: Microsoft
description: This model was obtained by quantizing activation and weights of Phi-4-mini-instruct to FP8 data type.
validated_on:
  - RHOAI 3.4 EA1
  - RHAIIS 3.4 EA1
  - vLLM 0.14.1
readme: https://huggingface.co/RedHatAI/Phi-4-mini-instruct-FP8-dynamic/blob/main/README.md
pipeline_tag: text-generation
tags:
- nlp
- code
- red hat
- FP8
- compressed-tensors
- llm-compressor
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Phi-4-mini-instruct-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>


## Model Overview
- **Model Architecture:** Phi3ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP8
  - **Weight quantization:** FP8
- **Intended Use Cases:** The model is intended for broad multilingual commercial and research use. The model provides uses for general purpose AI systems and applications which require:
1. Memory/compute constrained environments.
2. Latency bound scenarios.
3. Math reasoning and logic.
- **Release Date:** 03/03/2025
- **Version:** 1.0
- **Model Developers:** Red Hat
- **ModelCar Storage URI:** oci://registry.stage.redhat.io/rhai/phi-4-mini-instruct-fp8-dynamic:3.0
- **Validated on vLLM:** 0.14.1
- **Validated on RHAIIS:** 3.4 EA1
- **Validated on RHOAI:** 3.4 EA1

### Model Optimizations

This model was obtained by quantizing activation and weights of [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```bash
vllm serve RedHatAI/Phi-4-mini-instruct-FP8-dynamic --max_model_len 131072
```

```python
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

generated_text = client.chat.completions.create(
    model="RedHatAI/Phi-4-mini-instruct-FP8-dynamic",
    messages=[
        {"role": "user", "content": "Give me a short introduction to large language model."},
    ],
)
print(generated_text.choices[0].message.content)
```

## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from llmcompressor.modifiers.quantization import QuantizationModifier
  from llmcompressor import oneshot
  
  # Load model
  model_stub = "microsoft/Phi-4-mini-instruct"
  model_name = model_stub.split("/")[-1]
  
  tokenizer = AutoTokenizer.from_pretrained(model_stub)
  
  model = AutoModelForCausalLM.from_pretrained(
      model_stub,
      device_map="auto",
      torch_dtype="auto",
  )
  
  # Configure the quantization algorithm and scheme
  recipe = QuantizationModifier(
      targets="Linear",
      scheme="FP8_dynamic",
      ignore=["lm_head"],
  )
  
  # Apply quantization
  oneshot(
      model=model,
      recipe=recipe,
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-FP8-dynamic"
  model.save_pretrained(save_path)
  tokenizer.save_pretrained(save_path)
  print(f"Model and tokenizer saved to: {save_path}")
  ```
</details>
 


## Evaluation

The model was evaluated on the Mathh 500 benchmarks using [lighteval](https://github.com/huggingface/lighteval), and on GSM8k-Platinum, MMLU CoT, MMLU-Pro, and IFEval using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
In both cases [vLLM](https://vllm.ai) is used as the backend

<details>
  <summary>Evaluation commands</summary>

### Start vLLM server
```bash
vllm serve RedHatAI/Phi-4-mini-instruct-FP8-dynamic --max_model_len 131072
```

### lm-evaluation-harness
```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Phi-4-mini-instruct-FP8-dynamic,max_length=131072,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,timeout=600,tokenizer_backend=None" \
  --apply_chat_template \
  --num_fewshot 5 \
  --fewshot_as_multiturn \
  --output_path gsm8k_platinum_phi4_mini_instruct_fp8_dynamic \
  --gen_kwargs "do_sample=False,temperature=0.0,max_gen_toks=16000"
```

```bash
lm_eval --model local-chat-completions \
  --tasks mmlu_cot_llama \
  --model_args "model=RedHatAI/Phi-4-mini-instruct-FP8-dynamic,max_length=131072,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,timeout=600,tokenizer_backend=None" \
  --apply_chat_template \
  --output_path mmlu_cot_phi4_mini_instruct_fp8_dynamic \
  --gen_kwargs "do_sample=False,temperature=0.0,max_gen_toks=16000"
```

```bash
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=RedHatAI/Phi-4-mini-instruct-FP8-dynamic,max_length=131072,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,timeout=600,tokenizer_backend=None" \
  --apply_chat_template \
  --num_fewshot 5 \
  --fewshot_as_multiturn \
  --output_path mmlu_pro_phi4_mini_instruct_fp8_dynamic \
  --gen_kwargs "do_sample=False,temperature=0.0,max_gen_toks=16000"
```

```bash
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=RedHatAI/Phi-4-mini-instruct-FP8-dynamic,max_length=131072,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,timeout=600,tokenizer_backend=None" \
  --apply_chat_template \
  --output_path ifeval_phi4_mini_instruct_fp8_dynamic \
  --gen_kwargs "do_sample=False,temperature=0.0,max_gen_toks=16000"
```

### lighteval
litellm_config.yaml
```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/RedHatAI/Phi-4-mini-instruct-FP8-dynamic"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 600
  concurrent_requests: 128
  generation_parameters:
    temperature: 0.0
    max_new_tokens: 16000
```

```bash
lighteval endpoint litellm litellm_config.yaml \
    math_500|0 \
    --output-dir phi4_mini_instruct_fp8_dynamic \
    --save-details
```
</details>

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Phi-4-mini-instruct</strong>
   </td>
   <td><strong>Phi-4-mini-instruct-FP8-dynamic<br>(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
  <tr>
   <td>Math 500
   </td>
   <td>57.60
   </td>
   <td>58.20
   </td>
   <td>101.7%
   </td>
  </tr>
  <tr>
   <td>GSM8k-Platinum
   </td>
   <td>84.12
   </td>
   <td>84.70
   </td>
   <td>100.7%
   </td>
  </tr>
  <tr>
   <td>MMLU CoT
   </td>
   <td>67.01
   </td>
   <td>66.97
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td>MMLU-Pro
   </td>
   <td>46.75
   </td>
   <td>45.60
   </td>
   <td>97.5%
   </td>
  </tr>
</table>

