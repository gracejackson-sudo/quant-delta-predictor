---
license: mit
license_name: mit
name: RedHatAI/Phi-4-reasoning-FP8-dynamic
description: This model is designed to accelerate research on language models, for use as a building block for generative AI powered features.
readme: https://huggingface.co/RedHatAI/Phi-4-reasoning-FP8-dynamic/main/README.md
license_link: https://huggingface.co/microsoft/Phi-4-reasoning/resolve/main/LICENSE
provider: Microsoft
validated_on:
  - RHOAI 3.3
  - RHAIIS 3.3
  - vLLM 0.13.0
language:
- en
base_model:
- microsoft/Phi-4-reasoning
pipeline_tag: text-generation
tags:
- phi
- nlp
- math
- code
- chat
- conversational
- reasoning
- red hat
- FP8
- compressed-tensors
- llm-compressor
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Phi-4-reasoning-FP8-dynamic
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
- **Intended Use Cases:** This model is designed to accelerate research on language models, for use as a building block for generative AI powered features. It provides uses for general purpose AI systems and applications (primarily in English) which require:
1. Memory/compute constrained environments.
2. Latency bound scenarios.
3. Math reasoning and logic.
- **Release Date:** 01/26/2026
- **Version:** 1.0
- **Model Developers:** Red Hat
- **ModelCar Storage URI**: oci://registry.redhat.io/rhai/modelcar-phi-4-reasoning-fp8-dynamic:3.0
- **Validated on vLLM:** 0.13.0
- **Validated on RHAIIS:** 3.3
- **Validated on RHOAI:** 3.3


### Model Optimizations

This model was obtained by quantizing activation and weights of [Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```bash
vllm serve RedHatAI/Phi-4-reasoning-FP8-dynamic --reasoning-parser deepseek_r1
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
    model="RedHatAI/Phi-4-reasoning-FP8-dynamic",
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
  from llmcompressor.transformers import oneshot
  
  # Load model
  model_stub = "microsoft/Phi-4-reasoning"
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

The model was evaluated on the AIME25, GPQA Diamond and Mathh 500 benchmarks using [lighteval](https://github.com/huggingface/lighteval), and on MMLU-Pro using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
In both cases [vLLM](https://vllm.ai) is used as the backend

<details>
  <summary>Evaluation commands</summary>

### Start vLLM server
```bash
vllm serve RedHatAI/Phi-4-reasoning-FP8-dynamic --reasoning-parser deepseek_r1
```

### lm-evaluation-harness
```bash
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=RedHatAI/Phi-4-reasoning-FP8-dynamic,max_length=32000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,timeout=2400,tokenizer_backend=None" \
  --apply_chat_template \
  --num_fewshot 0 \
  --output_path mmlu_pro_phi4_reasoning_fp8_dynamic \
  --gen_kwargs "do_sample=True,temperature=0.8,top_k=50,top_p=0.95,max_gen_toks=24000"
```

### lighteval
litellm_config.yaml
```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/RedHatAI/Phi-4-reasoning-FP8-dynamic"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 1200
  concurrent_requests: 64
  generation_parameters:
    temperature: 0.8
    top_k: 50
    top_p: 0.95
    max_new_tokens: 24000
```

```bash
lighteval endpoint litellm litellm_config.yaml \
    gpqa:diamond|0,math_500|0,aime25|0 \
    --output-dir phi4_reasoning_fp8_dynamic \
    --save-details
```
</details>

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Phi-4-reasoning</strong>
   </td>
   <td><strong>Phi-4-reasoning-FP8-dynamic<br>(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>AIME25
   </td>
   <td>61.25
   </td>
   <td>64.58
   </td>
   <td>105.4%
   </td>
  </tr>
  <tr>
   <td>GPQA Diamond
   </td>
   <td>64.65
   </td>
   <td>66.50
   </td>
   <td>102.9%
   </td>
  </tr>
  <tr>
   <td>Math 500
   </td>
   <td>90.01
   </td>
   <td>88.60
   </td>
   <td>98.4%
   </td>
  </tr>
  <tr>
   <td>MMLU-Pro
   </td>
   <td>76.49
   </td>
   <td>76.85
   </td>
   <td>100.5%
   </td>
  </tr>
</table>

