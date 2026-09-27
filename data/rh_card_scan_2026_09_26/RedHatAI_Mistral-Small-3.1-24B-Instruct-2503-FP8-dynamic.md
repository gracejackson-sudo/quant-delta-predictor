---
language:
- en
- fr
- de
- es
- it
- pt
- hi
- id
- tl
- vi
- ar
- bg
- zh
- da
- el
- fa
- fi
- he
- ja
- ko
- ms
- nl
- no
- pl
- ro
- ru
- sr
- sv
- th
- tr
- uk
- ur
- zsm
- nld
base_model:
- mistralai/Mistral-Small-3.1-24B-Instruct-2503
pipeline_tag: image-text-to-text
tags:
- mistralai
- mistral
- mistral3
- mistral-small
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP8
- conversational
- compressed-tensors
- fast
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic
description: This model was obtained by quantizing activations and weights of Mistral-Small-3.1-24B-Instruct-2503 to FP8 data type.
readme: https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic/main/README.md
tasks:
- image-text-to-text
- text-to-text
provider: Mistral AI
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>


## Model Overview
- **Model Architecture:** Mistral3ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP8
  - **Weight quantization:** FP8
- **Intended Use Cases:** It is ideal for:
  - Fast-response conversational agents.
  - Low-latency function calling.
  - Subject matter experts via fine-tuning.
  - Local inference for hobbyists and organizations handling sensitive data.
  - Programming and math reasoning.
  - Long document understanding.
  - Visual understanding.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages not officially supported by the model.
- **Release Date:** 04/15/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **Model Developers:** RedHat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing activations and weights of [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.


## Deployment

1. Initialize vLLM server:
```
vllm serve RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16 --tensor_parallel_size 1 --tokenizer_mode mistral
```

2. Send requests to the server:

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16"


messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```

<details>
  <summary>Deploy on <strong>Red Hat AI Inference Server</strong></summary>
  
```bash
podman run --rm -it --device nvidia.com/gpu=all -p 8000:8000 \
 --ipc=host \
--env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
--env "HF_HUB_OFFLINE=0" -v ~/.cache/vllm:/home/vllm/.cache \
--name=vllm \
registry.access.redhat.com/rhaiis/rh-vllm-cuda \
vllm serve \
--tensor-parallel-size 1 \
--max-model-len 32768  \
--enforce-eager --model RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/mistral-small-3-1-24b-instruct-2503-fp8-dynamic:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/mistral-small-3-1-24b-instruct-2503-fp8-dynamic
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/mistral-small-3-1-24b-instruct-2503-fp8-dynamic
```
See [Red Hat Enterprise Linux AI documentation](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.4) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Openshift AI</strong></summary>
  
```python
# Setting up vllm server with ServingRuntime
# Save as: vllm-servingruntime.yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
 name: vllm-cuda-runtime # OPTIONAL CHANGE: set a unique name
 annotations:
   openshift.io/display-name: vLLM NVIDIA GPU ServingRuntime for KServe
   opendatahub.io/recommended-accelerators: '["nvidia.com/gpu"]'
 labels:
   opendatahub.io/dashboard: 'true'
spec:
 annotations:
   prometheus.io/port: '8080'
   prometheus.io/path: '/metrics'
 multiModel: false
 supportedModelFormats:
   - autoSelect: true
     name: vLLM
 containers:
   - name: kserve-container
     image: quay.io/modh/vllm:rhoai-2.20-cuda # CHANGE if needed. If AMD: quay.io/modh/vllm:rhoai-2.20-rocm
     command:
       - python
       - -m
       - vllm.entrypoints.openai.api_server
     args:
       - "--port=8080"
       - "--model=/mnt/models"
       - "--served-model-name={{.Name}}"
     env:
       - name: HF_HOME
         value: /tmp/hf_home
     ports:
       - containerPort: 8080
         protocol: TCP
```

```python
# Attach model to vllm server. This is an NVIDIA template
# Save as: inferenceservice.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    openshift.io/display-name: mistral-small-3-1-24b-instruct-2503-fp8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: mistral-small-3-1-24b-instruct-2503-fp8-dynamic          # specify model name. This value will be used to invoke the model in the payload
  labels:
    opendatahub.io/dashboard: 'true'
spec:
  predictor:
    maxReplicas: 1
    minReplicas: 1
    model:
      modelFormat:
        name: vLLM
      name: ''
      resources:
        limits:
          cpu: '2'			# this is model specific
          memory: 8Gi		# this is model specific
          nvidia.com/gpu: '1'	# this is accelerator specific
        requests:			# same comment for this block
          cpu: '1'
          memory: 4Gi
          nvidia.com/gpu: '1'
      runtime: vllm-cuda-runtime	# must match the ServingRuntime name above
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-mistral-small-3-1-24b-instruct-2503-fp8-dynamic:1.5
    tolerations:
    - effect: NoSchedule
      key: nvidia.com/gpu
      operator: Exists
```

```bash
# make sure first to be in the project where you want to deploy the model
# oc project <project-name>

# apply both resources to run model

# Apply the ServingRuntime
oc apply -f vllm-servingruntime.yaml

# Apply the InferenceService
oc apply -f qwen-inferenceservice.yaml
```

```python
# Replace <inference-service-name> and <cluster-ingress-domain> below:
# - Run `oc get inferenceservice` to find your URL if unsure.

# Call the server using curl:
curl https://<inference-service-name>-predictor-default.<domain>/v1/chat/completions
        -H "Content-Type: application/json" \
        -d '{
    "model": "mistral-small-3-1-24b-instruct-2503-fp8-dynamic",
    "stream": true,
    "stream_options": {
        "include_usage": true
    },
    "max_tokens": 1,
    "messages": [
        {
            "role": "user",
            "content": "How can a bee fly when its wings are so small?"
        }
    ]
}'

```

See [Red Hat Openshift AI documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/2025) for more details.
</details>


## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


  ```python
  from llmcompressor.modifiers.quantization import QuantizationModifier
  from llmcompressor.transformers import oneshot
  from transformers import AutoModelForImageTextToText, AutoProcessor
  
  # Load model
  model_stub = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  model_name = model_stub.split("/")[-1]

  model = AutoModelForImageTextToText.from_pretrained(model_stub)

  processor = AutoProcessor.from_pretrained(model_stub)

  # Configure the quantization algorithm and scheme
  recipe = QuantizationModifier(
      ignore=["language_model.lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
      targets="Linear",
      scheme="FP8_dynamic",
  )

  # Apply quantization
  oneshot(
      model=model,
      recipe=recipe,
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-FP8-dynamic"
  model.save_pretrained(save_path)
  processor.save_pretrained(save_path)
  print(f"Model and tokenizer saved to: {save_path}")
  ```
</details>
 


## Evaluation

The model was evaluated on the OpenLLM leaderboard tasks (version 1), MMLU-pro, GPQA, HumanEval and MBPP.
Non-coding tasks were evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), whereas coding tasks were evaluated with a fork of [evalplus](https://github.com/neuralmagic/evalplus).
[vLLM](https://docs.vllm.ai/en/stable/) is used as the engine in all cases.

<details>
  <summary>Evaluation details</summary>

  **MMLU**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks mmlu \
    --num_fewshot 5 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **ARC Challenge**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **GSM8k**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.9,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks gsm8k \
    --num_fewshot 8 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **Hellaswag**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks hellaswag \
    --num_fewshot 10 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **Winogrande**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks winogrande \
    --num_fewshot 5 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **TruthfulQA**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks truthfulqa \
    --num_fewshot 0 \
    --apply_chat_template\
    --batch_size auto
  ```

  **MMLU-pro**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks mmlu_pro \
    --num_fewshot 5 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

**Coding**

The commands below can be used for mbpp by simply replacing the dataset name.

*Generation*
```
python3 codegen/generate.py \
  --model RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval

```

*Sanitization*
```
python3 evalplus/sanitize.py \
  humaneval/RedHatAI--Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic_vllm_temp_0.2
```

*Evaluation*
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/RedHatAI--Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic_vllm_temp_0.2-sanitized
```
</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>Mistral-Small-3.1-24B-Instruct-2503
   </th>
   <th>Mistral-Small-3.1-24B-Instruct-2503-FP8-dynamic<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>80.67
   </td>
   <td>80.71
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>72.78
   </td>
   <td>72.87
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>58.68
   </td>
   <td>49.96
   </td>
   <td>85.1%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>83.70
   </td>
   <td>83.67
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>83.74
   </td>
   <td>82.56
   </td>
   <td>98.6%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>70.62
   </td>
   <td>70.88
   </td>
   <td>100.4%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>75.03</strong>
   </td>
   <td><strong>73.49</strong>
   </td>
   <td><strong>97.9%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" ><strong></strong>
   </td>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>67.25
   </td>
   <td>66.86
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>GPQA CoT main (5-shot)
   </td>
   <td>42.63
   </td>
   <td>41.07
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>GPQA CoT diamond (5-shot)
   </td>
   <td>45.96
   </td>
   <td>45.45
   </td>
   <td>98.9%
   </td>
  </tr>
  <tr>
   <td rowspan="4" ><strong>Coding</strong>
   </td>
   <td>HumanEval pass@1
   </td>
   <td>84.70
   </td>
   <td>84.70
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>79.50
   </td>
   <td>79.30
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>MBPP pass@1
   </td>
   <td>71.10
   </td>
   <td>70.00
   </td>
   <td>98.5%
   </td>
  </tr>
  <tr>
   <td>MBPP+ pass@1
   </td>
   <td>60.60
   </td>
   <td>59.50
   </td>
   <td>98.2%
   </td>
  </tr>
</table>

