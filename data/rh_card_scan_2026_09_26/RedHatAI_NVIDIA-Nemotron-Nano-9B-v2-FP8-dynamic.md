---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
pipeline_tag: text-generation
language:
- en
- es
- fr
- de
- it
- ja
library_name: transformers
name: RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic
description: This model was obtained by quantizing weights of NVIDIA-Nemotron-Nano-9B-v2 to FP8-Dynamic data type.
readme: https://huggingface.co/RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic/main/README.md
tags:
- nvidia
- pytorch
- fp8
- quantized
- llm-compressor
- compressed-tensors
- red hat
track_downloads: true
base_model:
- nvidia/NVIDIA-Nemotron-Nano-9B
validated_on:
  - RHOAI 2.25
  - RHAIIS 3.2.2
  - vLLM 0.10.1.1
provider: NVIDIA
tasks:
- text-to-text
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** NemotronHForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 9/30/2025
- **Version:** 1.0
- **Model Developers:** Red Hat
- **ModelCar Storage URI:** oci://registry.redhat.io/rhelai1/modelcar-nvidia-nemotron-nano-9b-v2-fp8-dynamic:1.5
- **Validated on RHOAI 2.25:** quay.io/modh/vllm:rhoai-2.25-cuda
- **Validated on RHAIIS 3.2.2:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.2
- **Validated on vLLM:** 0.10.1.1

Quantized version of [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [nvidia/NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic
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

model = "RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic"

messages = [
    {"role": "user", "content": "Give me a short introduction to large language model."},
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
--tensor-parallel-size 8 \
--max-model-len 32768  \
--enforce-eager --model RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic
```
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
     image: quay.io/modh/vllm:rhoai-2.25-cuda # CHANGE if needed. If AMD: quay.io/modh/vllm:rhoai-2.25-rocm
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
    openshift.io/display-name: NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic         # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-nvidia-nemotron-nano-9b-v2-fp8-dynamic:1.5
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

```

```python
# Replace <inference-service-name> and <cluster-ingress-domain> below:
# - Run `oc get inferenceservice` to find your URL if unsure.

# Call the server using curl:
curl https://<inference-service-name>-predictor-default.<domain>/v1/chat/completions
        -H "Content-Type: application/json" \
        -d '{
    "model": "NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic",
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

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 

<details>
  <summary>Model Creation Code</summary>

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_stub = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
model_name = model_stub.split("/")[-1]

model = AutoModelForCausalLM.from_pretrained(model_stub, dtype="auto")

tokenizer = AutoTokenizer.from_pretrained(model_stub)

# Configure the quantization algorithm and scheme
recipe = QuantizationModifier(
    ignore=["lm_head", "NemotronHMamba2Mixer"],
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
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")
```

</details>

## Evaluation


The model was evaluated on the OpenLLMv1 leaderboard task, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and on reasoning tasks using [lighteval](https://github.com/huggingface/lighteval).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>
  
  **lm-evaluation-harness**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.6,enable_chunked_prefill=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --output_path output_dir \
    --show_config
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic
    dtype: auto
    system_prompt: /think
    gpu_memory_utilization: 0.9
    generation_parameters:
      temperature: 0.6
      min_p: 0.0
      top_p: 0.95
      max_new_tokens: 32768
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|aime25|0 \
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|math_500|0 \
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|gpqa:diamond|0 \
    --use_chat_template = true
  ```

</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>nvidia/NVIDIA-Nemotron-Nano-9B-v2</th>
      <th>RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-FP8-dynamic</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>64.16</td>
      <td>63.23</td>
      <td>98.5</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>85.90</td>
      <td>86.50</td>
      <td>100.7</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>79.57</td>
      <td>79.75</td>
      <td>100.2</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>74.66</td>
      <td>74.51</td>
      <td>99.8</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>56.90</td>
      <td>55.90</td>
      <td>98.2</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>75.61</td>
      <td>75.61</td>
      <td>100.0</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>72.80</b></td>
      <td><b>72.58</b></td>
      <td><b>99.7</b></td>
    </tr>
  </tbody>
  <tbody>
    <!-- Reasoning -->
    <tr>
      <td rowspan="6" ><strong>Reasoning<br>(generation)</strong>
      </td>
      <td>AIME 2025*
      </td>
      <td>56.67
      </td>
      <td>53.33
      </td>
      <td>94.1
      </td>
  </tr>
  <tr>
    <td>GPQA diamond*
    </td>
    <td>55.05
    </td>
    <td>56.06
    </td>
    <td>101.8
    </td>
  </tr>
  <tr>
    <td>Math-500*
    </td>
    <td>95.90
    </td>
    <td>95.47
    </td>
    <td>99.6
    </td>
  </tr>
  </tbody>
</table>
* Average over 8 executions