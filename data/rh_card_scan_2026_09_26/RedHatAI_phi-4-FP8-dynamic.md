---
language:
- en
base_model:
- microsoft/phi-4
pipeline_tag: text-generation
tags:
- phi
- phi3
- nlp
- math
- code
- chat
- conversational
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP8
- compressed-tensors
license: mit
license_name: mit
name: RedHatAI/phi-4-FP8-dynamic
description: This model was obtained by quantizing activation and weights of phi-4 to FP8 data type.
readme: https://huggingface.co/RedHatAI/phi-4-FP8-dynamic/main/README.md
tasks:
- text-to-text
provider: Microsoft
license_link: https://choosealicense.com/licenses/mit/
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  phi-4-FP8-dynamic
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
  3. Reasoning and logic.
- **Out-of-scope:** This model is not specifically designed or evaluated for all downstream purposes, thus:
  1. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios.
  2. Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case, including the model’s focus on English.
  3. Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.
- **Release Date:** 03/03/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **Model Developers:** RedHat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing activation and weights of [phi-4](https://huggingface.co/microsoft/phi-4) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic-ent/phi-4-FP8-dynamic"
number_gpus = 1

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Give me a short introduction to large language model."},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

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
--enforce-eager --model RedHatAI/phi-4-FP8-dynamic
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/phi-4-fp8-dynamic:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/phi-4-fp8-dynamic
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/phi-4-fp8-dynamic
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
    openshift.io/display-name: phi-4-FP8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: phi-4-FP8-dynamic         # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-phi-4-fp8-dynamic:1.5
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
    "model": "phi-4-FP8-dynamic",
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
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from llmcompressor.modifiers.quantization import QuantizationModifier
  from llmcompressor.transformers import oneshot
  
  # Load model
  model_stub = "microsoft/phi-4"
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

The model was evaluated on the OpenLLM leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic-ent/phi-4-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.6,max_model_len=4096,enable_chunk_prefill=True,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>phi-4</strong>
   </td>
   <td><strong>phi-4-FP8-dynamic<br>(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>80.30
   </td>
   <td>80.30
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>64.42
   </td>
   <td>64.25
   </td>
   <td>99.7%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>90.07
   </td>
   <td>90.67
   </td>
   <td>100.7%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>84.37
   </td>
   <td>84.19
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>80.58
   </td>
   <td>79.87
   </td>
   <td>99.1%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>59.37
   </td>
   <td>59.54
   </td>
   <td>100.3%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>76.52</strong>
   </td>
   <td><strong>76.47</strong>
   </td>
   <td><strong>99.9%</strong>
   </td>
  </tr>
</table>

