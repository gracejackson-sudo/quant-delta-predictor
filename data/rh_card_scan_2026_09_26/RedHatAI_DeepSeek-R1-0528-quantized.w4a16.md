---
language:
- en
base_model:
- deepseek-ai/DeepSeek-R1-0528
pipeline_tag: text-generation
tags:
- deepseek_v3
- deepseek
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
- GPTQ
- conversational
- compressed-tensors
license: mit
license_name: mit
name: RedHatAI/DeepSeek-R1-0528-quantized.w4a16
description: This model was obtained by quantizing weights of DeepSeek-R1-0528 to INT4 data type.
readme: https://huggingface.co/RedHatAI/DeepSeek-R1-0528-quantized.w4a16/main/README.md
tasks:
- text-to-text
provider: DeepSeek
license_link: https://choosealicense.com/licenses/mit/
validated_on:
  - RHOAI 2.24
  - RHAIIS 3.2.1
  - vLLM v0.10.0
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  DeepSeek-R1-0528-quantized.w4a16
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** DeepseekV3ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** None
  - **Weight quantization:** INT4
- **Release Date:** 05/30/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.24, RHAIIS 3.2.1
- **Model Developers:** Red Hat (Neural Magic)
- **ModelCar Storage URI:** oci://registry.redhat.io/rhelai1/modelcar-deepseek-r1-0528-quantized-w4a16:1.5
- **Validated on RHOAI 2.24:** quay.io/modh/vllm:rhoai-2.24-cuda
- **Validated on RHAIIS 3.2.1:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1
- **Validated on vLLM:** 0.10.0

### Model Optimizations

This model was obtained by quantizing weights of [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) to INT4 data type.
This optimization reduces the number of bits used to represent weights from 8 to 4, reducing GPU memory requirements (by approximately 50%).
Weight quantization also reduces disk size requirements by approximately 50%.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
model_id = "RedHatAI/DeepSeek-R1-0528-quantized.w4a16"
number_gpus = 8
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=256)
tokenizer = AutoTokenizer.from_pretrained(model_id)
prompt = "Give me a short introduction to large language model."
llm = LLM(model=model_id, tensor_parallel_size=number_gpus)
outputs = llm.generate(prompt, sampling_params)
generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

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
--enforce-eager --model RedHatAI/DeepSeek-R1-0528-quantized.w4a16
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
     image: quay.io/modh/vllm:rhoai-2.24-cuda # CHANGE if needed. If AMD: quay.io/modh/vllm:rhoai-2.24-rocm
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
    openshift.io/display-name: DeepSeek-R1-0528-quantized.w4a16 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: DeepSeek-R1-0528-quantized.w4a16          # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-deepseek-r1-0528-quantized-w4a16:1.5
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
    "model": "DeepSeek-R1-0528-quantized.w4a16",
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

We created this model using **MoE-Quant**, a library developed jointly with **ISTA** and tailored for the quantization of very large Mixture-of-Experts (MoE) models.  

For more details, please refer to the [MoE-Quant repository](https://github.com/IST-DASLab/MoE-Quant).


## Evaluation

The model was evaluated on popular reasoning tasks (AIME 2024, MATH-500, GPQA-Diamond) via [LightEval](https://github.com/huggingface/open-r1).
For reasoning evaluations, we estimate pass@1 based on 10 runs with different seeds, `temperature=0.6`, `top_p=0.95` and `max_new_tokens=65536`.


### Accuracy

|                             | Recovery (%) | deepseek/DeepSeek-R1-0528 | RedHatAI/DeepSeek-R1-0528-quantized.w4a16<br>(this model) |
| --------------------------- | :----------: | :------------------: | :--------------------------------------------------: |
| AIME 2024<br>pass@1         | 98.50         | 88.66                | 87.33                                                |
| MATH-500<br>pass@1          | 99.88        | 97.52                | 97.40                                                |
| GPQA Diamond<br>pass@1      | 101.21        | 79.65                | 80.61                                                |
| **Reasoning<br>Average Score**  | **99.82**        | **88.61**                | **88.45**                                                |