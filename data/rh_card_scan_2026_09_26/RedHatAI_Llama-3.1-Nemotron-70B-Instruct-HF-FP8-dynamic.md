---
language:
- en
base_model:
- nvidia/Llama-3.1-Nemotron-70B-Instruct-HF
pipeline_tag: text-generation
tags:
- llama
- fp8
- vllm
- conversational
- compressed-tensors
- text-generation-inference
license: llama3.1
license_name: llama3.1
name: RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic
description: This model is a quantized version of Llama-3.1-Nemotron-70B-Instruct. It was obtained by quantizing the weights and activations to FP8 data type.
readme: https://huggingface.co/RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic/main/README.md
tasks:
- text-to-text
provider: NVIDIA
license_link: https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---
<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Llama-3.1-Nemotron
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. Similarly to [Llama-3.1-Nemotron-70B-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF), this model is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 10/17/2024
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

This model is a quantized version of [Llama-3.1-Nemotron-70B-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model, including multiple-choice, math reasoning, and open-ended text generation.
Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic achieves 99.41% recovery for the Arena-Hard evaluation, 100% for OpenLLM v1 (using Meta's prompting when available), and ToDo for OpenLLM v2.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Llama-3.1-Nemotron-70B-Instruct](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Instruct-HF) to FP8 data type, ready for inference with vLLM built from source.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the FP8 representations of the quantized weights and activations. Activations are also quantized on a per-token dynamic basis.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic"
number_gpus = 2

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

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
--enforce-eager --model RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/llama-3-1-nemotron-70b-instruct-hf-fp8-dynamic:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/llama-3-1-nemotron-70b-instruct-hf-fp8-dynamic
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/llama-3-1-nemotron-70b-instruct-hf-fp8-dynamic
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
    openshift.io/display-name: Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic         # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-llama-3-1-nemotron-70b-instruct-hf-fp8-dynamic:1.5
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
    "model": "Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",
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

This model was created by applying [LLM-Compressor](https://github.com/vllm-project/llm-compressor), as presented in the code snipet below.

```python
import torch

from transformers import AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import (  # noqa
    calculate_offload_device_map,
    custom_offload_device_map,
)

recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["Linear"]
"""

model_stub = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
model_name = model_stub.split("/")[-1]

device_map = calculate_offload_device_map(
    model_stub, reserve_for_hessians=False, num_gpus=1, torch_dtype="auto"
)

model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype="auto", device_map=device_map
)

output_dir = f"./{model_name}-FP8-dynamic"

oneshot(
    model=model,
    recipe=recipe,
    output_dir=output_dir,
    save_compressed=True,
    tokenizer=AutoTokenizer.from_pretrained(model_stub),
)
```

## Evaluation

This model was evaluated on the well-known Arena-Hard, OpenLLM v1, and OpenLLM v2.
In all cases, model outputs were generated with the [vLLM](https://docs.vllm.ai/en/stable/) engine.

Arena-Hard evaluations were conducted using the [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto) repository.

OpenLLM v1 and v2 evaluations were conducted using Neural Magic's fork of [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness/tree/llama_3.1_instruct) (branch llama_3.1_instruct).
This version of the lm-evaluation-harness includes versions of MMLU, ARC-Challenge and GSM-8K that match the prompting style of [Meta-Llama-3.1-Instruct-evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-70B-Instruct-evals) and a few fixes to OpenLLM v2 tasks.

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>nvidia/Llama-3.1-Nemotron-70B-Instruct-HF</strong>
   </td>
   <td><strong>neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic </br>(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Arena Hard</strong>
   </td>
   <td><strong>85.0</strong>
   </td>
   <td><strong>84.5</strong>
   </td>
   <td><strong>99.41%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>OpenLLM Leaderboard v1</strong>
   </td>
   <td><strong>80.1</strong>
   </td>
   <td><strong>80.3</strong>
   </td>
   <td><strong>100.2%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>OpenLLM Leaderboard v2</strong>
   </td>
   <td><strong>40.2</strong>
   </td>
   <td><strong>39.8</strong>
   </td>
   <td><strong>99.0%</strong>
   </td>
  </tr>
</table>

<table>
  <tr>
   <td><strong>Benchmark (per-task breakdown)</strong>
   </td>
   <td><strong>nvidia/Llama-3.1-Nemotron-70B-Instruct-HF</strong>
   </td>
   <td><strong>neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td><strong>OpenLLM v1</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>83.51
   </td>
   <td>83.49
   </td>
   <td>99.97%
   </td>
  </tr>
  <tr>
   <td>MMLU-cot (0-shot)
   </td>
   <td>85.89
   </td>
   <td>86.18
   </td>
   <td>100.33%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (0-shot)
   </td>
   <td>93.09
   </td>
   <td>93.09
   </td>
   <td>100%
   </td>
  </tr>
  <tr>
   <td>GSM-8K-cot (8-shot, strict-match)
   </td>
   <td>70.13
   </td>
   <td>69.98
   </td>
   <td>99.78%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>87.39
   </td>
   <td>87.22
   </td>
   <td>99.80%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>84.93
   </td>
   <td>84.93
   </td>
   <td>100%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>55.97
   </td>
   <td>57.12
   </td>
   <td>102.05%
   </td>
  </tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>80.1</strong>
   </td>
   <td><strong>80.3</strong>
   </td>
   <td><strong>100.2%</strong>
   </td>
  <tr>
   <td><strong>OpenLLM v2</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>43.45
   </td>
   <td>42.99
   </td>
   <td>98.94%
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>73.32
   </td>
   <td>74.08
   </td>
   <td>101.02%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>47.12
   </td>
   <td>46.88
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5 (4-shot)
   </td>
   <td>23.85
   </td>
   <td>21.78
   </td>
   <td>91.32%
   </td>
  </tr>
  <tr>
   <td>MuSR (0-shot)
   </td>
   <td>13.5
   </td>
   <td>13.35
   </td>
   <td>98.88%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>40.2</strong>
   </td>
   <td><strong>39.8</strong>
   </td>
   <td><strong>99%</strong>
   </td>
  </tr>
</table>

### Reproduction

The results were obtained using the following commands:

#### MMLU
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2 \
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU-cot
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2 \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2 \
  --tasks arc_challenge_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2 \
  --tasks gsm8k_cot_llama_3.1_instruct \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --num_fewshot 8 \
  --batch_size auto
```

#### Hellaswag
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2 \
  --tasks hellaswag \
  --num_fewshot 10 \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2 \
  --tasks winogrande \
  --num_fewshot 5 \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2 \
  --tasks truthfulqa \
  --num_fewshot 0 \
  --batch_size auto
```

#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```
