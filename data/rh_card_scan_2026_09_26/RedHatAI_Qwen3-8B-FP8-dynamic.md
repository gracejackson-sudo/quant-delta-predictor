---
language:
- zh
- en
- fr
- es
- pt
- de
- it
- ru
- ja
- ko
- vi
- th
- ar
- id
- tr
- fa
- nl
- pl
- cs
- he
- sv
- fi
- da
- 'no'
- el
- bg
- uk
- ur
- sr
- ms
- zsm
- nld
base_model:
- Qwen/Qwen3-8B
pipeline_tag: text-generation
tags:
- qwen
- qwen3
- fp8
- vllm
- conversational
- text-generation-inference
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Qwen3-8B-FP8-dynamic
description: >-
  This model was obtained by quantizing activations and weights of Qwen3-8B to
  FP8 data type.
readme: https://huggingface.co/RedHatAI/Qwen3-8B-FP8-dynamic/main/README.md
tool_calling_supported: true
required_cli_args: []
chat_template_file_name: None
chat_template_path: None
tool_call_parser: hermes
tasks:
- text-to-text
- tool-calling
provider: Alibaba Cloud
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 2.24
  - RHAIIS 3.2.1
  - vLLM 0.10.0
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Qwen3-8B-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Qwen3ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP8
  - **Weight quantization:** FP8
- **Intended Use Cases:**
  - Reasoning.
  - Function calling.
  - Subject matter experts via fine-tuning.
  - Multilingual instruction following.
  - Translation.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 05/02/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.24, RHAIIS 3.2.1
- **Model Developers:** RedHat (Neural Magic)
- **ModelCar Storage URI:** oci://registry.redhat.io/rhelai1/modelcar-qwen3-8b-fp8-dynamic:1.5
- **Validated on RHOAI 2.24:** quay.io/modh/vllm:rhoai-2.24-cuda
- **Validated on RHAIIS 3.2.1:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1
- **Validated on vLLM:** 0.10.0

### Model Optimizations

This model was obtained by quantizing activations and weights of [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) to FP8 data type.
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

model_id = "RedHatAI/Qwen3-8B-FP8-dynamic"
number_gpus = 1
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=256)

messages = [
    {"role": "user", "content": prompt}
]

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

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
--enforce-eager --model RedHatAI/Qwen3-8B-FP8-dynamic
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
    openshift.io/display-name: Qwen3-8B-FP8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: Qwen3-8B-FP8-dynamic          # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-qwen3-8b-fp8-dynamic:1.5
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
    "model": "Qwen3-8B-FP8-dynamic",
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
  from transformers import AutoModelForCausalLM, AutoTokenizer
  
  # Load model
  model_stub = "Qwen/Qwen3-8B"
  model_name = model_stub.split("/")[-1]

  model = AutoModelForCausalLM.from_pretrained(model_stub)

  tokenizer = AutoTokenizer.from_pretrained(model_stub)

  # Configure the quantization algorithm and scheme
  recipe = QuantizationModifier(
      ignore=["lm_head"],
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

The model was evaluated on the OpenLLM leaderboard tasks (versions 1 and 2), using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and on reasoning tasks using [lighteval](https://github.com/neuralmagic/lighteval/tree/reasoning).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>

  **lm-evaluation-harness**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-8B-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks openllm \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-8B-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks mgsm \
    --apply_chat_template\
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-8B-FP8-dynamic",dtype=auto,gpu_memory_utilization=0.5,max_model_len=16384,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks leaderboard \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-8B-FP8-dynamic
    dtype: auto
    gpu_memory_utilization: 0.9
    max_model_length: 40960
    generation_parameters:
      temperature: 0.6
      top_k: 20
      min_p: 0.0
      top_p: 0.95
      max_new_tokens: 32768
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|aime24|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|aime25|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|math_500|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|gpqa:diamond|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks extended|lcb:codegeneration \
    --use_chat_template = true
  ```

</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>Qwen3-8B
   </th>
   <th>Qwen3-8B-FP8-dynamic<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>71.95
   </td>
   <td>72.30
   </td>
   <td>100.5%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>61.69
   </td>
   <td>61.60
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>75.97
   </td>
   <td>80.52
   </td>
   <td>106.0%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>56.52
   </td>
   <td>55.95
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>65.98
   </td>
   <td>66.22
   </td>
   <td>100.4%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>53.17
   </td>
   <td>52.39
   </td>
   <td>98.5%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>64.21</strong>
   </td>
   <td><strong>64.83</strong>
   </td>
   <td><strong>101.0%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v2</strong>
   </td>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>34.57
   </td>
   <td>37.82
   </td>
   <td>109.4%
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>84.77
   </td>
   <td>84.56
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>25.47
   </td>
   <td>27.20
   </td>
   <td>106.8%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5 (4-shot)
   </td>
   <td>51.05
   </td>
   <td>51.90
   </td>
   <td>101.7%
   </td>
  </tr>
  <tr>
   <td>GPQA (0-shot)
   </td>
   <td>0.00
   </td>
   <td>0.00
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td>MuSR (0-shot)
   </td>
   <td>10.02
   </td>
   <td>10.65
   </td>
   <td>---
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>34.31</strong>
   </td>
   <td><strong>35.35</strong>
   </td>
   <td><strong>103.0%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Multilingual</strong>
   </td>
   <td>MGSM (0-shot)
   </td>
   <td>25.97
   </td>
   <td>25.80
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td rowspan="6" ><strong>Reasoning<br>(generation)</strong>
   </td>
   <td>AIME 2024
   </td>
   <td>74.58
   </td>
   <td>76.35
   </td>
   <td>102.4%
   </td>
  </tr>
  <tr>
   <td>AIME 2025
   </td>
   <td>65.21
   </td>
   <td>63.75
   </td>
   <td>97.8%
   </td>
  </tr>
  <tr>
   <td>GPQA diamond
   </td>
   <td>58.59
   </td>
   <td>61.11
   </td>
   <td>104.3%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5
   </td>
   <td>97.60
   </td>
   <td>96.60
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td>LiveCodeBench
   </td>
   <td>56.27
   </td>
   <td>56.60
   </td>
   <td>100.6%
   </td>
  </tr>
</table>