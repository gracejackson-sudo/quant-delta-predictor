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
- no
- el
- bg
- uk
- ur
- sr
- ms
- zsm
- nld
base_model:
- Qwen/Qwen2.5-7B-Instruct
pipeline_tag: text-generation
tags:
- qwen
- qwen2_5
- qwen2_5_instruct
- w8a8
- int8
- vllm
- conversational
- text-generation-inference
- compressed-tensors
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8
description: This model was obtained by quantizing the weights and activations of Qwen2.5-7B-Instruct to INT8 data type.
readme: https://huggingface.co/RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8/main/README.md
tasks:
- text-to-text
provider: Alibaba Cloud
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Qwen2.5-7B-Instruct-quantized.w8a8
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Qwen2
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 10/09/2024
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **License(s):** [apache-2.0](https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

### Model Optimizations

This model was obtained by quantizing activations and weights of [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
A combination of the [SmoothQuant](https://arxiv.org/abs/2211.10438) and [GPTQ](https://arxiv.org/abs/2210.17323) algorithms is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8"
number_gpus = 1
max_model_len = 8192

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Give me a short introduction to large language model."},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

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
--enforce-eager --model RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a8
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/qwen2-5-7b-instruct-quantized-w8a8:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/qwen2-5-7b-instruct-quantized-w8a8
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/qwen2-5-7b-instruct-quantized-w8a8
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
    openshift.io/display-name: Qwen2.5-7B-Instruct-quantized.w8a8 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: Qwen2.5-7B-Instruct-quantized.w8a8         # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-qwen2-5-7b-instruct-quantized-w8a8:1.5
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
    "model": "Qwen2.5-7B-Instruct-quantized.w8a8",
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
  from llmcompressor.modifiers.quantization import GPTQModifier
  from llmcompressor.modifiers.smoothquant import SmoothQuantModifier  
  from llmcompressor.transformers import oneshot
  from datasets import load_dataset
  
  # Load model
  model_stub = "Qwen/Qwen2.5-7B-Instruct"
  model_name = model_stub.split("/")[-1]
  
  num_samples = 512
  max_seq_len = 8192
  
  tokenizer = AutoTokenizer.from_pretrained(model_stub)
  
  model = AutoModelForCausalLM.from_pretrained(
      model_stub,
      device_map="auto",
      torch_dtype="auto",
  )
  
  def preprocess_fn(example):
      return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}
  
  ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
  ds = ds.map(preprocess_fn)
  
  # Configure the quantization algorithm and scheme
  recipe = [
      SmoothQuantModifier(
        smoothing_strength=0.8,
        mappings=[
            [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
            [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
            [["re:.*down_proj"], "re:.*up_proj"],
        ],
      ),
      GPTQModifier(
          ignore=["lm_head"],
          sequential_targets=["Qwen2DecoderLayer"],
          dampening_frac=0.01,
          targets="Linear",
          scheme="W8A8",
      ),
  ]
  
  # Apply quantization
  oneshot(
      model=model,
      dataset=ds, 
      recipe=recipe,
      max_seq_length=max_seq_len,
      num_calibration_samples=num_samples,
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-quantized.w8a8"
  model.save_pretrained(save_path)
  tokenizer.save_pretrained(save_path)
  print(f"Model and tokenizer saved to: {save_path}")
  ```
</details>

## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/387Bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 387Bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Qwen2.5-7B-Instruct-quantized.w8a8",dtype=auto,gpu_memory_utilization=0.5,max_model_len=4096,add_bos_token=True,enable_chunk_prefill=True,tensor_parallel_size=1 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <th>Benchmark
   </th>
   <th>Qwen2.5-7B-Instruct
   </th>
   <th>Qwen2.5-7B-Instruct-quantized.w8a8<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>74.24
   </td>
   <td>73.87
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>63.40
   </td>
   <td>63.23
   </td>
   <td>99.7%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>80.36
   </td>
   <td>80.74
   </td>
   <td>100.5%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>81.52
   </td>
   <td>81.06
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>74.66
   </td>
   <td>74.82
   </td>
   <td>100.2%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>64.76
   </td>
   <td>64.58
   </td>
   <td>99.7%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>73.16</strong>
   </td>
   <td><strong>73.05</strong>
   </td>
   <td><strong>99.4%</strong>
   </td>
  </tr>
</table>

