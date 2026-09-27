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
name: RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16
description: This model was obtained by quantizing weights of NVIDIA-Nemotron-Nano-9B-v2 to INT4 data type.
readme: https://huggingface.co/RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16/main/README.md
tags:
- nvidia
- pytorch
- int4
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

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16
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
  - **Weight quantization:** INT4
- **Release Date:** 10/22/2025
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)
- **ModelCar Storage URI:** oci://registry.redhat.io/rhelai1/modelcar-nvidia-nemotron-nano-9b-v2-quantized.w4a16:1.5
- **Validated on RHOAI 2.25:** quay.io/modh/vllm:rhoai-2.25-cuda
- **Validated on RHAIIS 3.2.2:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.2
- **Validated on vLLM:** 0.10.1.1

### Model Optimizations

This model was obtained by quantizing the weights of [NVIDIA-Nemotron-Nano-9B-v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-group scheme, with group size 64.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16"
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
--enforce-eager --model RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16
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
    openshift.io/display-name: NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16        # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-nvidia-nemotron-nano-9b-v2-quantized.w4a16:1.5
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
    "model": "NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16",
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
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from transformers import AutoModelForCausalLM, AutoTokenizer
  
# Load model
model_stub = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
model_name = model_stub.split("/")[-1]

num_samples = 1024
max_seq_len = 8192

model = AutoModelForCausalLM.from_pretrained(model_stub)

tokenizer = AutoTokenizer.from_pretrained(model_stub)

def preprocess_fn(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}
  
ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.map(preprocess_fn)

# Configure the quantization algorithm and scheme
quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=True,
        group_size=64,
        strategy=QuantizationStrategy.GROUP,
        observer="mse",
        actorder="weight"
    ),
    input_activations=None,
    output_activations=None,
)

recipe = [
    GPTQModifier(
        ignore=["lm_head", "NemotronHMamba2Mixer"],
        dampening_frac=0.07,
        config_groups={"group_0": quant_scheme},
    )
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
save_path = model_name + "-quantized.w4a16"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")
  ```
</details>
 


## Evaluation

The model was evaluated on the set of popular reasoning tasks AIME25, Math-500, and GPQA-Diamond, using [lighteval](https://github.com/huggingface/lighteval) `v0.11.1.dev0`.
[vLLM](https://docs.vllm.ai/en/stable/) `v0.11.1rc2.dev191+g80e945298.precompiled` was used as the inference engine for all evaluations.

<details>
  <summary>Evaluation details</summary>

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml
  model_parameters:
    model_name: "hosted_vllm/RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16"
    base_url: "http://0.0.0.0:8000/v1"
    generation_parameters:
      temperature: 0.6
      min_p: 0.0
      max_new_tokens: 65536
      top_p: 0.95
      seed: 0
  ```

  ```
  lighteval endpoint litellm lighteval_model_arguments.yaml \
      "lighteval|aime25|0,lighteval|math_500|0,lighteval|gpqa:diamond|0" \
      --output-dir $OUTPUT_DIR \
      --save-details
  ```

  ```
  vllm serve RedHatAI/NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16 \
    --trust-remote-code \
    --mamba_ssm_cache_dtype float32 \
    -tp 1 \
    --port 8000 \
    --gpu-memory-utilization 0.9
  ```

</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>NVIDIA-Nemotron-Nano-9B-v2
   </th>
   <th>NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="6" ><strong>Reasoning<br>(generation)</strong>
   </td>
  </tr>
  <tr>
   <td>AIME 2025
   </td>
   <td>61.33
   </td>
   <td>58.00
   </td>
   <td>94.6%
   </td>
  </tr>
  <tr>
   <td>GPQA diamond
   </td>
   <td>56.26
   </td>
   <td>56.16
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5
   </td>
   <td>96.08
   </td>
   <td>96.16
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>Average Score
   </td>
   <td>71.22
   </td>
   <td>70.11
   </td>
   <td>98.44%
   </td>
  </tr>
</table>