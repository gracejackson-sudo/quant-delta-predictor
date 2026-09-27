---
language:
- en
- fr
- de
- es
- it
- pt
- zh
- ja
- ru
- ko
base_model:
- mistralai/Mistral-Small-24B-Instruct-2501
pipeline_tag: text-generation
tags:
- mistral
- mistral-small
- quantized
- W8A8
- vllm
- conversational
- text-generation-inference
- compressed-tensors
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w8a8
description: This model was obtained by quantizing the weights and activations of Mistral-Small-24B-Instruct-2501 to INT8 data type.
readme: https://huggingface.co/RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w8a8/main/README.md
tasks:
- text-to-text
provider: Red Hat
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---
 
<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Mistral-Small-24B-Instruct-2501-quantized.w8a8
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
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** It is ideal for:
  - Fast-response conversational agents.
  - Low-latency function calling.
  - Subject matter experts via fine-tuning.
  - Local inference for hobbyists and organizations handling sensitive data.
  - Programming and math reasoning.
  - Long document understanding.
  - Visual understanding.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages not officially supported by the model.
- **Release Date:** 03/03/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **Model Developers:** Red Hat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing activations and weights of [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
A combination of the [SmoothQuant](https://arxiv.org/abs/2211.10438) and [GPTQ](https://arxiv.org/abs/2210.17323) algorithms is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

1. Initialize vLLM server:
```
vllm serve RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w8a8 --tensor_parallel_size 1 --tokenizer_mode mistral
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

model = "RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w8a8"


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
--tensor-parallel-size 8 \
--max-model-len 32768  \
--enforce-eager --model RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w8a8
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/mistral-small-24b-instruct-2501-quantized-w8a8:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/mistral-small-24b-instruct-2501-quantized-w8a8

# Chat with model
ilab model chat --model ~/.cache/instructlab/models/mistral-small-24b-instruct-2501-quantized-w8a8
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
    openshift.io/display-name: mistral-small-24b-instruct-2501-quantized-w8a8 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: mistral-small-24b-instruct-2501-quantized-w8a8         # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-mistral-small-24b-instruct-2501-quantized-w8a8:1.5
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
    "model": "mistral-small-24b-instruct-2501-quantized-w8a8",
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
  from transformers import AutoTokenizer, AutoModelForCausalLM
  from llmcompressor.modifiers.quantization import GPTQModifier
  from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
  from llmcompressor.transformers import oneshot
  from datasets import load_dataset
  
  # Load model
  model_stub = "mistralai/Mistral-Small-24B-Instruct-2501"
  model_name = model_stub.split("/")[-1]
  
  num_samples = 1024
  max_seq_len = 8192
  
  tokenizer = AutoTokenizer.from_pretrained(model_stub)
  
  model = AutoModelForCausalLM.from_pretrained(
      model_stub,
      device_map="auto",
      torch_dtype="auto",
  )

  # Data processing
  def preprocess_text(example):
      text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
      return tokenizer(text, padding=False, max_length=max_seq_len, truncation=True)
  
  ds = load_dataset("neuralmagic/calibration", name="LLM", split="train").select(range(num_samples))
  ds = ds.map(preprocess_text, remove_columns=ds.column_names)

  # Configure the quantization algorithm and scheme
  recipe = [
      SmoothQuantModifier(
        smoothing_strength=0.9,
        mappings=[
            [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
            [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
            [["re:.*down_proj"], "re:.*up_proj"],
        ],
      ),
      GPTQModifier(
          ignore=["lm_head"],
          sequential_targets=["MistralDecoderLayer"],
          dampening_frac=0.1,
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
      num_calibration_samples=num_samples
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-quantized.w8a8"
  model.save_pretrained(save_path)
  processor.save_pretrained(save_path)
  print(f"Model and tokenizer saved to: {save_path}")
  ```
</details>
 


## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) and [V2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/), using the following commands:

OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Mistral-Small-24B-Instruct-2501-FP8-Dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks openllm \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```

OpenLLM Leaderboard V2:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Mistral-Small-24B-Instruct-2501-FP8-Dynamic",dtype=auto,add_bos_token=False,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config

```

### Accuracy

#### OpenLLM Leaderboard V1 evaluation scores

| Metric                                   | mistralai/Mistral-Small-24B-Instruct-2501             | nm-testing/Mistral-Small-24B-Instruct-2501-quantized.w8a8 |
|-----------------------------------------|:---------------------------------:|:-------------------------------------------:|
| ARC-Challenge (Acc-Norm, 25-shot)       |  72.18                            |   68.86                                  |
| GSM8K (Strict-Match, 5-shot)            |  90.14                            |   90.00                                      |
| HellaSwag (Acc-Norm, 10-shot)           |  85.05                            |   85.06                                     |
| MMLU (Acc, 5-shot)                      |  80.69                            |   80.25                                     |
| TruthfulQA (MC2, 0-shot)                |  65.55                            |   65.69                                     |
| Winogrande (Acc, 5-shot)                |  83.11                            |   81.69                                     |
| **Average Score**                       | **79.45**                        | **78.59**                                   |
| **Recovery (%)**                            | **100.00**                       | **98.92**                                   |
