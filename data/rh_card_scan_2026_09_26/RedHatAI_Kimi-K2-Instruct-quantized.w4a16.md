---
language:
- en
base_model:
- moonshotai/Kimi-K2-Instruct
pipeline_tag: text-generation
tags:
- kimi-k2
- MOE
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
- GPTQ
- conversational
- compressed-tensors
license: other
license_name: modified-mit
name: RedHatAI/Kimi-K2-Instruct-quantized.w4a16
description: >-
  This model was obtained by quantizing weights of moonshotai/Kimi-K2-Instruct to INT4 data
  type.
readme: >-
  https://huggingface.co/RedHatAI/Kimi-K2-Instruct-quantized.w4a16/main/README.md
tasks:
- text-to-text
provider: Moonshot AI
license_link: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/LICENSE
validated_on:
  - RHOAI 2.24
  - RHAIIS 3.2.1
  - vLLM 0.10.0
---
<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Kimi-K2-Instruct-quantized.w4a16
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Mixture-of-Experts (MoE)
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** None
  - **Weight quantization:** INT4
- **Release Date:** 07/15/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.24, RHAIIS 3.2.1
- **Model Developers:** Red Hat (Neural Magic)
- **ModelCar Storage URI:** oci://registry.redhat.io/rhelai1/modelcar-kimi-k2-instruct-quantized-w4a16:1.5
- **Validated on RHOAI 2.24:** quay.io/modh/vllm:rhoai-2.24-cuda
- **Validated on RHAIIS 3.2.1:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1
- **Validated on vLLM:** 0.10.0



## 1. Model Introduction

This model was obtained by quantizing the weights of **`Kimi-K2-Instruct`** to the INT4 data type. This optimization reduces the number of bits used to represent weights from 16 (FP16/BF16) to 4, reducing GPU memory requirements (by approximately 75%). This weight quantization also reduces the model's disk size by approximately 75%.

The original `Kimi K2` is a state-of-the-art mixture-of-experts (MoE) language model with 32 billion activated parameters and 1 trillion total parameters. Trained with the Muon optimizer, Kimi K2 achieves exceptional performance across frontier knowledge, reasoning, and coding tasks while being meticulously optimized for agentic capabilities.

### Key Features
- INT4 Quantization: This model has been quantized to INT4, dramatically reducing memory footprint and enabling high-throughput, low-latency inference.
- Large-Scale Training: Pre-trained a 1T parameter MoE model on 15.5T tokens with zero training instability.
- MuonClip Optimizer: We apply the Muon optimizer to an unprecedented scale, and develop novel optimization techniques to resolve instabilities while scaling up.
- Agentic Intelligence: Specifically designed for tool use, reasoning, and autonomous problem-solving.

### Model Variants
- **Kimi-K2-Base**: The foundation model, a strong start for researchers and builders who want full control for fine-tuning and custom solutions.
- **Kimi-K2-Instruct**: The post-trained model best for drop-in, general-purpose chat and agentic experiences. It is a reflex-grade model without long thinking.
- **RedHatAI/Kimi-K2-Instruct-quantized.int4 (This Model)**: An INT4 quantized version of `Kimi-K2-Instruct` for efficient, high-performance inference, validated by Red Hat.

<div align="center">
  <picture>
      <img src="figures/banner.png" width="80%" alt="Evaluation Results">
  </picture>
</div>

## 2. Model Summary

<div align="center">


| | |
|:---:|:---:|
| **Architecture** | Mixture-of-Experts (MoE) |
| **Total Parameters** | 1T |
| **Activated Parameters** | 32B |
| **Number of Layers** (Dense layer included) | 61 |
| **Number of Dense Layers** | 1 |
| **Attention Hidden Dimension** | 7168 |
| **MoE Hidden Dimension** (per Expert) | 2048 |
| **Number of Attention Heads** | 64 |
| **Number of Experts** | 384 |
| **Selected Experts per Token** | 8 |
| **Number of Shared Experts** | 1 |
| **Vocabulary Size** | 160K |
| **Context Length** | 128K |
| **Attention Mechanism** | MLA |
| **Activation Function** | SwiGLU |
</div>

## 3. Preliminary Evaluations


- GSM8k, 5-shot via lm-evaluation-harness
```
moonshotai/Kimi-K2-Instruct                            = 94.92
RedHatAI/Kimi-K2-Instruct-quantized.w4a16 (this model) = 94.84
```
More evals coming very soon...


## Deployment

This model can be deployed efficiently on vLLM, Red Hat Enterprise Linux AI, and Openshift AI, as shown in the example below.

Deploy on <strong>vLLM</strong>

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Kimi-K2-Instruct-quantized.w4a16"
number_gpus = 8

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

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
--enforce-eager --model RedHatAI/Kimi-K2-Instruct-quantized.w4a16
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
    openshift.io/display-name: kimi-k2-instruct-quantized-w4a16 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: kimi-k2-instruct-quantized-w4a16          # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.stage.redhat.io/rhelai1/modelcar-kimi-k2-instruct-quantized-w4a16:1.5
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
    "model": "kimi-k2-instruct-quantized-w4a16",
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


---

## 5. Model Usage

### Chat Completion

Once the local inference service is up, you can interact with it through the chat endpoint:

```python
def simple_chat(client: OpenAI, model_name: str):
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": [{"type": "text", "text": "Please give a brief self-introduction."}]},
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        stream=False,
        temperature=0.6,
        max_tokens=256
    )
    print(response.choices[0].message.content)
```

> [!NOTE]
> The recommended temperature for Kimi-K2-Instruct.w4a16 is `temperature = 0.6`.
> If no special instructions are required, the system prompt above is a good default.

---

### Tool Calling

Kimi-K2-Instruct.w4a16 has strong tool-calling capabilities.
To enable them, you need to pass the list of available tools in each request, then the model will autonomously decide when and how to invoke them.

The following example demonstrates calling a weather tool end-to-end:

```python
# Your tool implementation
def get_weather(city: str) -> dict:
    return {"weather": "Sunny"}

# Tool schema definition
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Retrieve current weather information. Call this when the user asks about the weather.",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {
                    "type": "string",
                    "description": "Name of the city"
                }
            }
        }
    }
}]

# Map tool names to their implementations
tool_map = {
    "get_weather": get_weather
}

def tool_call_with_client(client: OpenAI, model_name: str):
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": "What's the weather like in Beijing today? Use the tool to check."}
    ]
    finish_reason = None
    while finish_reason is None or finish_reason == "tool_calls":
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6,
            tools=tools,          # tool list defined above
            tool_choice="auto"
        )
        choice = completion.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason == "tool_calls":
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                tool_call_name = tool_call.function.name
                tool_call_arguments = json.loads(tool_call.function.arguments)
                tool_function = tool_map[tool_call_name]
                tool_result = tool_function(**tool_call_arguments)
                print("tool_result:", tool_result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call_name,
                    "content": json.dumps(tool_result)
                })
    print("-" * 100)
    print(choice.message.content)
```

The `tool_call_with_client` function implements the pipeline from user query to tool execution.
This pipeline requires the inference engine to support Kimi-K2’s native tool-parsing logic.
For streaming output and manual tool-parsing, see the [Tool Calling Guide](docs/tool_call_guidance.md).

---

## 6. License

Both the code repository and the model weights are released under the [Modified MIT License](LICENSE).

---

## 7. Third Party Notices

See [THIRD PARTY NOTICES](THIRD_PARTY_NOTICES.md)