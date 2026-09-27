---
language:
- ca
- hr
- da
- nl
- en
- fi
- fr
- de
- he
- hu
- is
- id
- it
- ja
- ko
- ms
- no
- pl
- pt
- ro
- ru
- sr
- zh
- sk
- sl
- es
- sv
- th
- tr
- uk
- vi
base_model:
- google/gemma-3n-E4B-it
pipeline_tag: text-generation
tags:
- gemma
- gemma3
- gemma3n
- fp8
- quantized
- multimodal
- conversational
- text-generation-inference
- automatic-speech-recognition
- automatic-speech-translation
- audio-text-to-text
- video-text-to-text
license: gemma
license_name: gemma
name: RedHatAI/gemma-3n-E4B-it-FP8-dynamic
description: This model was obtained by quantizing the weights and activations of google/gemma-3n-E4B-it to FP8 data type.
readme: https://huggingface.co/RedHatAI/gemma-3n-E4B-it-FP8-dynamic/main/README.md
tasks:
- text-to-text
- image-to-text
- video-to-text
- audio-to-text
provider: Google
license_link: https://ai.google.dev/gemma/terms
validated_on:
  - RHOAI 2.24
  - RHAIIS 3.2.1
  - vLLM 0.10.0
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  gemma-3n-E4B-it-FP8-Dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** gemma-3n-E4B-it
  - **Input:** Audio-Vision-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 08/01/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.24, RHAIIS 3.2.1
- **Model Developers:** RedHatAI
- **ModelCar Storage URI:** oci://registry.redhat.io/rhelai1/modelcar-gemma-3n-e4b-it-fp8-dynamic:1.5
- **Validated on RHOAI 2.24:** quay.io/modh/vllm:rhoai-2.24-cuda
- **Validated on RHAIIS 3.2.1:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.1
- **Validated on vLLM:** 0.10.0

Quantized version of [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it).

### Model Optimizations

This model was obtained by quantizing the weights of [google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it) to FP8 data type, ready for inference with vLLM >= 0.10.0

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="RedHatAI/gemma-3n-E4B-it-FP8-Dynamic",
    trust_remote_code=True,
    max_model_len=4096,
    max_num_seqs=2,
)

# prepare inputs
question = "What is the content of this image?"
inputs = {
    "prompt": f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>\n",
    "multi_modal_data": {
        "image": ImageAsset("cherry_blossom").pil_image.convert("RGB")
    },
}

# generate response
print("========== SAMPLE GENERATION ==============")
outputs = llm.generate(inputs, SamplingParams(temperature=0.2, max_tokens=64))
print(f"PROMPT  : {outputs[0].prompt}")
print(f"RESPONSE: {outputs[0].outputs[0].text}")
print("==========================================")
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
--enforce-eager --model RedHatAI/gemma-3n-E4B-it-FP8-dynamic
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
    openshift.io/display-name: gemma-3n-E4B-it-FP8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: gemma-3n-E4B-it-FP8-dynamic          # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-gemma-3n-e4b-it-fp8-dynamic:1.5
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
    "model": "gemma-3n-E4B-it-FP8-dynamic",
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
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoProcessor, Gemma3nForConditionalGeneration

# Load model.
model_id = "google/gemma-3n-E4B-it"
model = Gemma3nForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Recipe
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=[
            "re:.*embed_audio.*",
            "re:.*embed_vision.*",
            "re:.*audio_tower.*",
            "re:.*vision_tower.*",
            "re:.*altup.*",
            "re:.*lm_head.*",
            "re:.*laurel.*",
            "re:model\.language_model\.layers\.\d+\.per_layer_input_gate",
            "re:model\.language_model\.layers\.\d+\.per_layer_projection",
            "model.language_model.per_layer_model_projection",
        ],
    ),
]

SAVE_DIR = f"{model_id.split('/')[1]}-{recipe[0].scheme}"

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    recipe=recipe,
    trust_remote_code_model=True,
    tie_word_embeddings=True,
    output_dir=SAVE_DIR,
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)


```
</details>

## Evaluation

The model was evaluated using [lm_evaluation_harness](https://github.com/EleutherAI/lm-evaluation-harness) for OpenLLM V1 and V2 text-based benchmarks. The evaluations were conducted using the following commands:

<details>
<summary>Evaluation Commands</summary>

### OpenLLM V1
  
```
lm_eval \
  --model vllm \
  --model_args pretrained="<model_name>",dtype=auto,add_bos_token=false,max_model_len=4096,gpu_memory_utilization=0.8,enable_chunked_prefill=True,enforce_eager=True,trust_remote_code=True \
  --tasks openllm \
  --batch_size auto \
  --apply_chat_template \
  --fewshot_as_multiturn

```

### Leaderboard V2

```
lm_eval \
  --model vllm \
  --model_args pretrained="<model_name>",dtype=auto,add_bos_token=false,max_model_len=15000,gpu_memory_utilization=0.5,enable_chunked_prefill=True,enforce_eager=True,trust_remote_code=True \
  --tasks leaderboard \
  --batch_size auto \
  --apply_chat_template \
  --fewshot_as_multiturn

```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>google/gemma-3n-E4B-it</th>
      <th>FP8 Dynamic</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>arc_challenge</td>
      <td>60.24</td>
      <td>59.04</td>
      <td>98.01%</td>
    </tr>
    <tr>
      <td>gsm8k</td>
      <td>60.12</td>
      <td>70.81</td>
      <td>117.79%</td>
    </tr>
    <tr>
      <td>hellaswag</td>
      <td>74.94</td>
      <td>73.28</td>
      <td>97.79%</td>
    </tr>
    <tr>
      <td>mmlu</td>
      <td>64.14</td>
      <td>64.82</td>
      <td>101.06%</td>
    </tr>
    <tr>
      <td>truthfulqa_mc2</td>
      <td>54.87</td>
      <td>54.61</td>
      <td>99.53%</td>
    </tr>
    <tr>
      <td>winogrande</td>
      <td>68.35</td>
      <td>67.72</td>
      <td>99.08%</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td>63.78</td>
      <td>65.05</td>
      <td><b>101.99%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>Leaderboard</b></td>
      <td>bbh</td>
      <td>55.46</td>
      <td>55.20</td>
      <td>99.53%</td>
    </tr>
    <tr>
      <td>mmlu_pro</td>
      <td>34.38</td>
      <td>34.28</td>
      <td>99.71%</td>
    </tr>
    <tr>
      <td>musr</td>
      <td>33.20</td>
      <td>34.26</td>
      <td>103.19%</td>
    </tr>
    <tr>
      <td>ifeval</td>
      <td>84.41</td>
      <td>83.93</td>
      <td>99.43%</td>
    </tr>
    <tr>
      <td>gpqa</td>
      <td>30.87</td>
      <td>31.38</td>
      <td>101.65%</td>
    </tr>
    <tr>
      <td>math_hard</td>
      <td>45.54</td>
      <td>46.60</td>
      <td>102.33%</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td>47.31</td>
      <td>47.61</td>
      <td><b>100.63%</b></td>
    </tr>
  </tbody>
</table>