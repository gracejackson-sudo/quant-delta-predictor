---
language:
- en
- fr
- es
- pt
- hi
- de
- nl
- it
base_model:
- mistralai/Voxtral-Mini-3B-2507
pipeline_tag: automatic-speech-recognition
tags:
- voxtral
- fp8
- quantized
- multimodal
- conversational
- text-generation-inference
- automatic-speech-recognition
- automatic-speech-translation
- audio-text-to-text
- video-text-to-text
- compressed-tensors
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Voxtral-Mini-3B-2507-FP8-dynamic
description: A quantized version of the Voxtral-Mini-3B-2507 model, optimized for speech transcription, translation, and audio understanding with FP8 data type quantization.
readme: https://huggingface.co/RedHatAI/Voxtral-Mini-3B-2507-FP8-dynamic/main/README.md
tasks:
- automatic-speech-recognition
- automatic-speech-translation
- audio-to-text
- text-to-text
provider: Mistral
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 2.25
  - RHAIIS 3.2.2
  - vLLM 0.10.1.1
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Voxtral-Mini-3B-2507-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** VoxtralForConditionalGeneration
  - **Input:** Audio-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Voxtral builds upon Ministral-3B with powerful audio understanding capabilities.
  - **Dedicated transcription mode:** Voxtral can operate in a pure speech transcription mode to maximize performance. By default, Voxtral automatically predicts the source audio language and transcribes the text accordingly
  - **Long-form context:** With a 32k token context length, Voxtral handles audios up to 30 minutes for transcription, or 40 minutes for understanding
  - **Built-in Q&A and summarization:** Supports asking questions directly through audio. Analyze audio and generate structured summaries without the need for separate ASR and language models
  - **Natively multilingual:** Automatic language detection and state-of-the-art performance in the world’s most widely used languages (English, Spanish, French, Portuguese, Hindi, German, Dutch, Italian)
  - **Function-calling straight from voice:** Enables direct triggering of backend functions, workflows, or API calls based on spoken user intents
  - **Highly capable at text:** Retains the text understanding capabilities of its language model backbone, Ministral-3B
- **Release Date:** 08/21/2025
- **Version:** 1.0
- **Model Developers:** Mistral
- **ModelCar Storage URI:** oci://registry.stage.redhat.io/rhelai1/modelcar-voxtral-mini-3b-2507-fp8-dynamic:1.5
- **Validated on RHOAI 2.25:** quay.io/modh/vllm:rhoai-2.25-cuda
- **Validated on RHAIIS 3.2.2:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.2
- **Validated on vLLM:** 0.10.1.1

Quantized version of [Voxtral-Mini-3B-2507](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507).

### Model Optimizations

This model was obtained by quantizing activation and weights of [Voxtral-Mini-3B-2507](https://huggingface.co//Llama-3.3-70B-Instruct) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks of the language model are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Voxtral-Mini-3B-2507-FP8-dynamic --tokenizer_mode mistral --config_format mistral --load_format mistral
```

2. Send requests to the server, according to the use case. See the following examples.

<details>
  <summary>Audio Instruct</summary>

```python
from mistral_common.protocol.instruct.messages import TextChunk, AudioChunk, UserMessage, AssistantMessage, RawAudio
from mistral_common.audio import Audio
from huggingface_hub import hf_hub_download

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
bcn_file = hf_hub_download("patrickvonplaten/audio_samples", "bcn_weather.mp3", repo_type="dataset")

def file_to_chunk(file: str) -> AudioChunk:
    audio = Audio.from_file(file, strict=False)
    return AudioChunk.from_audio(audio)

text_chunk = TextChunk(text="Which speaker is more inspiring? Why? How are they different from each other?")
user_msg = UserMessage(content=[file_to_chunk(obama_file), file_to_chunk(bcn_file), text_chunk]).to_openai()

print(30 * "=" + "USER 1" + 30 * "=")
print(text_chunk.text)
print("\n\n")

response = client.chat.completions.create(
    model=model,
    messages=[user_msg],
    temperature=0.2,
    top_p=0.95,
)
content = response.choices[0].message.content

print(30 * "=" + "BOT 1" + 30 * "=")
print(content)
print("\n\n")
# The speaker who is more inspiring is the one who delivered the farewell address, as they express
# gratitude, optimism, and a strong commitment to the nation and its citizens. They emphasize the importance of
# self-government and active citizenship, encouraging everyone to participate in the democratic process. In contrast,
# the other speaker provides a factual update on the weather in Barcelona, which is less inspiring as it
# lacks the emotional and motivational content of the farewell address.

# **Differences:**
# - The farewell address speaker focuses on the values and responsibilities of citizenship, encouraging active participation in democracy.
# - The weather update speaker provides factual information about the temperature in Barcelona, without any emotional or motivational content.


messages = [
    user_msg,
    AssistantMessage(content=content).to_openai(),
    UserMessage(content="Ok, now please summarize the content of the first audio.").to_openai()
]
print(30 * "=" + "USER 2" + 30 * "=")
print(messages[-1]["content"])
print("\n\n")

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=0.2,
    top_p=0.95,
)
content = response.choices[0].message.content
print(30 * "=" + "BOT 2" + 30 * "=")
print(content)
```
</details>

<details>
  <summary>Transcription</summary>

```python
from mistral_common.protocol.transcription.request import TranscriptionRequest
from mistral_common.protocol.instruct.messages import RawAudio
from mistral_common.audio import Audio
from huggingface_hub import hf_hub_download

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

obama_file = hf_hub_download("patrickvonplaten/audio_samples", "obama.mp3", repo_type="dataset")
audio = Audio.from_file(obama_file, strict=False)

audio = RawAudio.from_audio(audio)
req = TranscriptionRequest(model=model, audio=audio, language="en", temperature=0.0).to_openai(exclude=("top_p", "seed"))

response = client.audio.transcriptions.create(**req)
print(response)
```
</details>

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
--enforce-eager --model RedHatAI/Voxtral-Mini-3B-2507-FP8-dynamic
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
    openshift.io/display-name: Voxtral-Mini-3B-2507-FP8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: Voxtral-Mini-3B-2507-FP8-dynamic         # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.stage.redhat.io/rhelai1/modelcar-voxtral-mini-3b-2507-fp8-dynamic:1.5
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
    "model": "Voxtral-Mini-3B-2507-FP8-dynamic",
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

This model was quantized using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as shown below.

<details>
  <summary>Creation details</summary>

```python
import torch
from transformers import VoxtralForConditionalGeneration, AutoProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# Select model and load it.
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

model = VoxtralForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Recipe
recipe = QuantizationModifier(
    targets="Linear", 
    scheme="FP8_DYNAMIC", 
    ignore=["language_model.lm_head", "re:audio_tower.*" ,"re:multi_modal_projector.*"],
)

# Apply algorithms.
oneshot(
    model=model,
    recipe=recipe,
    processor=processor,
)

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-dynamic"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)
```

After quantization, the model can be converted back into the mistralai format using the `convert_voxtral_hf_to_mistral.py` script included with the model.
</details>

## Evaluation

The model was evaluated on the Fleurs transcription task.
Recovery is computed with respect to the complement of the word error rate (WER).

<table border="1" cellspacing="0" cellpadding="6">
  <tr>
    <th>Benchmark</th>
    <th>Language</th>
    <th>Voxtral-Mini-3B-2507</th>
    <th>Voxtral-Mini-3B-2507-FP8-dynamic<br>(this model)</th>
    <th>Recovery</th>
  </tr>
  <tr>
    <td rowspan="7"><strong>Fleurs<br>WER</strong></td>
    <td>English</td>
    <td>3.89%</td>
    <td>3.95%</td>
    <td>99.9%</td>
  </tr>
  <tr>
    <td>French</td>
    <td>5.07%</td>
    <td>4.86%</td>
    <td>100.2%</td>
  </tr>
  <tr>
    <td>Spanish</td>
    <td>3.63%</td>
    <td>3.55%</td>
    <td>100.1%</td>
  </tr>
  <tr>
    <td>German</td>
    <td>5.00%</td>
    <td>5.01%</td>
    <td>100.0%</td>
  </tr>
  <tr>
    <td>Italian</td>
    <td>2.54%</td>
    <td>2.57%</td>
    <td>100.0%</td>
  </tr>
  <tr>
    <td>Portuguese</td>
    <td>3.85%</td>
    <td>4.03%</td>
    <td>99.8%</td>
  </tr>
  <tr>
    <td>Dutch</td>
    <td>7.01%</td>
    <td>7.20%</td>
    <td>99.8%</td>
  </tr>
</table>
