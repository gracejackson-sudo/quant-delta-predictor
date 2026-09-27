---
language:
- en
base_model:
- openai/whisper-large-v3-turbo
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- w4a16
- int4
- vllm
- audio
- compressed-tensors
- text-generation-inference
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/whisper-large-v3-turbo-quantized.w4a16
description: This model was obtained by quantizing the weights of openai/whisper-large-v3-turbo to INT4 data type.
readme: https://huggingface.co/RedHatAI/whisper-large-v3-turbo-quantized.w4a16/main/README.md
tasks:
- automatic-speech-recognition
- automatic-speech-translation
provider: OpenAI
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 2.25
  - RHAIIS 3.2.2
  - vLLM 0.10.1.1
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  whisper-large-v3-turbo-quantized.w4a16
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>


## Model Overview
- **Model Architecture:** whisper-large-v3-turbo
  - **Input:** Audio-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Release Date:** 04/16/2025
- **Version:** 1.0
- **Model Developers:** Open AI

Quantized version of [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo).

### Model Optimizations

This model was obtained by quantizing the weights of [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) to INT4 data type, ready for inference with vLLM >= 0.5.2.

- **ModelCar Storage URI:** oci://registry.redhat.io/rhelai1/modelcar-whisper-large-v3-turbo-quantized-w4a16:1.5
- **Validated on RHOAI 2.25:** quay.io/modh/vllm:rhoai-2.25-cuda
- **Validated on RHAIIS 3.2.2:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.2
- **Validated on vLLM:** 0.10.1.1

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.audio import AudioAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="neuralmagic/whisper-large-v3-turbo-quantized.w4a16",
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
)

# prepare inputs
inputs = {  # Test explicit encoder/decoder prompt
    "encoder_prompt": {
        "prompt": "",
        "multi_modal_data": {
            "audio": AudioAsset("winning_call").audio_and_sample_rate,
        },
    },
    "decoder_prompt": "<|startoftranscript|>",
}

# generate response
print("========== SAMPLE GENERATION ==============")
outputs = llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=64))
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
--enforce-eager --model RedHatAI/whisper-large-v3-turbo-quantized.w4a16
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
    openshift.io/display-name: whisper-large-v3-turbo-quantized.w4a16 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: whisper-large-v3-turbo-quantized.w4a16          # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-whisper-large-v3-turbo-quantized-w4a16:1.5
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
    "model": "whisper-large-v3-turbo-quantized.w4a16",
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

```bash
python quantize.py --model_path openai/whisper-large-v3-turbo --quant_path "output_dir/whisper-large-v3-turbo-quantized.w4a16" --calib_size 1024 --group_size 64 --dampening_frac 0.01 --actorder weight
```


```python
import torch
import argparse
from datasets import load_dataset
from transformers import WhisperProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers.tracing import TraceableWhisperForConditionalGeneration
import os
from compressed_tensors.quantization import QuantizationArgs, QuantizationType, QuantizationStrategy, ActivationOrdering, QuantizationScheme
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--quant_path', type=str)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.1) 
parser.add_argument('--observer', type=str, default="minmax")
parser.add_argument('--actorder', type=str, default="dynamic")
parser.add_argument('--group_size', type=int, default=128)
parser.add_argument('--save_dir', type=str, required=True)


args = parser.parse_args()
model_id = args.model_path

model = TraceableWhisperForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(model_id)

# Configure processor the dataset task.
processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

# Select number of samples for calibration. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.

NUM_CALIBRATION_SAMPLES = args.calib_size
MAX_SEQUENCE_LENGTH = 2048
dampening_frac=args.dampening_frac
actorder_arg=args.actorder
group_size=args.group_size

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    trust_remote_code=True,
)

def preprocess(example):
    return {
        "array": example["audio"]["array"],
        "sampling_rate": example["audio"]["sampling_rate"],
        "text": " " + example["text"].capitalize(),
    }

ds = ds.map(preprocess, remove_columns=ds.column_names)

# Process inputs.
def process(sample):
    inputs = processor(
        audio=sample["array"],
        sampling_rate=sample["sampling_rate"],
        text=sample["text"],
        add_special_tokens=True,
        return_tensors="pt",
    )

    inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
    inputs["decoder_input_ids"] = inputs["labels"]
    del inputs["labels"]

    return inputs

ds = ds.map(process, remove_columns=ds.column_names)

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

ignore=["lm_head"]

# Recipe
recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "config_group": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=group_size,
                symmetric=True,
                dynamic=False,
                actorder=getattr(ActivationOrdering, actorder_arg.upper()),
            ),
        ),
    },
    sequential_targets=["WhisperEncoderLayer", "WhisperDecoderLayer"],
    ignore=["re:.*lm_head"],
    update_size=NUM_CALIBRATION_SAMPLES,
    dampening_frac=dampening_frac
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)


# Save to disk compressed.
save_name = f"{model_id.split('/')[-1]}-quantized.w4a16"
save_path = os.path.join(args.save_dir, save_name)
print("Saving model:", save_path)
model.save_pretrained(save_path, save_compressed=True)
processor.save_pretrained(save_path)
```
</details>

## Evaluation

The model was evaluated on [LibriSpeech](https://huggingface.co/datasets/lmms-lab/librispeech) and [Fleurs](https://huggingface.co/datasets/lmms-lab/fleurs) datasets using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), via the following commands:

<details>
<summary>Evaluation Commands</summary>
  
Librispeech:
```
lmms-eval \
    --model=whisper_vllm \
    --model_args="pretrained=neuralmagic-ent/whisper-large-v3-turbo-quantized.w4a16" \
    --batch_size 64 \
    --output_path <output_file_path> \
    --tasks librispeech
```

Fleurs:
```
lmms-eval \
    --model=whisper_vllm \
    --model_args="pretrained=neuralmagic-ent/whisper-large-v3-turbo-quantized.w4a16" \
    --batch_size 64 \
    --output_path <output_file_path> \
    --tasks fleurs
```
</details>

<table>
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>Split</th>
      <th>BF16</th>
      <th>W4A16</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>LibriSpeech (WER)</b></td>
      <td>test-clean</td>
      <td>2.1876</td>
      <td>2.1951</td>
      <td>99.66%</td>
    </tr>
    <tr>
      <td>test-other</td>
      <td>3.8992</td>
      <td>4.0411</td>
      <td>96.49%</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Fleurs (X→en, WER)</b></td>
      <td>cmn_hans_cn</td>
      <td>7.8019</td>
      <td>8.3448</td>
      <td>93.49%</td>
    </tr>
    <tr>
      <td>en</td>
      <td>4.0236</td>
      <td>4.0580</td>
      <td>99.15</td>
    </tr>
    <tr>
      <td>yue_hant_hk</td>
      <td>9.4210</td>
      <td>11.8108</td>
      <td>97.77%</td>
    </tr>
  </tbody>
</table>
