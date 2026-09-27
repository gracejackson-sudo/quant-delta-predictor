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
- w4a16
- int4
- vllm
- conversational
- text-generation-inference
- compressed-tensors
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w4a16
description: This model was obtained by quantizing the weights of Mistral-Small-24B-Instruct-2501 to INT4 data type.
readme: https://huggingface.co/RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w4a16/main/README.md
tasks:
- text-to-text
provider: Mistral AI
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Mistral-Small-24B-Instruct-2501-quantized.w4a16
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Mistral-Small-24B-Instruct-2501
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** None
- **Release Date:** 3/1/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **Model Developers:** Neural Magic

Quantized version of [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501).

### Model Optimizations

This model was obtained by quantizing the weights to INT4 data type, ready for inference with vLLM.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%. Only the weights of the linear operators within transformers blocks are quantized.

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w4a16 --tensor_parallel_size 1 --tokenizer_mode mistral
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

model = "RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w4a16"


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
--enforce-eager --model RedHatAI/Mistral-Small-24B-Instruct-2501-quantized.w4a16
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/mistral-small-24b-instruct-2501-quantized-w4a16:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/mistral-small-24b-instruct-2501-quantized-w4a16 --gpu 1 -- --trust-remote-code
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/mistral-small-24b-instruct-2501-quantized-w4a16
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
    openshift.io/display-name: Mistral-Small-24B-Instruct-2501-quantized.w4a16 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: Mistral-Small-24B-Instruct-2501-quantized.w4a16         # specify model name. This value will be used to invoke the model in the payload
  labels:
    opendatahub.io/dashboard: 'true'
spec:
  predictor:
    maxReplicas: 1
    minReplicas: 1
    model:
      args:
        - "--trust-remote-code"
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-mistral-small-24b-instruct-2501-quantized-w4a16:1.5
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
    "model": "Mistral-Small-24B-Instruct-2501-quantized.w4a16",
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

```bash
python quantize.py --model_path mistralai/Mistral-Small-24B-Instruct-2501 --quant_path "output_dir" --calib_size 1024 --dampening_frac 0.05 --observer minmax --actorder false
```

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot, apply
import argparse
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy

def parse_actorder(value):
    # Interpret the input value for --actorder
    if value.lower() == "false":
        return False
    elif value.lower() == "group":
        return "group"
    elif value.lower() == "weight":
        return "weight"
    else:
        raise argparse.ArgumentTypeError("Invalid value for --actorder. Use 'group' or 'False'.")


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--quant_path', type=str)
parser.add_argument('--num_bits', type=int, default=4)
parser.add_argument('--sequential_update', type=bool, default=True)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.05)
parser.add_argument('--observer', type=str, default="minmax")
parser.add_argument(
    '--actorder',
    type=parse_actorder,
    default=False,  # Default value is False
    help="Specify actorder as 'group' (string) or False (boolean)."
)

args = parser.parse_args()

model = SparseAutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype="auto",
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

NUM_CALIBRATION_SAMPLES = args.calib_size
DATASET_ID = "garage-bAInd/Open-Platypus"
DATASET_SPLIT = "train"
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    concat_txt = example["instruction"] + "\n" + example["output"]
    return {"text": concat_txt}

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=args.num_bits,
        type=QuantizationType.INT,
        symmetric=True,
        group_size=128,
        strategy=QuantizationStrategy.GROUP,
        observer=args.observer,
        actorder=args.actorder
    ),
    input_activations=None,
    output_activations=None,
)

recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        sequential_update=args.sequential_update,
        dampening_frac=args.dampening_frac,
        config_groups={"group_0": quant_scheme},
    )
]
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=args.calib_size,
)

# Save to disk compressed.
SAVE_DIR = args.quant_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) and [V2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/), using the following commands:

OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Mistral-Small-24B-Instruct-2501-quantized.w4a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
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
  --model_args pretrained="neuralmagic/Mistral-Small-24B-Instruct-2501-quantized.w4a16",dtype=auto,add_bos_token=False,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
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


| Metric                                   | mistralai/Mistral-Small-24B-Instruct-2501             | neuralmagic/Mistral-Small-24B-Instruct-2501-quantized.w4a16 |
|-----------------------------------------|:---------------------------------:|:-------------------------------------------:|
| ARC-Challenge (Acc-Norm, 25-shot)       |  72.18                            |    71.16                                    |
| GSM8K (Strict-Match, 5-shot)            |  90.14                            |    89.69                                     |
| HellaSwag (Acc-Norm, 10-shot)           |  85.05                            |    84.43                                    |
| MMLU (Acc, 5-shot)                      |  80.69                            |    80.00                                    |
| TruthfulQA (MC2, 0-shot)                |  65.55                            |    63.92                                    |
| Winogrande (Acc, 5-shot)                |  83.11                            |    82.24                                    |
| **Average Score**                       | **79.45**                        | **78.57**                                   |
| **Recovery (%)**                            | **100.00**                       | **98.9**                                   |

#### OpenLLM Leaderboard V2 evaluation scores

| Metric                                                   | mistralai/Mistral-Small-24B-Instruct-2501             | neuralmagic/Mistral-Small-24B-Instruct-2501-quantized.w4a16 |
|---------------------------------------------------------|:---------------------------------:|:-------------------------------------------:|
| IFEval (Inst-and-Prompt Level Strict Acc, 0-shot)       |     73.27                        |    74.37                                    |
| BBH (Acc-Norm, 3-shot)                                  |     45.18                        |    45.15                                       |
| MMLU-Pro (Acc, 5-shot)                                  |      38.83                       |    36.00                                    |
| **Average Score**                                       | **52.42**                        | **51.84**                                   |
| **Recovery (%)**                                            | **100.00**                       | **98.89**                                   |
| GPQA (Acc-Norm, 0-shot)                                 |      8.29                        |   6.81                                      |
| MUSR (Acc-Norm, 0-shot)                                 |      7.84                        |   9.46                                     |

Results on GPQA and MUSR are not considred for accuracy recovery calculation because the unquantized model has close to random prediction accuracy (8.29, 7.84) which doesn't provide a reliable baseline for recovery calculation.

