---
language:
- ar
- de
- en
- es
- fr
- hi
- id
- it
- pt
- th
- tl
- vi
base_model:
- meta-llama/Llama-4-Scout-17B-16E-Instruct
pipeline_tag: image-text-to-text
tags:
- facebook
- meta
- pytorch
- llama
- llama4
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP8
license: llama4
license_name: llama4
name: RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic
description: FP8-Dynamic Quantized variant of Llama-4-Scout-17B-16E-Instruct
readme: https://huggingface.co/RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic/main/README.md
tasks:
- image-text-to-text
- text-to-text
provider: Meta
license_link: https://www.llama.com/llama4/license/
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Llama-4-Scout-17B-16E-Instruct-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Llama4ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP8
  - **Weight quantization:** FP8
- **Release Date:** 04/15/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **Model Developers:** Red Hat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing activations and weights of [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%. The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

This model can be deployed efficiently on vLLM, Red Hat Enterprise Linux AI, and Openshift AI, as shown in the example below.

Deploy on <strong>vLLM</strong>

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic"
number_gpus = 4

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
--enforce-eager --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic
```
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/llama-4-scout-17b-16e-instruct-fp8-dynamic:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/llama-4-scout-17b-16e-instruct-fp8-dynamic
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/llama-4-scout-17b-16e-instruct-fp8-dynamic
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
    openshift.io/display-name: Llama-4-Scout-17B-16E-Instruct-FP8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: Llama-4-Scout-17B-16E-Instruct-FP8-dynamic          # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-llama-4-scout-17b-16e-instruct-fp8-dynamic:1.5
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
    "model": "Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",
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
#!/usr/bin/env python3
"""
This script loads an LLM model and applies FP8 quantization to
weights and activations. Activations are dynamically quantized, i.e. during
actual runtime.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Llama4ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Quantize a causal language model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        required=True,
        help="Output path for the quantized model",
    )
    return parser.parse_args()


def main():
    """Main function to load and quantize the model."""
    args = parse_arguments()

    print(f"Loading model from {args.model_path}...")
    model = Llama4ForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    quant_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            observer="mse",
        ),
        input_activations=QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.TOKEN,
            symmetric=True,
            dynamic=True,
        ),
        output_activations=None,
    )

    recipe = QuantizationModifier(
        targets="Linear",
        config_groups={"group_0": quant_scheme},
        ignore=[
            're:.*lm_head',
            're:.*self_attn',
            're:.*router',
            're:.*vision_model',
            're:.*multi_modal_projector',
        ]
    )

    print("Applying quantization...")
    oneshot(
        model=model,
        recipe=recipe,
        trust_remote_code_model=True,
    )

    model.save_pretrained(args.quant_path, save_compressed=True, skip_compression_stats=True, disable_sparse_compression=True)
    print(f"Quantized model saved to {args.quant_path}")


if __name__ == "__main__":
    main()
```
</details>
 


## Evaluation

The model was evaluated on the OpenLLM leaderboard tasks (v1 and v2), long context RULER, multimodal MMMU, and multimodal ChartQA.
All evaluations are obtained through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

<details>
  <summary>Evaluation details</summary>

  **OpenLLM v1**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8,gpu_memory_utilization=0.7,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --batch_size auto 
  ```

  **OpenLLM v2**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",dtype=auto,add_bos_token=False,max_model_len=16384,tensor_parallel_size=8,gpu_memory_utilization=0.5,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks leaderboard \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --batch_size auto 
  ```

  **Long Context RULER**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",dtype=auto,add_bos_token=False,max_model_len=524288,tensor_parallel_size=8,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks ruler \
    --metadata='{"max_seq_lengths":[131072]}' \
    --batch_size auto 
  ```

  **Multimodal MMMU**
  ```
  lm_eval \
    --model vllm-vlm \
    --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",dtype=auto,add_bos_token=False,max_model_len=1000000,tensor_parallel_size=8,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True,max_images=10 \
    --tasks mmmu_val \
    --apply_chat_template \
    --batch_size auto 
  ```

  **Multimodal ChartQA**
  ```
  export VLLM_MM_INPUT_CACHE_GIB=8
  lm_eval \
    --model vllm-vlm \
    --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic",dtype=auto,add_bos_token=False,max_model_len=1000000,tensor_parallel_size=8,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True,max_images=10 \
    --tasks chartqa \
    --apply_chat_template \
    --batch_size auto 
  ```

</details>

### Accuracy

|                                                | Recovery (%) | meta-llama/Llama-4-Scout-17B-16E-Instruct | RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic<br>(this model) |
| ---------------------------------------------- | :-----------: | :---------------------------------------: | :-----------------------------------------------------------------: |
| ARC-Challenge<br>25-shot                       | 100.36       | 69.37                                     | 69.62                                                               |
| GSM8k<br>5-shot                                | 99.24        | 90.45                                     | 89.76                                                               |
| HellaSwag<br>10-shot                           | 99.94        | 85.23                                     | 85.18                                                               |
| MMLU<br>5-shot                                 | 99.94        | 80.54                                     | 80.49                                                               |
| TruthfulQA<br>0-shot                           | 99.17        | 61.41                                     | 60.90                                                               |
| WinoGrande<br>5-shot                           | 98.88        | 77.90                                     | 77.03                                                               |
| **OpenLLM v1<br>Average Score**                    | **99.59**        | **77.48**                                     | **77.16**                                                               |
| IFEval<br>0-shot<br>avg of inst and prompt acc | 100.91       | 86.90                                     | 87.69                                                               |
| Big Bench Hard<br>3-shot                       | 99.82        | 65.13                                     | 65.01                                                               |
| Math Lvl 5<br>4-shot                           | 98.82        | 57.78                                     | 57.10                                                               |
| GPQA<br>0-shot                                 | 100.53       | 31.88                                     | 32.05                                                               |
| MuSR<br>0-shot                                 | 102.18       | 42.20                                     | 43.12                                                               |
| MMLU-Pro<br>5-shot                             | 99.82        | 55.70                                     | 55.60                                                               |
| **OpenLLM v2<br>Average Score**                    | **100.28**       | **56.60**                                     | **56.76**                                                               |
| RULER<br>seqlen = 131072<br>niah_multikey_1    | 101.36       | 88.20                                     | 89.40                                                               |
| RULER<br>seqlen = 131072<br>niah_multikey_2    | 100.72       | 83.60                                     | 84.20                                                               |
| RULER<br>seqlen = 131072<br>niah_multikey_3    | 96.19        | 78.80                                     | 75.80                                                               |
| RULER<br>seqlen = 131072<br>niah_multiquery    | 100.79       | 95.40                                     | 96.15                                                               |
| RULER<br>seqlen = 131072<br>niah_multivalue    | 97.22        | 73.75                                     | 71.70                                                               |
| RULER<br>seqlen = 131072<br>niah_single_1      | 100.00       | 100.00                                    | 100.00                                                              |
| RULER<br>seqlen = 131072<br>niah_single_2      | 100.00       | 99.80                                     | 99.80                                                               |
| RULER<br>seqlen = 131072<br>niah_single_3      | 100.00       | 99.80                                     | 99.80                                                               |
| RULER<br>seqlen = 131072<br>ruler_cwe          | 96.19        | 39.42                                     | 37.92                                                               |
| RULER<br>seqlen = 131072<br>ruler_fwe          | 98.86        | 92.93                                     | 91.87                                                               |
| RULER<br>seqlen = 131072<br>ruler_qa_hotpot    | 100.00       | 48.20                                     | 48.20                                                               |
| RULER<br>seqlen = 131072<br>ruler_qa_squad     | 98.81        | 53.57                                     | 52.93                                                               |
| RULER<br>seqlen = 131072<br>ruler_qa_vt        | 100.35       | 92.28                                     | 92.60                                                               |
| **RULER<br>seqlen = 131072<br>Average Score**      | **99.49**        | **80.44**                                     | **80.03**                                                               |
| MMMU<br>0-shot                                 | 97.92        | 53.44                                     | 52.33                                                               |
| ChartQA<br>0-shot<br>exact_match               | 100.12       | 65.88                                     | 65.96                                                               |
| ChartQA<br>0-shot<br>relaxed_accuracy          | 99.69        | 88.92                                     | 88.64                                                               |
| **Multimodal Average Score**                       | **99.38**        | **69.41**                                     | **68.98**                                                               |

