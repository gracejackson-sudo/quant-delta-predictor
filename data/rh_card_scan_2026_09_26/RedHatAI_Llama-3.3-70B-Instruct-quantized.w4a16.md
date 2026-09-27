---
language:
- en
- de
- fr
- it
- pt
- hi
- es
- th
base_model:
- meta-llama/Llama-3.3-70B-Instruct
pipeline_tag: text-generation
tags:
- llama
- facebook
- meta
- llama-3
- int4
- vllm
- chat
- neuralmagic
- llmcompressor
- conversational
- 4-bit precision
- compressed-tensors
license: llama3.3
license_name: llama3.3
name: RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16
description: This model was obtained by quantizing the weights of Llama-3.3-70B-Instruct to INT4 data type.
readme: https://huggingface.co/RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16/main/README.md
tasks:
- text-to-text
provider: Meta
license_link: https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---
<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Llama-3.3-70B-Instruct-quantized.w4a16
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Meta-Llama-3.1
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. Instruction tuned text only models are intended for assistant-like chat, whereas pretrained models can be adapted for a variety of natural language generation tasks. The Llama 3.3 model also supports the ability to leverage the outputs of its models to improve other models including synthetic data generation and distillation. The Llama 3.3 Community License allows for these use cases.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in any other way that is prohibited by the Acceptable Use Policy and Llama 3.3 Community License. Use in languages beyond English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai.
- **Release Date:** 12/11/2024
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **License(s):** llama3.3
- **Model Developers:** Red Hat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights of [Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-group scheme, with group size 128.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16"
number_gpus = 1

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False)

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
--enforce-eager --model RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/llama-3-3-70b-instruct-quantized-w4a16:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/llama-3-3-70b-instruct-quantized-w4a16
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/llama-3-3-70b-instruct-quantized-w4a16
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
    openshift.io/display-name: llama-3-3-70b-instruct-quantized-w4a16 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: llama-3-3-70b-instruct-quantized-w4a16          # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-llama-3-3-70b-instruct-quantized-w4a16:1.5
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
    "model": "llama-3-3-70b-instruct-quantized-w4a16",
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
  from llmcompressor.transformers import oneshot
  from datasets import load_dataset
  
  # Load model
  model_stub = "meta-llama/Llama-3.3-70B-Instruct"
  model_name = model_stub.split("/")[-1]

  num_samples = 1024
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
  recipe = GPTQModifier(
      targets="Linear",
      scheme="W4A16",
      ignore=["lm_head"],
      sequential_targets=["LlamaDecoderLayer"],
      dampening_frac=0.01,
  )
  
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

This model was evaluated on the well-known OpenLLM v1, HumanEval, and HumanEval+ benchmarks.
In all cases, model outputs were generated with the [vLLM](https://docs.vllm.ai/en/stable/) engine.

OpenLLM v1 evaluations were conducted using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and the prompting style of [Meta-Llama-3.1-Instruct-evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals) when available.

HumanEval and HumanEval+ evaluations were conducted using Neural Magic's fork of the [EvalPlus](https://github.com/neuralmagic/evalplus) repository.

<details>
  <summary>Evaluation details</summary>

  **MMLU**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
    --tasks mmlu_llama \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --num_fewshot 5 \
    --batch_size auto
  ```

  **MMLU-CoT**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",dtype=auto,max_model_len=4064,max_gen_toks=1024,tensor_parallel_size=1 \
    --tasks mmlu_cot_llama \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto
  ```

  **ARC-Challenge**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",dtype=auto,max_model_len=3940,max_gen_toks=100,tensor_parallel_size=1 \
    --tasks arc_challenge_llama \
    --apply_chat_template \
    --num_fewshot 0 \
    --batch_size auto
  ```

  **GSM-8K**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
    --tasks gsm8k_llama \
    --fewshot_as_multiturn \
    --apply_chat_template \
    --num_fewshot 8 \
    --batch_size auto
  ```

  **Hellaswag**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
    --tasks hellaswag \
    --num_fewshot 10 \
    --batch_size auto
  ```

  **Winogrande**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
    --tasks winogrande \
    --num_fewshot 5 \
    --batch_size auto
  ```

  **TruthfulQA**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
    --tasks truthfulqa \
    --num_fewshot 0 \
    --batch_size auto
  ```

  **HumanEval and HumanEval+**
  *Generation*
  ```
  python3 codegen/generate.py \
    --model RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16 \
    --bs 16 \
    --temperature 0.2 \
    --n_samples 50 \
    --root "." \
    --dataset humaneval
  ```

  *Sanitization*
  ```
  python3 evalplus/sanitize.py \
    humaneval/RedHatAI--Llama-3.3-70B-Instruct-quantized.w4a16_vllm_temp_0.2
  ```

  *Evaluation*
  ```
  evalplus.evaluate \
    --dataset humaneval \
    --samples humaneval/RedHatAI--Llama-3.3-70B-Instruct-quantized.w4a16_vllm_temp_0.2-sanitized
  ```
</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>Llama-3.3-70B-Instruct
   </th>
   <th>Llama-3.3-70B-Instruct-quantized.w4a16<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="8" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>81.60
   </td>
   <td>80.62
   </td>
   <td>98.8%
   </td>
  </tr>
  <tr>
   <td>MMLU (CoT, 0-shot)
   </td>
   <td>86.58
   </td>
   <td>85.81
   </td>
   <td>99.1%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (0-shot)
   </td>
   <td>49.23
   </td>
   <td>49.49
   </td>
   <td>100.5%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (CoT, 8-shot, strict-match)
   </td>
   <td>94.16
   </td>
   <td>94.47
   </td>
   <td>100.3%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>86.49
   </td>
   <td>85.97
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>84.77
   </td>
   <td>84.45
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>62.75
   </td>
   <td>61.66
   </td>
   <td>98.3%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>77.94</strong>
   </td>
   <td><strong>77.49</strong>
   </td>
   <td><strong>98.3%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>Coding</strong>
   </td>
   <td>HumanEval pass@1
   </td>
   <td>83.20
   </td>
   <td>83.40
   </td>
   <td>100.2%
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>78.40
   </td>
   <td>78.60
   </td>
   <td>100.3%
   </td>
  </tr>
</table>


