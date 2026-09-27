---
language:
- en
- fr
- de
- es
- it
- pt
- hi
- id
- tl
- vi
- ar
- bg
- zh
- da
- el
- fa
- fi
- he
- ja
- ko
- ms
- nl
- no
- pl
- ro
- ru
- sr
- sv
- th
- tr
- uk
- ur
- zsm
- nld
base_model:
- mistralai/Mistral-Small-3.1-24B-Instruct-2503
pipeline_tag: image-text-to-text
tags:
- mistralai
- mistral
- mistral3
- mistral-small
- neuralmagic
- redhat
- llmcompressor
- quantized
- W4A16
- INT4
- conversational
- compressed-tensors
- fast
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16
description: This model was obtained by quantizing the weights of Mistral-Small-3.1-24B-Instruct-2503 to INT4 data type.
readme: https://huggingface.co/RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16/main/README.md
tasks:
- image-text-to-text
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
  Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16
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
  - **Weight quantization:** INT4
- **Intended Use Cases:** It is ideal for:
  - Fast-response conversational agents.
  - Low-latency function calling.
  - Subject matter experts via fine-tuning.
  - Local inference for hobbyists and organizations handling sensitive data.
  - Programming and math reasoning.
  - Long document understanding.
  - Visual understanding.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages not officially supported by the model.
- **Release Date:** 04/15/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **Model Developers:** Red Hat (Neural Magic)


### Model Optimizations

This model was obtained by quantizing the weights of [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-group scheme, with group size 128.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

1. Initialize vLLM server:
```
vllm serve RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16 --tokenizer_mode mistral --config-format mistral
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

model = "RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16"


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
--config-format mistral \
--max-model-len 32768  \
--enforce-eager --model RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/mistral-small-3-1-24b-instruct-2503-quantized-w4a16:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/mistral-small-3-1-24b-instruct-2503-quantized-w4a16
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/mistral-small-3-1-24b-instruct-2503-quantized-w4a16
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
    openshift.io/display-name: mistral-small-3-1-24b-instruct-2503-quantized-w4a16 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: mistral-small-3-1-24b-instruct-2503-quantized-w4a16       # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-mistral-small-3-1-24b-instruct-2503-quantized-w4a16:1.5
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
    "model": "mistral-small-3-1-24b-instruct-2503-quantized-w4a16",
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
  from transformers import AutoProcessor
  from llmcompressor.modifiers.quantization import GPTQModifier
  from llmcompressor.transformers import oneshot
  from llmcompressor.transformers.tracing import TraceableMistral3ForConditionalGeneration
  from datasets import load_dataset, interleave_datasets
  from PIL import Image
  import io
  
  # Load model
  model_stub = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  model_name = model_stub.split("/")[-1]
  
  num_text_samples = 1024
  num_vision_samples = 1024
  max_seq_len = 8192
  
  processor = AutoProcessor.from_pretrained(model_stub)
  
  model = TraceableMistral3ForConditionalGeneration.from_pretrained(
      model_stub,
      device_map="auto",
      torch_dtype="auto",
  )

  # Text-only data subset
  def preprocess_text(example):
      input = {
          "text": processor.apply_chat_template(
              example["messages"],
              add_generation_prompt=False,
          ),
          "images": None,
      }
      tokenized_input = processor(**input, max_length=max_seq_len, truncation=True)
      tokenized_input["pixel_values"] = tokenized_input.get("pixel_values", None)
      tokenized_input["image_sizes"] = tokenized_input.get("image_sizes", None)
      return tokenized_input

  dst = load_dataset("neuralmagic/calibration", name="LLM", split="train").select(range(num_text_samples))
  dst = dst.map(preprocess_text, remove_columns=dst.column_names)

  # Text + vision data subset
  def preprocess_vision(example):
      messages = []
      image = None
      for message in example["messages"]:
          message_content = []
          for content in message["content"]:
              if content["type"] == "text":
                  message_content.append({"type": "text", "text": content["text"]})
              else:
                  message_content.append({"type": "image"})
                  image = Image.open(io.BytesIO(content["image"]))

          messages.append(
              {
                  "role": message["role"],
                  "content": message_content,
              }
          )

      input = {
          "text": processor.apply_chat_template(
              messages,
              add_generation_prompt=False,
          ),
          "images": image,
      }
      tokenized_input = processor(**input, max_length=max_seq_len, truncation=True)
      tokenized_input["pixel_values"] = tokenized_input.get("pixel_values", None)
      tokenized_input["image_sizes"] = tokenized_input.get("image_sizes", None)
      return tokenized_input

  dsv = load_dataset("neuralmagic/calibration", name="VLM", split="train").select(range(num_vision_samples))
  dsv = dsv.map(preprocess_vision, remove_columns=dsv.column_names)

  # Interleave subsets
  ds = interleave_datasets((dsv, dst))

  # Configure the quantization algorithm and scheme
  recipe = GPTQModifier(
      ignore=["language_model.lm_head", "re:vision_tower.*", "re:multi_modal_projector.*"],
      sequential_targets=["MistralDecoderLayer"],
      dampening_frac=0.01,
      targets="Linear",
      scheme="W4A16",
  )

  # Define data collator
  def data_collator(batch):
      import torch
      assert len(batch) == 1
      collated = {}
      for k, v in batch[0].items():
          if v is None:
              continue
          if k == "input_ids":
              collated[k] = torch.LongTensor(v)
          elif k == "pixel_values":
              collated[k] = torch.tensor(v, dtype=torch.bfloat16)
          else:
              collated[k] = torch.tensor(v)
      return collated


  # Apply quantization
  oneshot(
      model=model,
      dataset=ds, 
      recipe=recipe,
      max_seq_length=max_seq_len,
      data_collator=data_collator,
      num_calibration_samples=num_text_samples + num_vision_samples,
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-quantized.w4a16"
  model.save_pretrained(save_path)
  processor.save_pretrained(save_path)
  print(f"Model and tokenizer saved to: {save_path}")
  ```
</details>
 


## Evaluation

The model was evaluated on the OpenLLM leaderboard tasks (version 1), MMLU-pro, GPQA, HumanEval and MBPP.
Non-coding tasks were evaluated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), whereas coding tasks were evaluated with a fork of [evalplus](https://github.com/neuralmagic/evalplus).
[vLLM](https://docs.vllm.ai/en/stable/) is used as the engine in all cases.

<details>
  <summary>Evaluation details</summary>

  **MMLU**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks mmlu \
    --num_fewshot 5 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **ARC Challenge**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks arc_challenge \
    --num_fewshot 25 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **GSM8k**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.9,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks gsm8k \
    --num_fewshot 8 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **Hellaswag**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks hellaswag \
    --num_fewshot 10 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **Winogrande**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks winogrande \
    --num_fewshot 5 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **TruthfulQA**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks truthfulqa \
    --num_fewshot 0 \
    --apply_chat_template\
    --batch_size auto
  ```

  **MMLU-pro**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=8192,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks mmlu_pro \
    --num_fewshot 5 \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **MMMU**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.9,max_images=8,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks mmmu_val \
    --apply_chat_template\
    --batch_size auto
  ```

  **ChartQA**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.9,max_images=8,enable_chunk_prefill=True,tensor_parallel_size=2 \
    --tasks chartqa \
    --apply_chat_template\
    --batch_size auto
  ```

**Coding**

The commands below can be used for mbpp by simply replacing the dataset name.

*Generation*
```
python3 codegen/generate.py \
  --model RedHatAI/Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16 \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval

```

*Sanitization*
```
python3 evalplus/sanitize.py \
  humaneval/RedHatAI--Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16_vllm_temp_0.2
```

*Evaluation*
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/RedHatAI--Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16_vllm_temp_0.2-sanitized
```
</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>Mistral-Small-3.1-24B-Instruct-2503
   </th>
   <th>Mistral-Small-3.1-24B-Instruct-2503-quantized.w4a16<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="7" ><strong>OpenLLM v1</strong>
   </td>
   <td>MMLU (5-shot)
   </td>
   <td>80.67
   </td>
   <td>79.74
   </td>
   <td>98.9%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot)
   </td>
   <td>72.78
   </td>
   <td>72.18
   </td>
   <td>99.2%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>58.68
   </td>
   <td>59.59
   </td>
   <td>101.6%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>83.70
   </td>
   <td>83.25
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>83.74
   </td>
   <td>83.43
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>70.62
   </td>
   <td>69.56
   </td>
   <td>98.5%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>75.03</strong>
   </td>
   <td><strong>74.63</strong>
   </td>
   <td><strong>99.5%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="3" ><strong></strong>
   </td>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>67.25
   </td>
   <td>66.56
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td>GPQA CoT main (5-shot)
   </td>
   <td>42.63
   </td>
   <td>47.10
   </td>
   <td>110.5%
   </td>
  </tr>
  <tr>
   <td>GPQA CoT diamond (5-shot)
   </td>
   <td>45.96
   </td>
   <td>44.95
   </td>
   <td>97.80%
   </td>
  </tr>
  <tr>
   <td rowspan="4" ><strong>Coding</strong>
   </td>
   <td>HumanEval pass@1
   </td>
   <td>84.70
   </td>
   <td>84.60
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>79.50
   </td>
   <td>79.90
   </td>
   <td>100.5%
   </td>
  </tr>
  <tr>
   <td>MBPP pass@1
   </td>
   <td>71.10
   </td>
   <td>70.10
   </td>
   <td>98.6%
   </td>
  </tr>
  <tr>
   <td>MBPP+ pass@1
   </td>
   <td>60.60
   </td>
   <td>60.70
   </td>
   <td>100.2%
   </td>
  </tr>
  <tr>
   <td rowspan="2" ><strong>Vision</strong>
   </td>
   <td>MMMU (0-shot)
   </td>
   <td>52.11
   </td>
   <td>50.11
   </td>
   <td>96.2%
   </td>
  </tr>
  <tr>
   <td>ChartQA (0-shot)
   </td>
   <td>81.36
   </td>
   <td>80.92
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
</table>

