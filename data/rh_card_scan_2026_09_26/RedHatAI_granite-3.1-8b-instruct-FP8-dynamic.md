---
language:
- en
- de
- es
- fr
- ja
- pt
- ar
- cs
- it
- ko
- nl
- zh
base_model:
- ibm-granite/granite-3.1-8b-instruct
pipeline_tag: text-generation
tags:
- granite
- fp8
- vllm
- conversational
- compressed-tensors
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/granite-3.1-8b-instruct-FP8-dynamic
description: This model was obtained by quantizing the weights and activations of ibm-granite/granite-3.1-8b-instruct to FP8 data type.
readme: https://huggingface.co/RedHatAI/granite-3.1-8b-instruct-FP8-dynamic/main/README.md
tasks:
- text-to-text
provider: IBM
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---
<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Granite-3.1-8b-instruct-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** granite-3.1-8b-instruct
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 1/8/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **Model Developers:** Neural Magic

Quantized version of [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct).
It achieves an average score of 70.57 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 70.30.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) to FP8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%. Only the weights and activations of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 1
model_name = "neuralmagic/granite-3.1-8b-instruct-FP8-dynamic"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])

messages_list = [
    [{"role": "user", "content": "Who are you? Please respond in pirate speak!"}],
]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
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
--tensor-parallel-size 1 \
--max-model-len 32768  \
--enforce-eager --model RedHatAI/granite-3.1-8b-instruct-FP8-dynamic
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/granite-3-1-8b-instruct-fp8-dynamic:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/granite-3-1-8b-instruct-fp8-dynamic -- --trust-remote-code
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/granite-3-1-8b-instruct-fp8-dynamic
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
    openshift.io/display-name: granite-3-1-8b-instruct-fp8-dynamic # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: granite-3-1-8b-instruct-fp8-dynamic          # specify model name. This value will be used to invoke the model in the payload
  labels:
    opendatahub.io/dashboard: 'true'
spec:
  predictor:
    maxReplicas: 1
    minReplicas: 1
    model:
      args:
        - '--trust-remote-code'
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
      storageUri: registry.redhat.io/rhelai1/modelcar-granite-3-1-8b-instruct-fp8-dynamic:1.5
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
    "model": "granite-3-1-8b-instruct-fp8-dynamic",
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
python quantize.py --model_id ibm-granite/granite-3.1-8b-instruct --save_path "output_dir/"
```

```python
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
import os

def main():
    parser = argparse.ArgumentParser(description='Quantize a transformer model to FP8')
    parser.add_argument('--model_id', type=str, required=True,
                        help='The model ID from HuggingFace (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")')
    parser.add_argument('--save_path', type=str, default='.',
                        help='Custom path to save the quantized model. If not provided, will use model_name-FP8-dynamic')
    args = parser.parse_args()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Configure the quantization algorithm and scheme
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )

    # Apply quantization
    oneshot(model=model, recipe=recipe)

    save_path = os.path.join(args.save_path, args.model_id.split("/")[1] + "-FP8-dynamic")
    os.makedirs(save_path, exist_ok=True)

    # Save to disk in compressed-tensors format
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")

if __name__ == "__main__":
    main()
```
</details>

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard), OpenLLM Leaderboard [V2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/) and on [HumanEval](https://github.com/neuralmagic/evalplus), using the following commands:

<details>
<summary>Evaluation Commands</summary>
  
OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/granite-3.1-8b-instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
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
  --model_args pretrained="neuralmagic/granite-3.1-8b-instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks leaderboard \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```

#### HumanEval
##### Generation
```
python3 codegen/generate.py \
  --model neuralmagic/granite-3.1-8b-instruct-FP8-dynamic \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval
```
##### Sanitization
```
python3 evalplus/sanitize.py \
  humaneval/neuralmagic--granite-3.1-8b-instruct-FP8-dynamic_vllm_temp_0.2
```
##### Evaluation
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/neuralmagic--granite-3.1-8b-instruct-FP8-dynamic_vllm_temp_0.2-sanitized
```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>ibm-granite/granite-3.1-8b-instruct</th>
      <th>neuralmagic/granite-3.1-8b-instruct-FP8-dynamic</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>66.81</td>
      <td>66.81</td>
      <td>100.00</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>64.52</td>
      <td>66.64</td>
      <td>103.29</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>84.18</td>
      <td>84.16</td>
      <td>99.98</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>65.52</td>
      <td>65.36</td>
      <td>99.76</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>60.57</td>
      <td>60.52</td>
      <td>99.92</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>80.19</td>
      <td>79.95</td>
      <td>99.70</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>70.30</b></td>
      <td><b>70.57</b></td>
      <td><b>100.39</b></td>
    </tr>
    <!-- OpenLLM Leaderboard V2 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>74.10</td>
      <td>73.62</td>
      <td>99.35</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>53.19</td>
      <td>53.26</td>
      <td>100.13</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>14.77</td>
      <td>16.79</td>
      <td>113.66</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>31.76</td>
      <td>32.58</td>
      <td>102.58</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>46.01</td>
      <td>47.34</td>
      <td>102.89</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>35.81</td>
      <td>35.72</td>
      <td>99.75</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>42.61</b></td>
      <td><b>43.22</b></td>
      <td><b>101.43</b></td>
    </tr>
    <!-- HumanEval -->
    <tr>
      <td rowspan="2"><b>Coding</b></td>
      <td>HumanEval Pass@1</td>
      <td>71.00</td>
      <td>69.90</td>
      <td><b>98.45</b></td>
    </tr>
  </tbody>
</table>



## Inference Performance


This model achieves up to 1.5x speedup in single-stream deployment and up to 1.1x speedup in multi-stream asynchronous deployment on L40 GPUs.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.6.6.post1, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/granite-3.1-8b-instruct-FP8-dynamic --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
```

</details>


### Single-stream performance (measured with vLLM version 0.6.6.post1)
<table>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <th style="text-align: center;" colspan="7" >Latency (s)</th>
  </tr>
  <tr>
    <th>GPU class</th>
    <th>Model</th>
    <th>Speedup</th>
    <th>Code Completion<br>prefill: 256 tokens<br>decode: 1024 tokens</th>
    <th>Docstring Generation<br>prefill: 768 tokens<br>decode: 128 tokens</th>
    <th>Code Fixing<br>prefill: 1024 tokens<br>decode: 1024 tokens</th>
    <th>RAG<br>prefill: 1024 tokens<br>decode: 128 tokens</th>
    <th>Instruction Following<br>prefill: 256 tokens<br>decode: 128 tokens</th>
    <th>Multi-turn Chat<br>prefill: 512 tokens<br>decode: 256 tokens</th>
    <th>Large Summarization<br>prefill: 4096 tokens<br>decode: 512 tokens</th>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >L40</td>
    <td>granite-3.1-8b-instruct</td>
    <td></td>
    <td>25.1</td>
    <td>3.2</td>
    <td>25.3</td>
    <td>3.2</td>
    <td>3.2</td>
    <td>6.3</td>
    <td>13.4</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-FP8-dynamic<br>(this model)</td>
    <td>1.47</td>
    <td>16.8</td>
    <td>2.2</td>
    <td>17.1</td>
    <td>2.2</td>
    <td>2.1</td>
    <td>4.2</td>
    <td>9.3</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w4a16</td>
    <td>2.72</td>
    <td>8.9</td>
    <td>1.2</td>
    <td>9.2</td>
    <td>1.2</td>
    <td>1.1</td>
    <td>2.3</td>
    <td>5.3</td>
  </tr>
</table>


### Multi-stream asynchronous performance (measured with vLLM version 0.6.6.post1)
<table>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <th style="text-align: center;" colspan="7" >Maximum Throughput (Queries per Second)</th>
  </tr>
  <tr>
    <th>GPU class</th>
    <th>Model</th>
    <th>Speedup</th>
    <th>Code Completion<br>prefill: 256 tokens<br>decode: 1024 tokens</th>
    <th>Docstring Generation<br>prefill: 768 tokens<br>decode: 128 tokens</th>
    <th>Code Fixing<br>prefill: 1024 tokens<br>decode: 1024 tokens</th>
    <th>RAG<br>prefill: 1024 tokens<br>decode: 128 tokens</th>
    <th>Instruction Following<br>prefill: 256 tokens<br>decode: 128 tokens</th>
    <th>Multi-turn Chat<br>prefill: 512 tokens<br>decode: 256 tokens</th>
    <th>Large Summarization<br>prefill: 4096 tokens<br>decode: 512 tokens</th>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >L40</td>
    <td>granite-3.1-8b-instruct</td>
    <td></td>
    <td>1.4</td>
    <td>7.8</td>
    <td>1.1</td>
    <td>6.2</td>
    <td>15.5</td>
    <td>6.0</td>
    <td>0.7</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-FP8-dynamic<br>(this model)</td>
    <td>1.12</td>
    <td>2.1</td>
    <td>7.4</td>
    <td>1.3</td>
    <td>5.9</td>
    <td>15.3</td>
    <td>6.9</td>
    <td>0.8</td>
  </tr>
  <tr>
    <td>granite-3.1-2b-instruct-quantized.w4a16</td>
    <td>1.29</td>
    <td>2.4</td>
    <td>8.9</td>
    <td>1.4</td>
    <td>7.1</td>
    <td>17.8</td>
    <td>7.8</td>
    <td>1.0</td>
  </tr>
</table>
