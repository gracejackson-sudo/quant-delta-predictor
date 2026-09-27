---
language:
- en
base_model: 
- ibm-granite/granite-3.1-8b-instruct
pipeline_tag: text-generation
tags:
- w8a8
- int8
- vllm
- conversational
- text-generation-inference
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/granite-3.1-8b-instruct-quantized.w8a8
description: This model was obtained by quantizing the weights and activations of ibm-granite/granite-3.1-8b-instruct to INT8 data type.
readme: https://huggingface.co/RedHatAI/granite-3.1-8b-instruct-quantized.w8a8/main/README.md
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
  granite-3.1-8b-instruct-quantized.w8a8
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
  - **Weight quantization:** INT8
  - **Activation quantization:** INT8
- **Release Date:** 1/8/2025
- **Version:** 1.0
- **Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5
- **Model Developers:** Neural Magic

Quantized version of [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct).
It achieves an average score of 70.26 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 70.30.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [ibm-granite/granite-3.1-8b-instruct](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) to INT8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%. Only the weights and activations of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 1
model_name = "neuralmagic/granite-3.1-8b-instruct-quantized.w8a8"
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
--tensor-parallel-size 8 \
--max-model-len 32768  \
--enforce-eager --model RedHatAI/granite-3.1-8b-instruct-quantized.w8a8
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/granite-3-1-8b-instruct-quantized-w8a8:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/granite-3-1-8b-instruct-quantized-w8a8 -- --trust-remote-code
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/granite-3-1-8b-instruct-quantized-w8a8
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
    openshift.io/display-name: granite-3-1-8b-instruct-quantized-w8a8 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: granite-3-1-8b-instruct-quantized-w8a8          # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-granite-3-1-8b-instruct-quantized-w8a8:1.5
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
    "model": "granite-3-1-8b-instruct-quantized-w8a8",
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
python quantize.py --model_path ibm-granite/granite-3.1-8b-instruct --quant_path "output_dir/granite-3.1-8b-instruct-quantized.w8a8" --calib_size 3072 --dampening_frac 0.1 --observer mse
```


```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot, apply
import argparse
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--quant_path', type=str)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.1) 
parser.add_argument('--observer', type=str, default="minmax")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype="auto",
    use_cache=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

NUM_CALIBRATION_SAMPLES = args.calib_size
DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": example["text"]}

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

ignore=["lm_head"]
mappings=[
    [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
    [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
    [["re:.*down_proj"], "re:.*up_proj"]
]

recipe = [
    SmoothQuantModifier(smoothing_strength=0.8, ignore=ignore, mappings=mappings),
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W8A8",
        dampening_frac=args.dampening_frac,
        observer=args.observer,
    )
]
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=args.calib_size,
    max_seq_length=8196,
)

# Save to disk compressed.
model.save_pretrained(quant_path, save_compressed=True)
tokenizer.save_pretrained(quant_path)
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
  --model_args pretrained="neuralmagic/granite-3.1-8b-instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
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
  --model_args pretrained="neuralmagic/granite-3.1-8b-instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
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
  --model neuralmagic/granite-3.1-8b-instruct-quantized.w8a8 \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval
```
##### Sanitization
```
python3 evalplus/sanitize.py \
  humaneval/neuralmagic--granite-3.1-8b-instruct-quantized.w8a8_vllm_temp_0.2
```
##### Evaluation
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/neuralmagic--granite-3.1-8b-instruct-quantized.w8a8_vllm_temp_0.2-sanitized
```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>ibm-granite/granite-3.1-8b-instruct</th>
      <th>neuralmagic/granite-3.1-8b-instruct-quantized.w8a8</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>66.81</td>
      <td>67.06</td>
      <td>100.37</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>64.52</td>
      <td>65.66</td>
      <td>101.77</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>84.18</td>
      <td>83.93</td>
      <td>99.70</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>65.52</td>
      <td>65.03</td>
      <td>99.25</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>60.57</td>
      <td>60.02</td>
      <td>99.09</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>80.19</td>
      <td>79.87</td>
      <td>99.60</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>70.30</b></td>
      <td><b>70.26</b></td>
      <td><b>99.95</b></td>
    </tr>
    <!-- OpenLLM Leaderboard V2 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>74.01</td>
      <td>73.50</td>
      <td>99.31</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>53.19</td>
      <td>52.59</td>
      <td>98.87</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>14.77</td>
      <td>15.73</td>
      <td>106.50</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>31.76</td>
      <td>30.62</td>
      <td>96.40</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>46.01</td>
      <td>44.30</td>
      <td>96.28</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>35.81</td>
      <td>35.41</td>
      <td>98.88</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>42.61</b></td>
      <td><b>42.03</b></td>
      <td><b>98.64</b></td>
    </tr>
    <!-- HumanEval -->
    <tr>
      <td rowspan="2"><b>Coding</b></td>
      <td>HumanEval Pass@1</td>
      <td>71.00</td>
      <td>70.50</td>
      <td><b>99.30</b></td>
    </tr>
  </tbody>
</table>



## Inference Performance


This model achieves up to 1.6x speedup in single-stream deployment and up to 1.7x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.6.6.post1, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/granite-3.1-8b-instruct-quantized.w8a8 --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
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
    <td style="vertical-align: middle;" rowspan="3" >A5000</td>
    <td>granite-3.1-8b-instruct</td>
    <td></td>
    <td>28.3</td>
    <td>3.7</td>
    <td>28.8</td>
    <td>3.8</td>
    <td>3.6</td>
    <td>7.2</td>
    <td>15.7</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w8a8<br>(this model)</td>
    <td>1.60</td>
    <td>17.7</td>
    <td>2.3</td>
    <td>18.0</td>
    <td>2.4</td>
    <td>2.2</td>
    <td>4.5</td>
    <td>10.0</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w4a16</td>
    <td>2.61</td>
    <td>10.3</td>
    <td>1.5</td>
    <td>10.7</td>
    <td>1.5</td>
    <td>1.3</td>
    <td>2.7</td>
    <td>6.6</td>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >A6000</td>
    <td>granite-3.1-8b-instruct</td>
    <td></td>
    <td>25.8</td>
    <td>3.4</td>
    <td>26.2</td>
    <td>3.4</td>
    <td>3.3</td>
    <td>6.5</td>
    <td>14.2</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w8a8<br>(this model)</td>
    <td>1.50</td>
    <td>17.4</td>
    <td>2.3</td>
    <td>16.9</td>
    <td>2.2</td>
    <td>2.2</td>
    <td>4.4</td>
    <td>9.8</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w4a16</td>
    <td>2.48</td>
    <td>10.0</td>
    <td>1.4</td>
    <td>10.4</td>
    <td>1.5</td>
    <td>1.3</td>
    <td>2.5</td>
    <td>6.2</td>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >A100</td>
    <td>granite-3.1-8b-instruct</td>
    <td></td>
    <td>13.6</td>
    <td>1.8</td>
    <td>13.7</td>
    <td>1.8</td>
    <td>1.7</td>
    <td>3.4</td>
    <td>7.3</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w8a8<br>(this model)</td>
    <td>1.31</td>
    <td>10.4</td>
    <td>1.3</td>
    <td>10.5</td>
    <td>1.4</td>
    <td>1.3</td>
    <td>2.6</td>
    <td>5.6</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w4a16</td>
    <td>1.80</td>
    <td>7.3</td>
    <td>1.0</td>
    <td>7.4</td>
    <td>1.0</td>
    <td>0.9</td>
    <td>1.9</td>
    <td>4.3</td>
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
    <td style="vertical-align: middle;" rowspan="3" >A5000</td>
    <td>granite-3.1-8b-instruct</td>
    <td></td>
    <td>0.8</td>
    <td>3.1</td>
    <td>0.4</td>
    <td>2.5</td>
    <td>6.7</td>
    <td>2.7</td>
    <td>0.3</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w8a8<br>(this model)</td>
    <td>1.71</td>
    <td>1.3</td>
    <td>5.2</td>
    <td>0.9</td>
    <td>4.0</td>
    <td>10.5</td>
    <td>4.4</td>
    <td>0.5</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w4a16</td>
    <td>1.46</td>
    <td>1.3</td>
    <td>3.9</td>
    <td>0.8</td>
    <td>2.9</td>
    <td>8.2</td>
    <td>3.6</td>
    <td>0.5</td>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >A6000</td>
    <td>granite-3.1-8b-instruct</td>
    <td></td>
    <td>1.3</td>
    <td>5.1</td>
    <td>0.9</td>
    <td>4.0</td>
    <td>0.3</td>
    <td>4.3</td>
    <td>0.6</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w8a8<br>(this model)</td>
    <td>1.39</td>
    <td>1.8</td>
    <td>7.0</td>
    <td>1.3</td>
    <td>5.6</td>
    <td>14.0</td>
    <td>6.3</td>
    <td>0.8</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w4a16</td>
    <td>1.09</td>
    <td>1.9</td>
    <td>4.8</td>
    <td>1.0</td>
    <td>3.8</td>
    <td>10.0</td>
    <td>5.0</td>
    <td>0.6</td>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >A100</td>
    <td>granite-3.1-8b-instruct</td>
    <td></td>
    <td>3.1</td>
    <td>10.7</td>
    <td>2.1</td>
    <td>8.5</td>
    <td>20.6</td>
    <td>9.6</td>
    <td>1.4</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w8a8<br>(this model)</td>
    <td>1.23</td>
    <td>3.8</td>
    <td>14.2</td>
    <td>2.1</td>
    <td>11.4</td>
    <td>25.9</td>
    <td>12.1</td>
    <td>1.7</td>
  </tr>
  <tr>
    <td>granite-3.1-8b-instruct-quantized.w4a16</td>
    <td>0.96</td>
    <td>3.4</td>
    <td>9.0</td>
    <td>2.6</td>
    <td>7.2</td>
    <td>18.0</td>
    <td>8.8</td>
    <td>1.3</td>
  </tr>
</table>
