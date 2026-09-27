---
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/granite-4.0-h-small-FP8-dynamic
description:  Granite-4.0-H-Small is a 32B parameter long-context instruct model finetuned from Granite-4.0-H-Small-Base using a combination of open source instruction datasets with permissive license and internally collected synthetic datasets.
readme: https://huggingface.co/RedHatAI/granite-4.0-h-small-FP8-dynamic/main/README.md
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
provider: IBM
validated_on:
  - RHOAI 3.3
  - RHAIIS 3.3
  - vLLM 0.13.0
pipeline_tag: text-generation
tags:
- fp8
- quantized
- llm-compressor
- compressed-tensors
- red hat
base_model:
- ibm-granite/granite-4.0-h-small
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Granite-4.0-h-small-FP8-dynamic
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** GraniteMoeHybridForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat
- **ModelCar Storage URI**: oci://registry.redhat.io/rhai/modelcar-granite-4-0-h-small-fp8-dynamic:3.0
- **Validated on vLLM:** 0.13.0
- **Validated on RHAIIS:** 3.3
- **Validated on RHOAI:** 3.3

Quantized version of [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Install vLLM from main:
```
uv pip install -U git+https://github.com/vllm-project/vllm.git \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --no-deps \
  --no-cache


uv pip install compressed-tensors==0.12.3a20251114 --no-cache
uv pip install --upgrade torchvision --break-system-packages --no-cache
uv pip install cloudpickle msgspec zmq blake3 cachetools prometheus_client fastapi openai openai_harmony pybase64 llguidance diskcache xgrammar lm-format-enforcer partial-json-parser cbor2 einops gguf numba --no-cache
```

2. Initialize vLLM server:
```
vllm serve RedHatAI/granite-4.0-h-small-FP8-dynamic --tensor_parallel_size 1
```

3. Send requests to the server:

```python
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/granite-4.0-h-small-FP8-dynamic"

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

## Creation

This model was quantized using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as shown below.


<details>
  <summary>Creation details</summary>

Install specific llm-compression version:
```
uv pip install git+https://github.com/vllm-project/llm-compressor.git@refs/pull/2001/head --no-cache
uv pip install --upgrade torchvision --break-system-packages --no-cache
```


```python
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modeling import replace_modules_for_calibration
from llmcompressor.modeling.granite4 import pack_3d_experts


MODEL_ID = "ibm-granite/granite-4.0-h-small"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = replace_modules_for_calibration(model)

ignore_lay = ["lm_head", "re:.*block_sparse_moe.router"]

recipe = QuantizationModifier(
    targets=["Linear"],
    scheme="FP8_DYNAMIC",
    ignore=ignore_lay,
)

oneshot(model=model, recipe=recipe)

print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer(
    "Describe Large Language Model", return_tensors="pt"
).input_ids.to(model.device)
output = model.generate(input_ids, max_new_tokens=35)
print(tokenizer.decode(output[0]))
print("==========================================")

SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-dynamic"
print(f"Saving to {SAVE_DIR}")

model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
pack_3d_experts(SAVE_DIR)
```
</details>


## Evaluation


The model was evaluated on the OpenLLM leaderboard task, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>

  Install vLLM from main:
```
uv pip install -U git+https://github.com/vllm-project/vllm.git \
  --extra-index-url https://wheels.vllm.ai/nightly \
  --no-deps \
  --no-cache


uv pip install compressed-tensors==0.12.3a20251114 --no-cache
uv pip install --upgrade torchvision --break-system-packages --no-cache
uv pip install cloudpickle msgspec zmq blake3 cachetools prometheus_client fastapi openai openai_harmony pybase64 llguidance diskcache xgrammar lm-format-enforcer partial-json-parser cbor2 einops gguf numba --no-cache
```
  
  **Openllm V1**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/granite-4.0-h-small-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=16384,tensor_parallel_size=1,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Openllm V2**  
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/granite-4.0-h-small-FP8-dynamic",dtype=auto,add_bos_token=False,max_model_len=16384,tensor_parallel_size=1,gpu_memory_utilization=0.7,disable_log_stats=True,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks leaderboard \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Coding Benchmarks**

  ```
  evalplus.evaluate --model "RedHatAI/granite-4.0-h-small-FP8-dynamic" \
                    --dataset "humaneval" \
                    --backend vllm \
                    --tp 1 \
                    --greedy

  evalplus.evaluate --model "RedHatAI/granite-4.0-h-small-FP8-dynamic" \
                  --dataset "mbpp" \
                  --backend vllm \
                  --tp 1 \
                  --greedy

  ```

</details>



### Accuracy Comparison
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>ibm-granite/granite-4.0-h-small</th>
      <th>ibm-granite/granite-4.0-h-small-FP8</th>
      <th>RedHatAI/granite-4.0-h-small-FP8-block</th>
      <th>RedHatAI/granite-4.0-h-small-FP8-dynamic</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM Leaderboard V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>72.27</td>
      <td>72.10 (99.76%)</td>
      <td>72.27 (100.00%)</td>
      <td>72.10 (99.76%)</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>85.22</td>
      <td>85.29 (100.09%)</td>
      <td>85.52 (100.36%)</td>
      <td>84.84 (99.56%)</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>86.08</td>
      <td>85.88 (99.77%)</td>
      <td>85.96 (99.86%)</td>
      <td>85.88 (99.77%)</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>77.15</td>
      <td>77.18 (100.03%)</td>
      <td>77.23 (100.09%)</td>
      <td>77.18 (100.03%)</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>57.64</td>
      <td>57.63 (99.99%)</td>
      <td>57.94 (100.52%)</td>
      <td>57.63 (100.00%)</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>81.37</td>
      <td>81.45 (100.10%)</td>
      <td>80.82 (99.32%)</td>
      <td>81.45 (100.10%)</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>76.62</b></td>
      <td><b>76.59 (99.96%)</b></td>
      <td><b>76.62 (100.00%)</b></td>
      <td><b>76.51 (99.86%)</b></td>
    </tr>
    <!-- OpenLLM Leaderboard V2 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>87.53</td>
      <td>87.17 (99.59%)</td>
      <td>86.69 (99.04%)</td>
      <td>87.41 (99.86%)</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>61.52</td>
      <td>61.31 (99.66%)</td>
      <td>61.40 (99.80%)</td>
      <td>61.19 (99.46%)</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>46.22</td>
      <td>43.73 (94.61%)</td>
      <td>43.88 (94.93%)</td>
      <td>41.77 (90.36%)</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>35.23</td>
      <td>34.98 (99.29%)</td>
      <td>34.23 (97.14%)</td>
      <td>34.23 (97.14%)</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>46.69</td>
      <td>46.56 (99.72%)</td>
      <td>45.77 (98.02%)</td>
      <td>45.77 (98.02%)</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>47.99</td>
      <td>47.63 (99.26%)</td>
      <td>47.93 (99.88%)</td>
      <td>47.58 (99.15%)</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>54.20</b></td>
      <td><b>53.56 (98.82%)</b></td>
      <td><b>53.32 (98.38%)</b></td>
      <td><b>52.99 (97.77%)</b></td>
    </tr>
  </tbody>
</table>
