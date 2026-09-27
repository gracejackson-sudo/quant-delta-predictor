---
license: apache-2.0
pipeline_tag: text-generation
tags:
- int8
- quantized
- llm-compressor
- compressed-tensors
- red hat
base_model:
- Qwen/Qwen3-Next-80B-A3B-Instruct
---


# Qwen3-Next-80B-A3B-Instruct-quantized.w8a8

## Model Overview
- **Model Architecture:** Qwen3NextForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
  - **Activation quantization:** INT8
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) to INT8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```bash
vllm serve RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8 --tensor_parallel_size 2
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

model = "RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8"

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

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

# NOTE: Requires a minimum of transformers 4.57.0

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Instruct"


# Select calibration dataset.
DATASET_ID = "garage-bAInd/Open-Platypus"
DATASET_SPLIT = "train"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 8192

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to int8 with per channel via ptq
#   * quantize the activations to int8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="W8A8", ignore=[
        "lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

# Load calibration dataset.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    }

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=20)
print(tokenizer.decode(output[0]))
print("==========================================")

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-quantized.w8a8"
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
```
</details>


## Evaluation

The model was evaluated on the AIME25, GPQA Diamond and Mathh 500 benchmarks using [lighteval](https://github.com/huggingface/lighteval), and on MMLU-Pro, IFEval and GSM8k using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
In all cases [vLLM](https://vllm.ai) is used as the backend.
All results were averaged over 6 repetitions with different random seeds.


<details>
  <summary>Evaluation commands</summary>

### Start vLLM server
```bash
vllm serve RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8 --tensor_parallel_size 2
```

### lm-evaluation-harness
```bash
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,timeout=600,tokenizer_backend=None" \
  --apply_chat_template \
  --num_fewshot 5 \
  --fewshot_as_multiturn \
  --output_path mmlu_pro_qwen3_next_w8a8 \
  --gen_kwargs "do_sample=True,temperature=0.7,top_p=0.8,top_k=20,max_gen_toks=16000"
```

```bash
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,timeout=600,tokenizer_backend=None" \
  --apply_chat_template \
  --output_path ifeval_qwen3_next_w8a8 \
  --gen_kwargs "do_sample=True,temperature=0.7,top_p=0.8,top_k=20,max_gen_toks=16000"
```

```bash
lm_eval --model local-chat-completions \
  --tasks gsm8k \
  --model_args "model=RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,timeout=600,tokenizer_backend=None" \
  --apply_chat_template \
  --num_fewshot 5 \
  --fewshot_as_multiturn \
  --output_path gsm8k_qwen3_next_w8a8 \
  --gen_kwargs "do_sample=True,temperature=0.7,top_p=0.8,top_k=20,max_gen_toks=16000"
```

### lighteval
litellm_config.yaml
```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/RedHatAI/Phi-4-reasoning-FP8-dynamic"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 600
  concurrent_requests: 128
  generation_parameters:
    temperature: 0.7
    top_k: 20
    top_p: 0.8
    max_new_tokens: 16000
```

```bash
lighteval endpoint litellm litellm_config.yaml \
    gpqa:diamond|0,math_500|0,aime25|0 \
    --output-dir qwen3_next_w8a8 \
    --save-details
```
</details>

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Qwen3-Next-80B-A3B-Instruct</strong>
   </td>
   <td><strong>Qwen3-Next-80B-A3B-Instruct-quantized.w8a8<br>(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>AIME25
   </td>
   <td>62.78
   </td>
   <td>65.00
   </td>
   <td>103.5%
   </td>
  </tr>
  <tr>
   <td>GPQA Diamond
   </td>
   <td>74.58
   </td>
   <td>75.17
   </td>
   <td>100.8%
   </td>
  </tr>
  <tr>
   <td>Math 500
   </td>
   <td>89.73
   </td>
   <td>90.57
   </td>
   <td>100.9%
   </td>
  </tr>
  <tr>
   <td>MMLU-Pro
   </td>
   <td>78.62
   </td>
   <td>78.85
   </td>
   <td>100.3%
   </td>
  </tr>
  <tr>
   <td>IFEval
   </td>
   <td>91.45
   </td>
   <td>91.51
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>GSM8k
   </td>
   <td>69.71
   </td>
   <td>69.74
   </td>
   <td>100.0%
   </td>
  </tr>
</table>