---
license: apache-2.0
license_name: apache-2.0
pipeline_tag: text-generation
name: RedHatAI/Qwen3-Coder-Next-NVFP4
description: This model was obtained by quantizing the weights and activations of Qwen/Qwen3-Coder-Next to FP4 data type.
readme: https://huggingface.co/RedHatAI/Qwen3-Coder-Next-NVFP4/blob/main/README.md
license_link: https://huggingface.co/RedHatAI/Qwen3-Coder-Next-NVFP4/blob/main/LICENSE
provider: Alibaba Cloud
validated_on:
  - RHOAI 3.4 EA1
  - RHAIIS 3.4 EA1
  - vLLM 0.14.1
tags:
- NVFP4
- quantized
- llm-compressor
- compressed-tensors
- red hat
base_model:
- Qwen/Qwen3-Coder-Next
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Qwen3-Coder-Next-NVFP4
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

## Model Overview
- **Model Architecture:** Qwen3NextForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat
- **ModelCar Storage URI:** oci://registry.redhat.io/rhai/modelcar-qwen3-coder-next-nvfp4:3.0
- **Validated on vLLM:** 0.14.1
- **Validated on RHAIIS:** 3.4 EA1
- **Validated on RHOAI:** 3.4 EA1

Quantized version of [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) to FP4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve inference-optimization/Qwen3-Coder-Next-NVFP4 --port 8000 --tensor-parallel-size 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder
 
```

2. Send requests to the server:

```python
# Your tool implementation
def square_the_number(num: float) -> dict:
    return num ** 2

# Define Tools
tools=[
    {
        "type":"function",
        "function":{
            "name": "square_the_number",
            "description": "output the square of the number.",
            "parameters": {
                "type": "object",
                "required": ["input_num"],
                "properties": {
                    'input_num': {
                        'type': 'number', 
                        'description': 'input_num is a number that will be squared'
                        }
                },
            }
        }
    }
]

from openai import OpenAI
# Define LLM
client = OpenAI(
    # Use a custom endpoint compatible with OpenAI API
    base_url='http://localhost:8000/v1',  # api_base
    api_key="EMPTY"
)
 
messages = [{'role': 'user', 'content': 'square the number 1024'}]

completion = client.chat.completions.create(
    messages=messages,
    model="RedHatAI/Qwen3-Coder-Next-NVFP4",
    max_tokens=65536,
    tools=tools,
)

print(completion.choices[0])
```


## Creation

This model was quantized using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as shown below.

<details>
  <summary>Creation details</summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.offload import dispatch_model

MODEL_ID = "Qwen/Qwen3-Coder-Next"

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples
NUM_CALIBRATION_SAMPLES = 20
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)



recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    weight_observer="mse",
    ignore= ['re:.*lm_head', 're:.*mlp.gate$', 're:.*mlp.shared_expert_gate$', 're:.*linear_attn.*'],
)


oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
)


print("\n\n")
print("========== SAMPLE GENERATION ==============")

dispatch_model(model)

input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```
</details>


## Evaluation


The model was evaluated on the OpenLLM leaderboard task, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>

  **Coding Benchmarks **
  
  **SWE-Bench**
  ```
  python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path preds.json \
    --run_id validate-preds
  ```

</details>


## Accuracy

| Category | Metric | Qwen3-Coder-Next | Qwen3-Coder-Next-NVFP4 | Recovery (%) |
|----------|--------|-------------|-------------------|--------------|
| SWE-Bench | Lite | 49.33 | 52 | 105.4 |