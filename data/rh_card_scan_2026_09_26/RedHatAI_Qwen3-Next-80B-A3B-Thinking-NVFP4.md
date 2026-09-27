---
license: apache-2.0
pipeline_tag: text-generation
tags:
- fp8
- quantized
- llm-compressor
- compressed-tensors
- red hat
base_model:
- Qwen/Qwen3-Next-80B-A3B-Thinking
---

# Qwen3-Next-80B-A3B-Thinking-NVFP4

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

Quantized version of [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Qwen3-Next-80B-A3B-Thinking-NVFP4 --tensor_parallel_size 2
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

model = "RedHatAI/Qwen3-Next-80B-A3B-Thinking-NVFP4"

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

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Thinking"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


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

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

# Apply quantization.
# MoE calibration is now handled automatically by the pipeline.
# We set `moe_calibrate_all_experts` to True to ensure all experts receive
# calibration data. This temporarily updates the model definition to use
# `CalibrationQwen3NextSparseMoeBlock` (from `llmcompressor.modeling.qwen3_next_moe`)
# which replaces the original `Qwen3NextSparseMoeBlock` class.
# This updates how the forward pass is handled in the MoE block during calibration.
# Feel free to update the definition under
# llm-compressor/src/llmcompressor/modeling/qwen3_next_moe.py to play around with
# this behavior and evaluate its impact on quantization performance.
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
dispatch_for_generation(model)
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
  
  **Openllm V1**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Thinking-NVFP4",dtype=auto,add_bos_token=True,max_model_len=16384,tensor_parallel_size=2,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Openllm V2**  
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Thinking-NVFP4",dtype=auto,add_bos_token=False,max_model_len=16384,tensor_parallel_size=2,gpu_memory_utilization=0.7,disable_log_stats=True,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks leaderboard \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Coding Benchmarks**

  ```
  evalplus.evaluate --model "RedHatAI/Qwen3-Next-80B-A3B-Thinking-NVFP4" \
                    --dataset "humaneval" \
                    --backend vllm \
                    --tp 2 \
                    --greedy

  evalplus.evaluate --model "RedHatAI/Qwen3-Next-80B-A3B-Thinking-NVFP4" \
                  --dataset "mbpp" \
                  --backend vllm \
                  --tp 2 \
                  --greedy

  ```

</details>
