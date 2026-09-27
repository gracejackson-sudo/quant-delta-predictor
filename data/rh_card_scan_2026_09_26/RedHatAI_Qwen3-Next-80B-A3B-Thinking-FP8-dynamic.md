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


# Qwen3-Next-80B-A3B-Thinking-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen3NextForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 
- **Version:** 1.0
- **Model Developers:**: Red Hat

Quantized version of [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking).

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Qwen3-Next-80B-A3B-Thinking-FP8-dynamic --tensor_parallel_size 2
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

model = "RedHatAI/Qwen3-Next-80B-A3B-Thinking-FP8-dynamic"

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


# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per channel via ptq
#   * quantize the activations to fp8 with dynamic per token
recipe = QuantizationModifier(
    targets="Linear", scheme="FP8_DYNAMIC", ignore=[
        "lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

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
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-FP8-dynamic"
model.save_pretrained(SAVE_DIR)
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
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Thinking-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=16384,tensor_parallel_size=2,gpu_memory_utilization=0.9,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks openllm \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Openllm V2**  
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Thinking-FP8-dynamic",dtype=auto,add_bos_token=False,max_model_len=16384,tensor_parallel_size=2,gpu_memory_utilization=0.7,disable_log_stats=True,enable_chunked_prefill=True,trust_remote_code=True \
    --tasks leaderboard \
    --apply_chat_template \
    --fewshot_as_multiturn \
    --write_out \
    --batch_size auto \
    --show_config
  ```


  **Coding Benchmarks**

  ```
  evalplus.evaluate --model "RedHatAI/Qwen3-Next-80B-A3B-Thinking-FP8-dynamic" \
                    --dataset "humaneval" \
                    --backend vllm \
                    --tp 2 \
                    --greedy

  evalplus.evaluate --model "RedHatAI/Qwen3-Next-80B-A3B-Thinking-FP8-dynamic" \
                  --dataset "mbpp" \
                  --backend vllm \
                  --tp 2 \
                  --greedy

  ```

</details>



## Accuracy

| Category | Metric | Qwen/Qwen3-Next-80B-A3B-Thinking | RedHatAI/Qwen3-Next-80B-A3B-Thinking-FP8-dynamic | Recovery (%) |
|----------|--------|----------------------------------|---------------------------------------------------|--------------|
| OpenLLM V1 | ARC-Challenge (Acc-Norm, 25-shot) | 70.14 | 70.05 | 99.87 |
|  | GSM8K (Strict-Match, 5-shot) | 84.61 | 84.76 | 100.18 |
|  | HellaSwag (Acc-Norm, 10-shot) | 62.19 | 62.06 | 99.79 |
|  | MMLU (Acc, 5-shot) | 84.95 | 85.05 | 100.12 |
|  | TruthfulQA (MC2, 0-shot) | 59.29 | 59.51 | 100.37 |
|  | Winogrande (Acc, 5-shot) | 77.98 | 77.90 | 99.90 |
| | **Average Score** | **73.19** | **73.22** | **100.04** |
| OpenLLM V2 | IFEval (Inst Level Strict Acc, 0-shot) | 44.84 | 42.45 | 94.67 |
|  | BBH (Acc-Norm, 3-shot) | 29.73 | 30.74 | 103.40 |
|  | Math-Hard (Exact-Match, 4-shot) | 18.35 | 16.84 | 91.77 |
|  | GPQA (Acc-Norm, 0-shot) | 26.34 | 26.59 | 100.95 |
|  | MUSR (Acc-Norm, 0-shot) | 42.33 | 41.01 | 96.88 |
|  | MMLU-Pro (Acc, 5-shot) | 72.70 | 72.77 | 100.10 |
| | **Average Score** | **39.05** | **38.40** | **98.34** |
| Reasoning | AIME25 (pass@1, n=8) | 60.00 | 50.00 | 83.33 |
|  | MATH-500 (pass@1, n=8) | 94.00 | 86.80 | 92.34 |
|  | GPQA-Diamond (pass@1, n=8) | 73.74 | 74.75 | 101.37 |
| | **Average Score** | **75.91** | **70.52** | **92.90** |