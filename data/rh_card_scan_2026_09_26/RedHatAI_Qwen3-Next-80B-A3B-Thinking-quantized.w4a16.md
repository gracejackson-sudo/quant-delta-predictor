---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-Next-80B-A3B-Thinking
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
---

# Qwen3-Next-80B-A3B-Thinking-quantized.w4a16

## Model Overview
- **Model Architecture:** Qwen3NextForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen/Qwen3-Next-80B-A3B-Thinking](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-group scheme, with group size 128.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-Next-80B-A3B-Thinking-quantized.w4a16"
number_gpus = 1
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=256)

messages = [
    {"role": "user", "content": prompt}
]

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


  ```python
  from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from llmcompressor.modifiers.quantization import GPTQModifier

# NOTE: Requires a minimum of transformers 4.57.0

MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Thinking"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
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

# Configure the quantization algorithm to run.
#   * quantize the weights to 4 bit with GPTQ with a group size 128
recipe = GPTQModifier(targets="Linear", scheme="W4A16", 
    ignore=[
        "lm_head",
        "re:.*mlp.gate$",
        "re:.*mlp.shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
sample = tokenizer("Describe Large Language Model", return_tensors="pt")
sample = {key: value.to(model.device) for key, value in sample.items()}
output = model.generate(**sample, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
  ```
</details>
 


## Evaluation

The model was evaluated on the OpenLLM leaderboard tasks versions 2, using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and on reasoning tasks using [lighteval](https://github.com/neuralmagic/lighteval/tree/reasoning).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>

  **lm-evaluation-harness**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Thinking-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=15000,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks openllm \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Thinking-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=15000,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks mgsm \
    --apply_chat_template\
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Thinking-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=15000,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks leaderboard \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-Next-80B-A3B-Thinking-quantized.w4a16
    dtype: auto
    gpu_memory_utilization: 0.9
    max_model_length: 40960
    generation_parameters:
      temperature: 0.6
      top_k: 20
      min_p: 0.0
      top_p: 0.95
      max_new_tokens: 32000
  ```


  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|aime25|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|math_500|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks lighteval|gpqa:diamond|0|0 \
    --use_chat_template = true
  ```

  ```
  lighteval vllm \
    --model_args lighteval_model_arguments.yaml \
    --tasks extended|lcb:codegeneration \
    --use_chat_template = true
  ```

</details>


## Accuracy

| Category | Metric | Qwen/Qwen3-Next-80B-A3B-Thinking | RedHatAI/Qwen3-Next-80B-A3B-Thinking-quantized.w4a16 | Recovery (%) |
|----------|--------|----------------------------------|-------------------------------------------------------|--------------|
| OpenLLM V1 | ARC-Challenge (Acc-Norm, 25-shot) | 70.14 | 69.20 | 98.66 |
|  | GSM8K (Strict-Match, 5-shot) | 84.61 | 83.78 | 99.02 |
|  | HellaSwag (Acc-Norm, 10-shot) | 62.19 | 61.90 | 99.53 |
|  | MMLU (Acc, 5-shot) | 84.95 | 84.56 | 99.54 |
|  | TruthfulQA (MC2, 0-shot) | 59.29 | 58.97 | 99.46 |
|  | Winogrande (Acc, 5-shot) | 77.98 | 79.16 | 101.51 |
| | **Average Score** | **73.19** | **72.93** | **99.64** |
| OpenLLM V2 | IFEval (Inst Level Strict Acc, 0-shot) | 44.84 | 46.04 | 102.68 |
|  | BBH (Acc-Norm, 3-shot) | 29.73 | 30.05 | 101.08 |
|  | Math-Hard (Exact-Match, 4-shot) | 18.35 | 16.92 | 92.21 |
|  | GPQA (Acc-Norm, 0-shot) | 26.34 | 26.59 | 100.95 |
|  | MUSR (Acc-Norm, 0-shot) | 42.33 | 42.06 | 99.36 |
|  | MMLU-Pro (Acc, 5-shot) | 72.70 | 71.43 | 98.25 |
| | **Average Score** | **39.05** | **38.85** | **99.49** |
| Reasoning | AIME25 (pass@1, n=8) | 60.00 | 50.00 | 83.33 |
|  | MATH-500 (pass@1, n=8) | 94.00 | 87.20 | 92.77 |
|  | GPQA-Diamond (pass@1, n=8) | 73.74 | 73.74 | 100.00 |
| | **Average Score** | **75.91** | **70.31** | **92.62** |


