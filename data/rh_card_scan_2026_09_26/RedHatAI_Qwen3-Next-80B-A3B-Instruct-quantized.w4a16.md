---
library_name: transformers
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16
description: Latest Qwen3 architecture with hybrid attention, high-sparsity MoE, stability optimizations and multi-token prediction for great reasoning performance. 
readme: https://huggingface.co/RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16/main/README.md
license_link: https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/LICENSE
provider: Alibaba Cloud
validated_on:
  - RHOAI 3.3
  - RHAIIS 3.3
  - vLLM 0.13.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-Next-80B-A3B-Instruct
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Qwen3-Next-80B-A3B-Instruct-quantized.w4a16
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
  - **Weight quantization:** INT4
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)
- **ModelCar Storage URI**: oci://registry.redhat.io/rhai/modelcar-qwen3-next-80b-a3b-instruct-quantized-w4a16:3.0
- **Validated on vLLM:** 0.13.0
- **Validated on RHAIIS:** 3.3
- **Validated on RHOAI:** 3.3

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-group scheme, with group size 128.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16"
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
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=15000,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks openllm \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=15000,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks mgsm \
    --apply_chat_template\
    --batch_size auto
  ```

  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16",dtype=auto,gpu_memory_utilization=0.5,max_model_len=15000,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks leaderboard \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16
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

| Category | Metric | Qwen/Qwen3-Next-80B-A3B-Instruct | RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16 | Recovery (%) |
|----------|--------|----------------------------------|------------------------------------------------------|--------------|
| OpenLLM V1 | ARC-Challenge (Acc-Norm, 25-shot) | 73.29 | 72.70 | 99.19 |
|  | GSM8K (Strict-Match, 5-shot) | 81.58 | 82.18 | 100.74 |
|  | HellaSwag (Acc-Norm, 10-shot) | 63.90 | 63.64 | 99.59 |
|  | MMLU (Acc, 5-shot) | 85.56 | 85.03 | 99.38 |
|  | TruthfulQA (MC2, 0-shot) | 60.70 | 60.63 | 99.88 |
|  | Winogrande (Acc, 5-shot) | 78.30 | 78.37 | 100.09 |
| | **Average Score** | **73.89** | **73.76** | **99.82** |
| OpenLLM V2 | IFEval (Inst Level Strict Acc, 0-shot) | 77.46 | 80.70 | 104.18 |
|  | BBH (Acc-Norm, 3-shot) | 67.78 | 67.33 | 99.34 |
|  | Math-Hard (Exact-Match, 4-shot) | 56.04 | 55.36 | 98.79 |
|  | GPQA (Acc-Norm, 0-shot) | 28.61 | 28.61 | 100.00 |
|  | MUSR (Acc-Norm, 0-shot) | 39.68 | 40.08 | 101.01 |
|  | MMLU-Pro (Acc, 5-shot) | 76.35 | 75.48 | 98.86 |
| | **Average Score** | **57.65** | **57.93** | **100.49** |