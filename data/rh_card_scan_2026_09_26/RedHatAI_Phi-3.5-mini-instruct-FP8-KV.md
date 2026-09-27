---
tags:
- fp8
- vllm
pipeline_tag: text-generation
license: mit
license_link: https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/LICENSE
base_model: microsoft/Phi-3.5-mini-instruct
---

# Phi-3.5-mini-instruct-FP8-KV

## Model Overview
- **Model Architecture:** Phi-3.5
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in English. Similarly to [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 8/11/2024
- **Version:** 1.1
- **License(s):** [mit](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/LICENSE)
- **Model Developers:** Neural Magic

Quantized version of [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct), with the new configuration files.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) to FP8 data type, ready for inference with vLLM >= 0.5.1.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-tensor quantization is applied, in which a single linear scaling maps the FP8 representations of the quantized weights and activations.
[AutoFP8](https://github.com/neuralmagic/AutoFP8) is used for quantization with 512 sequences of UltraChat.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Phi-3.5-mini-instruct-FP8-KV"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you? Remember to respond in pirate speak!"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_id, kv_cache_dtype="fp8")

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [LLM Compressor with calibration samples from UltraChat](https://github.com/vllm-project/llm-compressor/blob/sa/big_model_support/examples/big_model_offloading/big_model_w8a8_calibrate.py), as presented in the code snipet below.

```python
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

# Select model and load it.
# Phi-3.5 is a special case for KV cache quantization because it has
# fused QKV linear layers.
MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
model = SparseAutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


def process_and_tokenize(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp8 with per-tensor scales
#   * quantize the activations to fp8 with per-tensor scales
#   * quantize the kv cache to fp8 with per-tensor scales
recipe = """
quant_stage:
    quant_modifiers:
        QuantizationModifier:
            ignore: ["lm_head"]
            config_groups:
                group_0:
                    weights:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: tensor
                        dynamic: false
                        symmetric: true
                    targets: ["Linear"]
            kv_cache_scheme:
                num_bits: 8
                type: float
                strategy: tensor
                dynamic: false
                symmetric: true
"""

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
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-KV"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Evaluation

The model was evaluated on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) leaderboard tasks (version 1) with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/383bbd54bc621086e05aa1b030d8d4d5635b25e6) (commit 383bbd54bc621086e05aa1b030d8d4d5635b25e6) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Phi-3.5-mini-instruct-FP8-KV",kv_cache_dtype="fp8",gpu_memory_utilization=0.4,add_bos_token=True,max_model_len=4096 \
  --tasks openllm \
  --batch_size auto
```

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Phi-3.5-mini-instruct</strong>
   </td>
   <td><strong>Phi-3.5-mini-instruct-FP8-KV(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>68.81
   </td>
   <td>68.56
   </td>
   <td>99.64%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (25-shot, acc_norm)
   </td>
   <td>64.68
   </td>
   <td>64.51
   </td>
   <td>99.74%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (5-shot, strict-match)
   </td>
   <td>78.24
   </td>
   <td>77.26
   </td>
   <td>98.75%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot, acc_norm)
   </td>
   <td>79.03
   </td>
   <td>78.88
   </td>
   <td>99.81%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot, acc)
   </td>
   <td>73.40
   </td>
   <td>73.80
   </td>
   <td>100.5%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>56.39
   </td>
   <td>56.95
   </td>
   <td>100.9%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>70.09</strong>
   </td>
   <td><strong>70.00</strong>
   </td>
   <td><strong>99.89%</strong>
   </td>
  </tr>
</table>
