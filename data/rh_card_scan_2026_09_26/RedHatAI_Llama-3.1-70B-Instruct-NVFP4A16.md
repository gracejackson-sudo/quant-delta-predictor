---
tags:
- fp4
- vllm
language:
- en
- de
- fr
- it
- pt
- hi
- es
- th
pipeline_tag: text-generation
license: llama3.1
base_model: meta-llama/Meta-Llama-3.1-70B-Instruct
---

# Meta-Llama-3.1-70B-Instruct-NVFP4A16

## Model Overview
- **Model Architecture:** Meta-Llama-3.1
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP16
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. Similarly to [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 6/25/2025
- **Version:** 1.0
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** RedHatAI

This model is a quantized version of [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights of [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) to FP4 data type, ready for inference with vLLM>=0.9.1
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 25%.

Only the weights of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16"
number_gpus = 2

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [LLM Compressor with calibration samples from UltraChat](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/llama3_example.py), as presented in the code snipet below.

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model.
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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

# Configure the quantization algorithm and scheme.
# In this case, we:
#   * quantize the weights to fp4 with per group 16 via ptq
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4A16", ignore=["lm_head"])

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4A16"

# Apply quantization.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir=SAVE_DIR,
)

print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

```

## Evaluation

This model was evaluated on the well-known OpenLLM v1, OpenLLM v2, HumanEval, and HumanEval_64 benchmarks. All evaluations were conducted using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness).

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Meta-Llama-3.1-70B-Instruct</th>
      <th>RedHatAI/Llama-3.1-70B-Instruct-NVFP4 (this model)</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8"><b>OpenLLM V1</b></td>
      <td>ARC Challenge Llama</td>
      <td>95.02</td>
      <td>94.33</td>
      <td>99.27%</td>
    </tr>
  <tr>
    <td>GSM8K Llama (8-shot, strict-match)</td>
    <td>94.62</td>
    <td>93.48</td>
    <td>98.80%</td>
  </tr>
  <tr>
    <td>MMLU Llama</td>
    <td>84.10</td>
    <td>82.58</td>
    <td>98.19%</td>
  </tr>
  <tr>
    <td>MMLU cot Llama (0-shot)</td>
    <td>86.08</td>
    <td>84.58</td>
    <td>98.26%</td>
  </tr>
  <tr>
    <td>Hellaswag (10-shot)</td>
    <td>78.82</td>
    <td>78.60</td>
    <td>98.26%</td>
  </tr>
  <tr>
    <td>TruthfulQA (0-shot, mc2)</td>
    <td>66.88</td>
    <td>65.42</td>
    <td>97.82%</td>
  </tr>
  <tr>
    <td>Winogrande (5-shot)</td>
    <td>74.27</td>
    <td>73.24</td>
    <td>98.61%</td>
  </tr>
  <tr>
    <td><b>Average</b></td>
    <td><b>82.83</b></td>
    <td><b>81.75</b></td>
    <td><b>98.70%</b></td>
  </tr>
  <tr>
    <td rowspan="7"><b>OpenLLM V2</b></td>
    <td>MMLU-Pro (5-shot)</td>
    <td>53.42</td>
    <td>51.09</td>
    <td>95.64%</td>
  </tr>
  <tr>
    <td>IFEval (0-shot)</td>
    <td>89.21</td>
    <td>87.89</td>
    <td>98.52%</td>
  </tr>
  <tr>
    <td>BBH (3-shot)</td>
    <td>69.05</td>
    <td>67.84</td>
    <td>98.25%</td>
  </tr>
  <tr>
    <td>Math-|v|-5 (4-shot)</td>
    <td>2.79</td>
    <td>3.25</td>
    <td>116.49%</td>
  </tr>
  <tr>
    <td>GPQA (0-shot)</td>
    <td>38.17</td>
    <td>34.82</td>
    <td>91.22%</td>
  </tr>
  <tr>
    <td>MuSR (0-shot)</td>
    <td>45.77</td>
    <td>45.77</td>
    <td>100.00%</td>
  </tr>
  <tr>
    <td><b>Average</b></td>
    <td><b>49.81</b></td>
    <td><b>48.45</b></td>
    <td><b>97.28%</b></td>
  </tr>

  <tr>
    <td><b>Coding</b></td>
    <td>HumanEval Instruct pass@1</td>
    <td>57.93</td>
    <td>56.71</td>
    <td>97.89%</td>
  </tr>
  <tr>
    <td></td>
    <td>HumanEval 64 Instruct pass@2</td>
    <td>69.75</td>
    <td>68.91</td>
    <td>98.80%</td>
  </tr>
    <tr>
    <td></td>
    <td>HumanEval 64 Instruct pass@8</td>
    <td>82.59</td>
    <td>81.53</td>
    <td>98.72%</td>
  </tr>
    <tr>
    <td></td>
    <td>HumanEval 64 Instruct pass@16</td>
    <td>85.75</td>
    <td>84.45</td>
    <td>98.48%</td>
  </tr>
    <tr>
    <td></td>
    <td>HumanEval 64 Instruct pass@32</td>
    <td>88.05</td>
    <td>86.63</td>
    <td>98.39%</td>
  </tr>
    <tr>
    <td></td>
    <td>HumanEval 64 Instruct pass@64</td>
    <td>89.63</td>
    <td>88.41</td>
    <td>98.64%</td>
  </tr>
 </tbody>
</table>


### Reproduction

The results were obtained using the following commands:

#### MMLU_LLAMA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks mmlu_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### MMLU_COT_LLAMA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks mmlu_cot_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks arc_challenge_llama \
  --apply_chat_template \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks gsm8k_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### Hellaswag
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks hellaswag \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks winogrande \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks truthfulqa \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval and HumanEval_64
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_instruct \
  --batch_size auto


lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Meta-Llama-3.1-70B-Instruct-NVFP4A16",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_64_instruct \
  --batch_size auto
```