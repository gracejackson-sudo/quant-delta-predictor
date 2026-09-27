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
license: apache-2.0
base_model: Qwen/Qwen3-32B
---

# Qwen3-32B-NVFP4A16

## Model Overview
- **Model Architecture:** Qwen/Qwen3-32B
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP16
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 6/25/2025
- **Version:** 10
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) to FP4 data type, ready for inference with vLLM>=9.1
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 25%.

Only the weights of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-32B-NVFP4A16"
number_gpus = 2

sampling_params = SamplingParams(temperature=6, top_p=9, max_tokens=256)

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

MODEL_ID = "Qwen/Qwen3-32B"

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

<h3>Accuracy</h3>
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Qwen/Qwen3-32B</th>
      <th>RedHatAI/Qwen3-32B-NVFP4A16 (this model)</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>MMLU</td>
      <td>80.94</td>
      <td>80.57</td>
      <td>99.55%</td>
    </tr>
    <tr>
      <td>ARC Challenge (0-shot)</td>
      <td>68.34</td>
      <td>68.43</td>
      <td>100.12%</td>
    </tr>
    <tr>
      <td>GSM8K (8-shot, strict-match)</td>
      <td>87.34</td>
      <td>87.72</td>
      <td>100.43%</td>
    </tr>
    <tr>
      <td>Hellaswag (10-shot)</td>
      <td>71.16</td>
      <td>70.48</td>
      <td>99.05%</td>
    </tr>
    <tr>
      <td>Winogrande (5-shot)</td>
      <td>69.93</td>
      <td>70.09</td>
      <td>100.23%</td>
    </tr>
    <tr>
      <td>TruthfulQA (0-shot, mc2)</td>
      <td>58.63</td>
      <td>58.96</td>
      <td>100.56%</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>72.72</b></td>
      <td><b>72.71</b></td>
      <td><b>99.98%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>MMLU-Pro (5-shot)</td>
      <td>54.48</td>
      <td>51.61</td>
      <td>94.73%</td>
    </tr>
    <tr>
      <td>IFEval (0-shot)</td>
      <td>88.85</td>
      <td>88.49</td>
      <td>99.59%</td>
    </tr>
    <tr>
      <td>BBH (3-shot)</td>
      <td>62.61</td>
      <td>62.14</td>
      <td>99.25%</td>
    </tr>
    <tr>
      <td>Math-|v|-5 (4-shot)</td>
      <td>56.87</td>
      <td>56.27</td>
      <td>98.94%</td>
    </tr>
    <tr>
      <td>GPQA (0-shot)</td>
      <td>30.45</td>
      <td>30.29</td>
      <td>99.47%</td>
    </tr>
    <tr>
      <td>MuSR (0-shot)</td>
      <td>39.15</td>
      <td>40.48</td>
      <td>103.40%</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>55.40</b></td>
      <td><b>54.88</b></td>
      <td><b>99.06%</b></td>
    </tr>
    <tr>
      <td><b>Coding</b></td>
      <td>HumanEval Instruct pass@1</td>
      <td>88.41</td>
      <td>87.20</td>
      <td>98.63%</td>
    </tr>
    <tr>
      <td></td>
      <td>HumanEval 64 Instruct pass@2</td>
      <td>90.27</td>
      <td>89.66</td>
      <td>99.32%</td>
    </tr>
    <tr>
      <td></td>
      <td>HumanEval 64 Instruct pass@8</td>
      <td>92.20</td>
      <td>92.13</td>
      <td>99.92%</td>
    </tr>
    <tr>
      <td></td>
      <td>HumanEval 64 Instruct pass@16</td>
      <td>92.96</td>
      <td>93.27</td>
      <td>100.33%</td>
    </tr>
    <tr>
      <td></td>
      <td>HumanEval 64 Instruct pass@32</td>
      <td>93.58</td>
      <td>94.47</td>
      <td>100.95%</td>
    </tr>
    <tr>
      <td></td>
      <td>HumanEval 64 Instruct pass@64</td>
      <td>93.90</td>
      <td>95.73</td>
      <td>101.95%</td>
    </tr>
  </tbody>
</table>

### Reproduction

The results were obtained using the following commands:

#### OpenLLM v1
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-32B-NVFP4A16",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks openllm \
  --batch_size auto
```


#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-32B-NVFP4A16",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval and HumanEval_64
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-32B-NVFP4A16",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_instruct \
  --batch_size auto


lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-32B-NVFP4A16",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_64_instruct \
  --batch_size auto
```