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
base_model: Qwen/Qwen3-30B-A3B
---

# Qwen3-30B-A3B-NVFP4

## Model Overview
- **Model Architecture:** Qwen/Qwen3-30B-A3B
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 10/29/2025
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) to FP4 data type, ready for inference with vLLM>=0.9.1
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights and activations of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-30B-A3B-NVFP4"
number_gpus = 1

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

<details>
  
```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "Qwen/Qwen3-30B-A3B"

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
#   * calibrate a global_scale for activations, which will be used to
#       quantize activations to fp4 on the fly
recipe = [
    QuantizationModifier(
        ignore=["re:.*lm_head.*"],
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "symmetric": True,
                    "observer": "minmax",
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "tensor_group",
                    "group_size": 16,
                    "symmetric": True,
                    "dynamic": "local",
                    "observer": "minmax",
                },
            }
        },
    )
]

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"

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
</details>

## Evaluation

This model was evaluated on the well-known OpenLLM v1, OpenLLM v2 and HumanEval_64 benchmarks using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness). The Reasoning evals were done using [ligheval](https://github.com/neuralmagic/lighteval).

### Accuracy
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Qwen3-30B-A3B</th>
      <th>Qwen3-30B-A3B-NVFP4 (this model)</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>arc_challenge</td>
      <td>67.15</td>
      <td>64.59</td>
      <td>96.19</td>
    </tr>
    <tr>
      <td>gsm8k</td>
      <td>89.46</td>
      <td>87.72</td>
      <td>98.05</td>
    </tr>
    <tr>
      <td>hellaswag</td>
      <td>77.55</td>
      <td>76.74</td>
      <td>98.96</td>
    </tr>
    <tr>
      <td>mmlu</td>
      <td>79.51</td>
      <td>77.54</td>
      <td>97.52</td>
    </tr>
    <tr>
      <td>truthfulqa_mc2</td>
      <td>53.50</td>
      <td>54.14</td>
      <td>101.20</td>
    </tr>
    <tr>
      <td>winogrande</td>
      <td>72.30</td>
      <td>70.80</td>
      <td>97.93</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>73.25</b></td>
      <td><b>71.92</b></td>
      <td><b>98.19</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>BBH (3-shot)</td>
      <td>54.97</td>
      <td>45.63</td>
      <td>83.01</td>
    </tr>
    <tr>
      <td>MMLU-Pro (5-shot)</td>
      <td>47.39</td>
      <td>42.56</td>
      <td>89.81</td>
    </tr>
    <tr>
      <td>MuSR (0-shot)</td>
      <td>39.55</td>
      <td>39.15</td>
      <td>98.99</td>
    </tr>
    <tr>
      <td>IFEval (0-shot)</td>
      <td>88.61</td>
      <td>86.93</td>
      <td>98.10</td>
    </tr>
    <tr>
      <td>GPQA (0-shot)</td>
      <td>27.94</td>
      <td>26.26</td>
      <td>93.99</td>
    </tr>
    <tr>
      <td>Math-|v|-5 (4-shot)</td>
      <td>58.23</td>
      <td>53.40</td>
      <td>91.71</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>52.78</b></td>
      <td><b>48.99</b></td>
      <td><b>92.81</b></td>
    </tr>
    <tr>
      <td rowspan="1"><b>Coding</b></td>
      <td>HumanEval_64 pass@2</td>
      <td>93.62</td>
      <td>91.13</td>
      <td>97.34</td>
    </tr>
  </tbody>
</table>




### Reproduction

The results were obtained using the following commands:

<details>

```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-30B-A3B-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks openllm \
  --batch_size auto
```


#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-30B-A3B-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval_64
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-30B-A3B-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_64_instruct \
  --batch_size auto
```
</details>