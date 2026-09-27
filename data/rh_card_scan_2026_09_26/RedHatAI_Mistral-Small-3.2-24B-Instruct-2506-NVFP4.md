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
base_model: unsloth/Mistral-Small-3.2-24B-Instruct-2506
---

# Mistral-Small-3.2-24B-Instruct-2506-NVFP4

## Model Overview
- **Model Architecture:** unsloth/Mistral-Small-3.2-24B-Instruct-2506
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 10/29/2025
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [unsloth/Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [unsloth/Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506) to FP4 data type, ready for inference with vLLM>=0.9.1
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights and activations of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

1. Initialize vLLM server:
```
vllm serve RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4 --tensor_parallel_size 1 --tokenizer_mode mistral
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

model = "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4"


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

This model was created by applying [LLM Compressor with calibration samples from UltraChat](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/llama3_example.py), as presented in the code snipet below.

<details>
  
```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.utils import dispatch_for_generation

MODEL_ID = "unsloth/Mistral-Small-3.2-24B-Instruct-2506"

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
smoothing_strength = 0.9
recipe = [
    SmoothQuantModifier(smoothing_strength=smoothing_strength),
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
                    "observer": "mse",
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

This model was evaluated on the well-known OpenLLM v1, OpenLLM v2 and HumanEval_64 benchmarks using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness).

### Accuracy
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>unsloth/Mistral-Small-3.2-24B-Instruct-2506</th>
      <th>RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>arc_challenge</td>
      <td>68.52</td>
      <td>66.98</td>
      <td>97.75</td>
    </tr>
    <tr>
      <td>gsm8k</td>
      <td>89.61</td>
      <td>87.11</td>
      <td>97.21</td>
    </tr>
    <tr>
      <td>hellaswag</td>
      <td>85.70</td>
      <td>85.11</td>
      <td>99.31</td>
    </tr>
    <tr>
      <td>mmlu</td>
      <td>81.06</td>
      <td>79.43</td>
      <td>97.99</td>
    </tr>
    <tr>
      <td>truthfulqa_mc2</td>
      <td>61.35</td>
      <td>60.34</td>
      <td>98.35</td>
    </tr>
    <tr>
      <td>winogrande</td>
      <td>83.27</td>
      <td>81.61</td>
      <td>98.01</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>78.25</b></td>
      <td><b>76.76</b></td>
      <td><b>98.10</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>BBH (3-shot)</td>
      <td>65.86</td>
      <td>64.05</td>
      <td>97.25</td>
    </tr>
    <tr>
      <td>MMLU-Pro (5-shot)</td>
      <td>50.84</td>
      <td>48.45</td>
      <td>95.30</td>
    </tr>
    <tr>
      <td>MuSR (0-shot)</td>
      <td>39.15</td>
      <td>40.21</td>
      <td>102.71</td>
    </tr>
    <tr>
      <td>IFEval (0-shot)</td>
      <td>84.05</td>
      <td>84.41</td>
      <td>100.43</td>
    </tr>
    <tr>
      <td>GPQA (0-shot)</td>
      <td>33.14</td>
      <td>32.55</td>
      <td>98.22</td>
    </tr>
    <tr>
      <td>Math-|v|-5 (4-shot)</td>
      <td>41.69</td>
      <td>37.76</td>
      <td>90.57</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>52.46</b></td>
      <td><b>51.24</b></td>
      <td><b>97.68</b></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Coding</b></td>
      <td>HumanEval_64 pass@2</td>
      <td>88.88</td>
      <td>88.84</td>
      <td>99.95</td>
    </tr>
  </tbody>
</table>




### Reproduction

The results were obtained using the following commands:

<details>

```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks openllm \
  --batch_size auto
```


#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval_64
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_64_instruct \
  --batch_size auto

```
</details>