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
base_model: Qwen/Qwen3-235B-A22B-Instruct-2507
---

# Qwen3-235B-A22B-Instruct-2507-NVFP4

## Model Overview
- **Model Architecture:** Qwen/Qwen3-235B-A22B-Instruct-2507
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 10/29/2025
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Qwen/Qwen3-235B-A22B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) to FP4 data type, ready for inference with vLLM>=0.9.1
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights and activations of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4"
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

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modeling.prepare import replace_modules_for_calibration

MODEL_ID = "Qwen/Qwen3-235B-A22B-Instruct-2507"

 #Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=None, torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(model)


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 1024

# --- Replace MoE modules for calibration ---
model = replace_modules_for_calibration(model, calibrate_all_experts=False)

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
    ignore=["re:.*lm_head.*", "re:.*mlp.gate$", "re:.*self_attn",
],
)

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
    pipeline="sequential",
    sequential_targets=["Qwen3MoeDecoderLayer"],
    calibrate_moe_context=True,

)
# Save to disk in compressed-tensors format.
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
      <th>Qwen/Qwen3-235B-A22B-Instruct-2507</th>
      <th>RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4 (this model)</th>
      <th>Recovery</th>
    </tr>
  </thead>
<tbody>
<<!-- OpenLLM V1 -->
<tr>
  <td rowspan="7"><b>OpenLLM V1</b></td>
  <td>arc_challenge</td>
  <td>72.78</td>
  <td>72.27</td>
  <td>99.29</td>
</tr>
<tr>
  <td>gsm8k</td>
  <td>89.92</td>
  <td>90.37</td>
  <td>100.50</td>
</tr>
<tr>
  <td>hellaswag</td>
  <td>87.77</td>
  <td>87.54</td>
  <td>99.74</td>
</tr>
<tr>
  <td>mmlu</td>
  <td>87.48</td>
  <td>87.08</td>
  <td>99.54</td>
</tr>
<tr>
  <td>truthfulqa_mc2</td>
  <td>62.10</td>
  <td>60.89</td>
  <td>98.06</td>
</tr>
<tr>
  <td>winogrande</td>
  <td>82.95</td>
  <td>81.69</td>
  <td>98.47</td>
</tr>
<tr>
  <td><b>Average</b></td>
  <td><b>80.50</b></td>
  <td><b>79.97</b></td>
  <td><b>101.43</b></td>
</tr>
<!-- OpenLLM V2 -->
<tr>
  <td rowspan="6"><b>OpenLLM V2</b></td>
  <td>BBH</td>
  <td>68.3</td>
  <td>69.05</td>
  <td>101.01%</td>
</tr>
<tr>
  <td>MMLU-Pro</td>
  <td>63.51</td>
  <td>63.33</td>
  <td>99.72</td>
</tr>
<tr>
  <td>MuSR</td>
  <td>45.11</td>
  <td>45.77</td>
  <td>101.46</td>
</tr>
<tr>
  <td>IFEval</td>
  <td>90.17</td>
  <td>91.01</td>
  <td>100.93</td>
</tr>
<tr>
  <td>GPQA</td>
  <td>34.56</td>
  <td>35.49</td>
  <td>102.7</td>
</tr>
<tr>
  <td><b>Average</b></td>
  <td>60.33</td>
  <td><b>60.93</b></td>
  <td>101.00</td>
</tr>
<tr>
  <td rowspan="3"><b>Reasoning</b></td>
  <td>GPQA (Diamond, 0-shot)</td>
  <td>69.19</td>
  <td>64.65</td>
  <td>93.44</td>
</tr>
<tr>
  <td>Math-500 (0-shot)</td>
  <td>94.20</td>
  <td>89.20</td>
  <td>94.69</td>
</tr>
<tr>
  <td><b>Average</b></td>
  <td><b>81.70</b></td>
  <td><b>76.93</b></td>
  <td><b>94.16</b></td>
</tr>

  <!-- Coding -->
<tr>
  <td rowspan="1"><b>Coding</b></td>
  <td>HumanEval_64 pass@2</td>
  <td>96.67</td>
  <td>96.46</td>
  <td>99.78</td>
</tr>
</tbody>
</table>




### Reproduction

The results were obtained using the following commands:

<details>

```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks openllm \
  --batch_size auto
```


#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval_64
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Qwen3-235B-A22B-Instruct-2507-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_64_instruct \
  --batch_size auto
```
</details>