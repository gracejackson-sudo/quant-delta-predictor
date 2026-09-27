---
tags:
- fp8
- vllm
license: other
license_name: bigcode-openrail-m
license_link: https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement
---

# starcoder2-7b-FP8

## Model Overview
- **Model Architecture:** starcoder2-7b
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in English.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 8/1/2024
- **Version:** 1.0
- **License(s):** [bigcode-openrail-m](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement)
- **Model Developers:** Neural Magic

Quantized version of [starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b).
<!-- It achieves an average score of 73.19 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 73.48. -->
It achieves an average score of 39.30 on the [HumanEval+](https://github.com/openai/human-eval?tab=readme-ov-file) benchmark, whereas the unquantized model achieves 39.65.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [starcoder2-7b](https://huggingface.co/bigcode/starcoder2-7b) to FP8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-tensor quantization is applied, in which a single linear scaling maps the FP8 representations of the quantized weights and activations.
[AutoFP8](https://github.com/neuralmagic/AutoFP8) is used for quantization with 512 sequences of UltraChat.

<!-- ## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/starcoder2-7b-FP8"

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_id, trust_remote_code=True, max_model_len=4096)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details. -->

## Creation

This model was created by applying [LLM Compressor with calibration samples from UltraChat](https://github.com/vllm-project/llm-compressor/blob/sa/big_model_support/examples/big_model_offloading/big_model_w8a8_calibrate.py), as presented in the code snipet below.
A slight modification to the code was made due to the parameters of the model. Running the below code will throw an index error, and simply replacing the erroneous line with ```max_quant_shape = param.shape[0]``` resolves the issue. 

```python
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import (
    calculate_offload_device_map,
    custom_offload_device_map,
)

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
"""

model_stub = "bigcode/starcoder2-7b"
model_name = model_stub.split("/")[-1]

device_map = calculate_offload_device_map(
    model_stub, reserve_for_hessians=False, num_gpus=8, torch_dtype=torch.float16
)

model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype=torch.float16, device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_stub)

output_dir = f"./{model_name}-FP8"

DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {
        "text": " ".join([msg["content"] for msg in example["messages"]])
    }

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

ds = ds.map(tokenize, remove_columns=ds.column_names)

oneshot(
    model=model,
    output_dir=output_dir,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    save_compressed=True,
)
```

## Evaluation

The model was evaluated on the [HumanEval+](https://github.com/openai/human-eval?tab=readme-ov-file) benchmark with the [Neural Magic fork](https://github.com/neuralmagic/evalplus) of the [EvalPlus implementation of HumanEval+](https://github.com/evalplus/evalplus) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following command:
```
python codegen/generate.py --model neuralmagic/starcoder2-7b-FP8 --temperature 0.2 --n_samples 50 --resume --root ~ --dataset humaneval
python evalplus/sanitize.py ~/humaneval/neuralmagic--starcoder2-7b-FP8_vllm_temp_0.2
evalplus.evaluate --dataset humaneval --samples ~/humaneval/neuralmagic--starcoder2-7b-FP8_vllm_temp_0.2-sanitized
```

### Accuracy

#### HumanEval+ evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>starcoder2-7b</strong>
   </td>
   <td><strong>starcoder2-7b-FP8(this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>base pass@1
   </td>
   <td>34.9
   </td>
   <td>34.6
   </td>
   <td>99.14%
   </td>
  </tr>
  <tr>
   <td>base pass@10
   </td>
   <td>50.7
   </td>
   <td>50.1
   </td>
   <td>98.82%
   </td>
  </tr>
  <tr>
   <td>base+extra pass@1
   </td>
   <td>30.0
   </td>
   <td>30.3
   </td>
   <td>101.00%
   </td>
  </tr>
  <tr>
   <td>base+extra pass@10
   </td>
   <td>43.0
   </td>
   <td>42.2
   </td>
   <td>98.14%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>39.65</strong>
   </td>
   <td><strong>39.30</strong>
   </td>
   <td><strong>99.27%</strong>
   </td>
  </tr>
</table>