---
library_name: vllm
license: apache-2.0
language:
  - en
  - fr
  - es
  - it
  - pt
  - zh
  - ar
  - ru
base_model:
  - HuggingFaceTB/SmolLM3-3B
tags:
- neuralmagic
- redhat
- llmcompressor
- int4
- w4a16
- quantized
---

## Model Overview
- **Model Architecture:** SmolLM3-3B
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** None
- **Release Date:** 07/31/2025
- **Version:** 1.0
- **License(s):** Apache-2.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing weights of [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) to INT4 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 4, reducing GPU memory requirements (by approximately 75%).
Weight quantization also reduces disk size requirements by approximately 75%.
Only weights of the linear operators within transformers blocks are quantized.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/SmolLM3-3B-quantized.w4a16"
number_gpus = 1

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.


## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below with:

```bash
python int4.py --model_path HuggingFaceTB/SmolLM3-3B --calib_size 1024 --dampening_frac 0.1 --observer minmax --actorder group --sym false
```
where `int4.py` is as follows:


  ```python
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
)
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot

# Constants
DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"
MAX_SEQ_LENGTH = 8192
IGNORE_MODULES = ["lm_head"]

# Argument Parsing Utilities
def parse_actorder(value: str):
    value_lower = value.lower()
    if value_lower == "false":
        return False
    if value_lower in {"weight", "group"}:
        return value_lower
    raise argparse.ArgumentTypeError(f"Invalid --actorder. Choose 'group', 'weight', or 'false', got {value}")

def parse_sym(value: str):
    value_lower = value.lower()
    if value_lower in {"true", "false"}:
        return value_lower == "true"
    raise argparse.ArgumentTypeError(f"Invalid --sym. Use 'true' or 'false', got {value}")

# Argument Parser
def get_args():
    parser = argparse.ArgumentParser(description="Quantize a model with GPTQModifier.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the unquantized model.")
    parser.add_argument('--calib_size', type=int, default=256, help="Number of samples for calibration.")
    parser.add_argument('--dampening_frac', type=float, default=0.1, help="Dampening fraction for quantization.")
    parser.add_argument('--observer', type=str, default="minmax", help="Observer type used for quantization.")
    parser.add_argument('--sym', type=parse_sym, default=True, help="Symmetric quantization (true/false).")
    parser.add_argument('--actorder', type=parse_actorder, default=False,
                        help="Activation order: 'group', 'weight', or 'false'.")
    return parser.parse_args()

def main():
    args = get_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype="auto",
        use_cache=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load and preprocess dataset
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(args.calib_size))
    ds = ds.map(lambda x: {"text": x["text"]})
    ds = ds.map(
        lambda x: tokenizer(x["text"], padding=False, truncation=False, add_special_tokens=True),
        remove_columns=ds.column_names
    )

    # Build Quantization Scheme
    quant_scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type=QuantizationType.INT,
            symmetric=args.sym,
            group_size=128,
            strategy=QuantizationStrategy.GROUP,
            observer=args.observer,
            actorder=args.actorder
        ),
        input_activations=None,
        output_activations=None,
    )

    # Define compression recipe
    recipe = [
        GPTQModifier(
            targets=["Linear"],
            ignore=IGNORE_MODULES,
            dampening_frac=args.dampening_frac,
            config_groups={"group_0": quant_scheme},
        )
    ]

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=args.calib_size,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Save the quantized model
    save_path = f"{args.model_path}-quantized.w4a16"
    model.save_pretrained(save_path, save_compressed=True)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()
```

</details>

## Evaluation

This model was evaluated on the well-known reasoning tasks: AIME24, Math-500, and GPQA-Diamond.
In all cases, model outputs were generated with the [vLLM](https://docs.vllm.ai/en/stable/) engine, and evals are collected through [LightEval](https://github.com/huggingface/lighteval) library.


<details>
  <summary>Evaluation details</summary>

  ```
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export MODEL="RedHatAI/SmolLM3-3B-quantized.w4a16"
    export MODEL_ARGS="model_name=$MODEL,dtype=auto,max_model_length=65536,gpu_memory_utilization=0.9,tensor_parallel_size=1,add_special_tokens=False,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"

    export TASK=aime24 # {aime24, math_500, gpqa:diamond}

    lighteval vllm $MODEL_ARGS "lighteval|${TASK}|0|0" \
        --use-chat-template \
        --output-dir out_dir
  ```
</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>HuggingFaceTB/SmolLM3-3B
   </th>
   <th>RedHatAI/SmolLM3-3B-quantized.w4a16<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="8" ><strong>Reasoning</strong>
   </td>
   <td>AIME24 (pass@1:64)
   </td>
   <td>45.31
   </td>
   <td>39.27
   </td>
   <td>86.67%
   </td>
  </tr>
  <tr>
   <td>MATH-500 (pass@1:4)
   </td>
   <td>89.30
   </td>
   <td>87.55
   </td>
   <td>98.04%
   </td>
  </tr>
  <tr>
   <td>GPQA-Diamond (pass@1:8)
   </td>
   <td>41.22
   </td>
   <td>41.86
   </td>
   <td>101.55%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>58.61</strong>
   </td>
   <td><strong>56.23</strong>
   </td>
   <td><strong>95.94%</strong>
   </td>
  </tr>
  <tr>
</table>



