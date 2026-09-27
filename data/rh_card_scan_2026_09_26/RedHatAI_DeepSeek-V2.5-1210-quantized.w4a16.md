---
language:
- en
tags:
- moe
- int4
- w4a16
- vllm
license: other
license_name: deepseek
license_link: https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL
library_name: transformers
---

# DeepSeek-V2.5-1210-quantized.w4a16

## Model Overview
- **Model Architecture:** DeepSeek-V2.5-1210
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** None
- **Release Date:** 3/1/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-V2.5-1210](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210).
It achieves an average score of 77.2 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 77.82.

### Model Optimizations

This model was obtained by quantizing only the weights to INT4 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%. The weights of the linear operators within transformers blocks are quantized, except the MLP routers. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 2
model_name = "neuralmagic-ent/DeepSeek-V2.5-1210-quantized.w4a16"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True)
sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])

messages_list = [
    [{"role": "user", "content": "Who are you? Please respond in pirate speak!"}],
]

prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

generated_text = [output.outputs[0].text for output in outputs]
print(generated_text)
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below with the following command:

```bash
python quantize.py --model_path deepseek-ai/DeepSeek-V2.5-1210 --quant_path "output_dir" --calib_size 256 --dampening_frac 0.1 --observer minmax --actorder weight
```


```python
`from datasets import load_dataset
from transformers import AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot, apply
import argparse
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
import torch


def parse_actorder(value):
    # Interpret the input value for --actorder
    if value.lower() == "false":
        return False
    elif value.lower() == "weight":
        return "weight"
    elif value.lower() == "group":
        raise ValueError("group not supported for TP>1 and MoEs")
    else:
        raise argparse.ArgumentTypeError("Invalid value for --actorder. Use 'group' or 'False'.")


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--quant_path', type=str)
parser.add_argument('--num_bits', type=int, default=4)
parser.add_argument('--sequential_update', type=bool, default=True)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.05)
parser.add_argument('--observer', type=str, default="minmax")
parser.add_argument(
    '--actorder',
    type=parse_actorder,
    default=False,  # Default value is False
    help="Specify actorder as 'group' (string) or False (boolean)."
)

args = parser.parse_args()

device_map = calculate_offload_device_map(
    args.model_path,
    reserve_for_hessians=True,
    num_gpus=torch.cuda.device_count(),
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = SparseAutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

NUM_CALIBRATION_SAMPLES = args.calib_size
DATASET_ID = "garage-bAInd/Open-Platypus"
DATASET_SPLIT = "train"
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    concat_txt = example["instruction"] + "\n" + example["output"]
    return {"text": concat_txt}

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=args.num_bits,
        type=QuantizationType.INT,
        symmetric=True,
        group_size=128,
        strategy=QuantizationStrategy.GROUP,
        observer=args.observer,
        actorder=args.actorder
    ),
    input_activations=None,
    output_activations=None,
)

recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head", "re:.*\.mlp\.gate$"],
        sequential_update=args.sequential_update,
        dampening_frac=args.dampening_frac,
        config_groups={"group_0": quant_scheme},
    )
]
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=args.calib_size,
)

# Save to disk compressed.
SAVE_DIR = args.quant_path
model.save_pretrained(SAVE_DIR, save_compressed=True, skip_compression_stats=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) using the following command:

OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic-ent/DeepSeek-V2.5-1210-quantized.w4a16",dtype=float16,add_bos_token=True,max_model_len=4096,tensor_parallel_size=4,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks openllm \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```


### Accuracy

#### OpenLLM Leaderboard V1 evaluation scores

| Metric                                   | deepseek-ai/DeepSeek-V2.5-1210             | neuralmagic-ent/DeepSeek-V2.5-1210-quantized.w4a16 |
|-----------------------------------------|:---------------------------------:|:-------------------------------------------:|
| ARC-Challenge (Acc-Norm, 25-shot)       | 72.61                            | 72.18                                       |
| GSM8K (Strict-Match, 5-shot)            | 88.25                            | 88.10                                       |
| HellaSwag (Acc-Norm, 10-shot)           | 85.01                            | 83.48                                       |
| MMLU (Acc, 5-shot)                      | 79.60                            | 78.36                                       |
| TruthfulQA (MC2, 0-shot)                | 57.18                            | 57.01                                       |
| Winogrande (Acc, 5-shot)                | 84.29                            | 84.06                                       |
| **Average Score**                       | **77.82**                        | **77.20**                                   |
| **Recovery**                            | **100.00**                       | **99.20**                                   |

