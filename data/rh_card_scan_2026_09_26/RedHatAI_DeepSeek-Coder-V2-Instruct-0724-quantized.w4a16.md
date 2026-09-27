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

# DeepSeek-Coder-V2-Instruct-0724-quantized.w4a16

## Model Overview
- **Model Architecture:** DeepSeek-Coder-V2-Instruct-0724
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** None
- **Release Date:** 3/1/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-Coder-V2-Instruct-0724](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724).

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
model_name = "neuralmagic-ent/DeepSeek-Coder-V2-Instruct-0724-quantized.w4a16"
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
python quantize.py --model_path deepseek-ai/DeepSeek-Coder-V2-Instruct-0724 --quant_path "output_dir" --calib_size 256 --dampening_frac 0.1 --observer mse --actorder False
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

The model was evaluated on [HumanEval and HumanEval+](https://github.com/openai/human-eval?tab=readme-ov-file) benchmark with the [Neural Magic fork](https://github.com/neuralmagic/evalplus) of the [EvalPlus implementation of HumanEval+](https://github.com/evalplus/evalplus) and the [vLLM](https://docs.vllm.ai/en/stable/) engine, using the following commands:

```
python evalplus/codegen/generate.py --model neuralmagic-ent/DeepSeek-Coder-V2-Instruct-0724-quantized.w4a16 --bs 16 --temperature 0.2 --n_samples 50 --root "./results" --dataset humaneval --backend vllm --dtype auto --tp 8 

python evalplus/evalplus/sanitize.py results/humaneval/neuralmagic-ent--DeepSeek-Coder-V2-Instruct-0724-quantized.w4a16_vllm_temp_0.2

evalplus.evaluate --dataset humaneval --samples results/humaneval/neuralmagic-ent--DeepSeek-Coder-V2-Instruct-0724-quantized.w4a16_vllm_temp_0.2-sanitized
```


### Accuracy

#### HumanEval evaluation scores

| Metric                 | deepseek-ai/DeepSeek-Coder-V2-Instruct-0724             | neuralmagic-ent/DeepSeek-Coder-V2-Instruct-0724-quantized.w4a16 |
|------------------------|:---------------------------------:|:-------------------------------------------:|
| HumanEval pass@1       | 89.3                            | 85.5                                        |
| HumanEval pass@10      | 93.1                            | 91.1                                        |
| HumanEval+ pass@1      | 82.9                            | 80.7                                       |
| HumanEval+ pass@10     | 87.6                            | 85.9                                       |
| **Average Score**                       | **88.23**                        | **85.8**                                   |
| **Recovery**                            | **100.00**                       | **97.25**                                   |

