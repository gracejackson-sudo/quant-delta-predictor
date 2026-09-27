---
tags:
- w8a8
- int8
- vllm
license: apache-2.0
license_link: https://huggingface.co/Qwen/QwQ-32B-Preview/blob/main/LICENSE
language:
  - en
base_model: Qwen/Qwen2.5-32B-Instruct
library_name: transformers
---

# QwQ-32B-Preview-quantized.w8a8

## Model Overview
- **Model Architecture:** QwQ-32B-Preview
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
  - **Activation quantization:** INT8
- **Release Date:** 3/1/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview).
It achieves an average score of 76.49 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 77.20.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) to INT8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%. Only the weights and activations of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 1
model_name = "neuralmagic-ent/QwQ-32B-Preview-quantized.w8a8"
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

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below with the following arguments:


```bash
python quantize.py --model_path Qwen/QwQ-32B-Preview --quant_path "output_dir/QwQ-32B-Preview-quantized.w8a8" --calib_size 1024 --dampening_frac 0.1 --observer mse 
```


```python
from datasets import load_dataset
from transformers import AutoTokenizer
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot, apply
import argparse
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--quant_path', type=str)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.1) 
parser.add_argument('--observer', type=str, default="minmax")
args = parser.parse_args()

model = SparseAutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype="auto",
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

recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="W8A8",
        dampening_frac=args.dampening_frac,
        observer=args.observer,
    )
]
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=args.calib_size,
    max_seq_length=8192,
)

# Save to disk compressed.
SAVE_DIR = args.quant_path
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) and [V2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/), using the following commands:

OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic-ent/QwQ-32B-Preview-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks openllm \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```

OpenLLM Leaderboard V2:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic-ent/QwQ-32B-Preview-quantized.w8a8",dtype=auto,add_bos_token=False,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config

```

### Accuracy

#### OpenLLM Leaderboard V1 evaluation scores

| Metric                                   | Qwen/QwQ-32B-Preview             | neuralmagic-ent/QwQ-32B-Preview-quantized.w8a8 |
|-----------------------------------------|:---------------------------------:|:-------------------------------------------:|
| ARC-Challenge (Acc-Norm, 25-shot)       | 70.73                            | 70.73                                       |
| GSM8K (Strict-Match, 5-shot)            | 83.09                            | 79.91                                       |
| HellaSwag (Acc-Norm, 10-shot)           | 85.77                            | 85.75                                       |
| MMLU (Acc, 5-shot)                      | 82.67                            | 82.24                                       |
| TruthfulQA (MC2, 0-shot)                | 60.88                            | 59.18                                       |
| Winogrande (Acc, 5-shot)                | 80.03                            | 81.14                                       |
| **Average Score**                       | **77.20**                        | **76.49**                                   |
| **Recovery**                            | **100.00**                       | **99.08**                                   |

#### OpenLLM Leaderboard V2 evaluation scores

| Metric                                                   | Qwen/QwQ-32B-Preview             | neuralmagic-ent/QwQ-32B-Preview-quantized.w8a8 |
|---------------------------------------------------------|:---------------------------------:|:-------------------------------------------:|
| IFEval (Inst-and-Prompt Level Strict Acc, 0-shot)       | 42.34                            | 43.49                                       |
| BBH (Acc-Norm, 3-shot)                                  | 53.03                            | 52.95                                       |
| Math-Hard (Exact-Match, 4-shot)                         | 21.15                            | 22.36                                       |
| GPQA (Acc-Norm, 0-shot)                                 | 2.97                             | 3.5                                        |
| MUSR (Acc-Norm, 0-shot)                                 | 9.57                             | 10.87                                       |
| MMLU-Pro (Acc, 5-shot)                                  | 52.00                            | 51.4                                       |
| **Average Score**                                       | **30.18**                        | **30.76**                                   |
| **Recovery**                                            | **100.00**                       | **101.92**                                   |

