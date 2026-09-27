---
language:
- en
tags:
- moe
- fp8
- vllm
license: other
license_name: deepseek
license_link: https://github.com/deepseek-ai/DeepSeek-V2/blob/main/LICENSE-MODEL
library_name: transformers
---

# DeepSeek-V2.5-1210-FP8

## Model Overview
- **Model Architecture:** DeepSeek-V2.5-1210
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 3/1/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-V2.5-1210](https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210).
It achieves an average score of 77.8 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 77.82.

### Model Optimizations

This model was obtained by quantizing the weights and activations to FP8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%. The weights and activations of the linear operators within transformers blocks are quantized, except the MLP routers. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 4
model_name = "neuralmagic-ent/DeepSeek-V2.5-1210-FP8"
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
python quantize.py --model_path deepseek-ai/DeepSeek-V2.5-1210 --quant_path "output_dir" --calib_size 128
```


```python
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
import torch
import os


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Quantize a transformer model to FP8')
    parser.add_argument('--model_id', type=str, required=True,
                        help='The model ID from HuggingFace (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")')
    parser.add_argument('--save_path', type=str, default='.',
                        help='Custom path to save the quantized model. If not provided, will use model_name-FP8')
    parser.add_argument('--calib_size', type=int, default=256)
    args = parser.parse_args()

    device_map = calculate_offload_device_map(
        args.model_id,
        reserve_for_hessians=False,
        num_gpus=torch.cuda.device_count(),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

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

    # Configure the quantization algorithm and scheme
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8", ignore=["lm_head", "re:.*\.mlp\.gate$"]
    )

    # Apply quantization
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        num_calibration_samples=args.calib_size
    )

    save_path = os.path.join(args.save_path, args.model_id.split("/")[1] + "-FP8")
    os.makedirs(save_path, exist_ok=True)

    # Save to disk in compressed-tensors format
    model.save_pretrained(save_path, save_compressed=True, skip_compression_stats=True)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")

if __name__ == "__main__":
    main()
```

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) using the following command:

OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic-ent/DeepSeek-V2.5-1210-FP8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=4,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks openllm \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```


### Accuracy

#### OpenLLM Leaderboard V1 evaluation scores

| Metric                                   | deepseek-ai/DeepSeek-V2.5-1210             | neuralmagic-ent/DeepSeek-V2.5-1210-FP8 |
|-----------------------------------------|:---------------------------------:|:-------------------------------------------:|
| ARC-Challenge (Acc-Norm, 25-shot)       | 72.61                            | 72.44                                       |
| GSM8K (Strict-Match, 5-shot)            | 88.25                            | 88.02                                       |
| HellaSwag (Acc-Norm, 10-shot)           | 85.01                            | 84.75                                       |
| MMLU (Acc, 5-shot)                      | 79.60                            | 79.44                                       |
| TruthfulQA (MC2, 0-shot)                | 57.18                            | 57.53                                       |
| Winogrande (Acc, 5-shot)                | 84.29                            | 84.61                                       |
| **Average Score**                       | **77.82**                        | **77.80**                                   |
| **Recovery**                            | **100.00**                       | **99.97**                                   |

