---
tags:
- fp8
- vllm
license: apache-2.0
license_link: https://huggingface.co/Qwen/QwQ-32B-Preview/blob/main/LICENSE
language:
  - en
base_model: Qwen/Qwen2.5-32B-Instruct
library_name: transformers
---

# QwQ-32B-Preview-FP8-dynamic

## Model Overview
- **Model Architecture:** QwQ-32B-Preview
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 3/1/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview).
It achieves an average score of 77.08 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 77.20.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) to FP8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%. Only the weights and activations of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 1
model_name = "neuralmagic-ent/QwQ-32B-Preview-FP8-dynamic"
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

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


```python
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
import os

def main():
    parser = argparse.ArgumentParser(description='Quantize a transformer model to FP8')
    parser.add_argument('--model_id', type=str, required=True,
                        help='The model ID from HuggingFace (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")')
    parser.add_argument('--save_path', type=str, default='.',
                        help='Custom path to save the quantized model. If not provided, will use model_name-FP8-dynamic')
    args = parser.parse_args()

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # Configure the quantization algorithm and scheme
    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )

    # Apply quantization
    oneshot(model=model, recipe=recipe)

    save_path = os.path.join(args.save_path, args.model_id.split("/")[1] + "-FP8-dynamic")
    os.makedirs(save_path, exist_ok=True)

    # Save to disk in compressed-tensors format
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model and tokenizer saved to: {save_path}")

if __name__ == "__main__":
    main()
```

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) and [V2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/), using the following commands:

OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic-ent/QwQ-32B-Preview-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
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
  --model_args pretrained="neuralmagic-ent/QwQ-32B-Preview-FP8-dynamic",dtype=auto,add_bos_token=False,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
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

| Metric                                   | Qwen/QwQ-32B-Preview             | neuralmagic-ent/QwQ-32B-Preview-FP8-dynamic |
|-----------------------------------------|:---------------------------------:|:-------------------------------------------:|
| ARC-Challenge (Acc-Norm, 25-shot)       | 70.73                            | 71.08                                       |
| GSM8K (Strict-Match, 5-shot)            | 83.09                            | 82.18                                       |
| HellaSwag (Acc-Norm, 10-shot)           | 85.77                            | 85.88                                       |
| MMLU (Acc, 5-shot)                      | 82.67                            | 82.67                                       |
| TruthfulQA (MC2, 0-shot)                | 60.88                            | 60.54                                       |
| Winogrande (Acc, 5-shot)                | 80.03                            | 80.11                                       |
| **Average Score**                       | **77.20**                        | **77.08**                                   |
| **Recovery**                            | **100.00**                       | **99.84**                                   |

#### OpenLLM Leaderboard V2 evaluation scores

| Metric                                                   | Qwen/QwQ-32B-Preview             | neuralmagic-ent/QwQ-32B-Preview-FP8-dynamic |
|---------------------------------------------------------|:---------------------------------:|:-------------------------------------------:|
| IFEval (Inst-and-Prompt Level Strict Acc, 0-shot)       | 42.34                            | 40.48                                       |
| BBH (Acc-Norm, 3-shot)                                  | 53.03                            | 52.96                                       |
| Math-Hard (Exact-Match, 4-shot)                         | 21.15                            | 20.82                                       |
| GPQA (Acc-Norm, 0-shot)                                 | 2.97                             | 1.99                                        |
| MUSR (Acc-Norm, 0-shot)                                 | 9.57                             | 10.93                                       |
| MMLU-Pro (Acc, 5-shot)                                  | 52.00                            | 51.62                                       |
| **Average Score**                                       | **30.18**                        | **29.80**                                   |
| **Recovery**                                            | **100.00**                       | **98.74**                                   |

