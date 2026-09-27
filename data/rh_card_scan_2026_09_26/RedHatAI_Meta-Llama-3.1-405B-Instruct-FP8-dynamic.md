---
tags:
- fp8
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
license: llama3.1
base_model: meta-llama/Meta-Llama-3.1-405B-Instruct
---

# Meta-Llama-3.1-405B-Instruct-FP8-dynamic

## Model Overview
- **Model Architecture:** Meta-Llama-3.1
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. Similarly to [Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 8/22/2024
- **Version:** 1.1
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

This model is a quantized version of [Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model, including multiple-choice, math reasoning, and open-ended text generation.
Meta-Llama-3.1-405B-Instruct-FP8-dynamic achieves 99.0% recovery for the Arena-Hard evaluation, 100.0% for OpenLLM v1 (using Meta's prompting when available), 99.9% for OpenLLM v2, 100.2% for HumanEval pass@1, and 101.1% for HumanEval+ pass@1.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct) to FP8 data type, ready for inference with vLLM built from source.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%. In particular, this model can now be loaded and evaluated with a single node of 8xH100 GPUs, as opposed to multiple nodes.

Only the weights and activations of the linear operators within transformers blocks are quantized. Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the FP8 representations of the quantized weights and activations. Activations are also quantized on a per-token dynamic basis.
[LLM Compressor](https://github.com/vllm-project/llm-compressor) is used for quantization.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic"
number_gpus = 8

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=4096)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [LLM Compressor with calibration samples from UltraChat](https://github.com/vllm-project/llm-compressor/blob/sa/big_model_support/examples/big_model_offloading/big_model_w8a8_calibrate.py), as presented in the code snipet below.

```python
import torch

from transformers import AutoTokenizer

from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.transformers.compression.helpers import (  # noqa
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
                        strategy: channel
                        dynamic: false
                        symmetric: true
                    input_activations:
                        num_bits: 8
                        type: float
                        strategy: token
                        dynamic: true
                        symmetric: true
                    targets: ["Linear"]
"""

model_stub = "meta-llama/Meta-Llama-3.1-405B-Instruct"
model_name = model_stub.split("/")[-1]

device_map = calculate_offload_device_map(
    model_stub, reserve_for_hessians=False, num_gpus=8, torch_dtype="auto"
)

model = SparseAutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype="auto", device_map=device_map
)

output_dir = f"./{model_name}-FP8-dynamic"

oneshot(
    model=model,
    recipe=recipe,
    output_dir=output_dir,
    save_compressed=True,
    tokenizer=AutoTokenizer.from_pretrained(model_stub),
)
```

## Evaluation

This model was evaluated on the well-known Arena-Hard, OpenLLM v1, OpenLLM v2, HumanEval, and HumanEval+ benchmarks.
In all cases, model outputs were generated with the [vLLM](https://docs.vllm.ai/en/stable/) engine.

Arena-Hard evaluations were conducted using the [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto) repository.
The model generated a single answer for each prompt form Arena-Hard, and each answer was judged twice by GPT-4.
We report below the scores obtained in each judgement and the average.

OpenLLM v1 and v2 evaluations were conducted using Neural Magic's fork of [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness/tree/llama_3.1_instruct) (branch llama_3.1_instruct).
This version of the lm-evaluation-harness includes versions of MMLU, ARC-Challenge and GSM-8K that match the prompting style of [Meta-Llama-3.1-Instruct-evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-405B-Instruct-evals) and a few fixes to OpenLLM v2 tasks.

HumanEval and HumanEval+ evaluations were conducted using Neural Magic's fork of the [EvalPlus](https://github.com/neuralmagic/evalplus) repository.

Detailed model outputs are available as HuggingFace datasets for [Arena-Hard](https://huggingface.co/datasets/neuralmagic/quantized-llama-3.1-arena-hard-evals), [OpenLLM v2](https://huggingface.co/datasets/neuralmagic/quantized-llama-3.1-leaderboard-v2-evals), and [HumanEval](https://huggingface.co/datasets/neuralmagic/quantized-llama-3.1-humaneval-evals).

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Meta-Llama-3.1-405B-Instruct </strong>
   </td>
   <td><strong>Meta-Llama-3.1-405B-Instruct-FP8-dynamic (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Arena Hard</strong>
   </td>
   <td>67.4 (67.3 / 67.5)
   </td>
   <td>66.7 (66.7 / 66.6)
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td><strong>OpenLLM v1</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>87.4
   </td>
   <td>87.5
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>MMLU-cot (0-shot)
   </td>
   <td>88.1
   </td>
   <td>88.1
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (0-shot)
   </td>
   <td>95.0
   </td>
   <td>95.0
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>GSM-8K-cot (8-shot, strict-match)
   </td>
   <td>96.0
   </td>
   <td>95.8
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>88.5
   </td>
   <td>88.5
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>87.2
   </td>
   <td>88.0
   </td>
   <td>100.9%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>65.3
   </td>
   <td>65.3
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>86.8</strong>
   </td>
   <td><strong>86.9</strong>
   </td>
   <td><strong>100.0%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>OpenLLM v2</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>59.7
   </td>
   <td>59.4
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>87.7
   </td>
   <td>86.8
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>67.0
   </td>
   <td>67.1
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>Math-|v|-5 (4-shot)
   </td>
   <td>39.0
   </td>
   <td>38.8
   </td>
   <td>99.7%
   </td>
  </tr>
  <tr>
   <td>GPQA (0-shot)
   </td>
   <td>19.5
   </td>
   <td>19.0
   </td>
   <td>97.4%
   </td>
  </tr>
  <tr>
   <td>MuSR (0-shot)
   </td>
   <td>19.5
   </td>
   <td>20.8
   </td>
   <td>106.9%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>48.7</strong>
   </td>
   <td><strong>48.7</strong>
   </td>
   <td><strong>99.9%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Coding</strong>
   </td>
  </tr>
  <tr>
   <td>HumanEval pass@1
   </td>
   <td>86.8
   </td>
   <td>87.0
   </td>
   <td>100.2%
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>80.1
   </td>
   <td>81.0
   </td>
   <td>101.1%
   </td>
  </tr>
</table>


### Reproduction

The results were obtained using the following commands:

#### MMLU
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,max_gen_toks=10,tensor_parallel_size=8 \
  --tasks mmlu_llama_3.1_instruct \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU-cot
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=8 \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks arc_challenge_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks gsm8k_cot_llama_3.1_instruct \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --num_fewshot 8 \
  --batch_size auto
```

#### Hellaswag
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks hellaswag \
  --num_fewshot 10 \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks winogrande \
  --num_fewshot 5 \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks truthfulqa \
  --num_fewshot 0 \
  --batch_size auto
```

#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic",dtype=auto,max_model_len=4096,tensor_parallel_size=8,enable_chunked_prefill=True \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval and HumanEval+
##### Generation
```
python3 codegen/generate.py \
  --model neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8-dynamic \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval \
  --tp 8
```
##### Sanitization
```
python3 evalplus/sanitize.py \
  humaneval/neuralmagic--Meta-Llama-3.1-405B-Instruct-FP8-dynamic_vllm_temp_0.2
```
##### Evaluation
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/neuralmagic--Meta-Llama-3.1-405B-Instruct-FP8-dynamic_vllm_temp_0.2-sanitized
```
