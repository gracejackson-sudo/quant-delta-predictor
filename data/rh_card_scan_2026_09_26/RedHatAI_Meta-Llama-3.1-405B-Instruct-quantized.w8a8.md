---
tags:
- int8
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

# Meta-Llama-3.1-405B-Instruct-quantized.w8a8

## Model Overview
- **Model Architecture:** Meta-Llama-3
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 8/19/2024
- **Version:** 1.0
- **License(s):** Llama3.1
- **Model Developers:** Neural Magic

This model is a quantized version of [Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model, including multiple-choice, math reasoning, and open-ended text generation.
Meta-Llama-3.1-405B-Instruct-FP8-dynamic achieves 95.8% recovery for the Arena-Hard evaluation, 99.3% for OpenLLM v1 (using Meta's prompting when available), 98.4% for OpenLLM v2, 100.1% for HumanEval pass@1, and 100.4% for HumanEval+ pass@1.

### Model Optimizations

This model was obtained by quantizing the weights of [Meta-Llama-3.1-405B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between INT8 and floating point representations for each output channel dimension.
Linear scaling factors are computed via by minimizing the mean squarred error (MSE).
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between INT8 and floating point representations.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.
GPTQ used a 1% damping factor and 512 sequences sequences taken from Neural Magic's [LLM compression calibration dataset](https://huggingface.co/datasets/neuralmagic/LLM_compression_calibration).


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8"
number_gpus = 8
max_model_len = 8192

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.


## Creation

This model was created by using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as presented in the code snipet below (using 8 A100 80GB GPUs).

```python
from transformers import AutoTokenizer
from datasets import load_dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers.compression.helpers import custom_offload_device_map

model_id = "meta-llama/Meta-Llama-3.1-405B-Instruct"

num_samples = 512
max_seq_len = 4096
num_gpus = 8
max_memory_per_gpu = "20GB"

tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_fn(example):
  return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.shuffle().select(range(num_samples))
ds = ds.map(preprocess_fn)

recipe = GPTQModifier(
  sequential=True,
  targets="Linear",
  scheme="W8A8",
  ignore=["lm_head"],
  dampening_frac=0.01,
  observer="mse"
)

device_map = custom_offload_device_map(
  model_id, 
  max_memory_per_gpu=max_memory_per_gpu,
  num_gpus=num_gpus, 
  torch_dtype="auto",
)

model = SparseAutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="auto",
)

oneshot(
  model=model,
  dataset=ds,
  recipe=recipe,
  max_seq_length=max_seq_len,
  num_calibration_samples=num_samples,
)

model.save_pretrained("Meta-Llama-3.1-405B-Instruct-quantized.w8a8")
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

**Note:** Results have been updated after Meta modified the chat template.

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Meta-Llama-3.1-405B-Instruct </strong>
   </td>
   <td><strong>Meta-Llama-3.1-405B-Instruct-quantized.w8a8 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Arena Hard</strong>
   </td>
   <td>67.4 (67.3 / 67.5)
   </td>
   <td>64.6 (64.3 / 64.8)
   </td>
   <td>95.8%
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
   <td>87.1
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (0-shot)
   </td>
   <td>95.0
   </td>
   <td>94.4
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (CoT, 8-shot, strict-match)
   </td>
   <td>96.4
   </td>
   <td>95.5
   </td>
   <td>99.0%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td> 
   <td>88.3
   </td>
   <td>88.2
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>87.2
   </td>
   <td>86.1
   </td>
   <td>98.7%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot)
   </td>
   <td>64.6
   </td>
   <td>64.4
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>86.8</strong>
   </td>
   <td><strong>86.2</strong>
   </td>
   <td><strong>99.3%</strong>
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
   <td>58.4
   </td>
   <td>97.8%
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>87.7
   </td>
   <td>87.0
   </td>
   <td>99.2%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>67.0
   </td>
   <td>66.7
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5 (4-shot)
   </td>
   <td>39.0
   </td>
   <td>35.8
   </td>
   <td>91.9%
   </td>
  </tr>
  <tr>
   <td>GPQA (0-shot)
   </td>
   <td>19.5
   </td>
   <td>20.4
   </td>
   <td>104.5%
   </td>
  </tr>
  <tr>
   <td>MuSR (0-shot)
   </td>
   <td>19.5
   </td>
   <td>19.2
   </td>
   <td>98.8%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>48.7</strong>
   </td>
   <td><strong>47.9</strong>
   </td>
   <td><strong>98.4%</strong>
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
   <td>86.9
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>80.1
   </td>
   <td>80.4
   </td>
   <td>100.4%
   </td>
  </tr>
</table>

### Reproduction

The results were obtained using the following commands:

#### MMLU
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=8 \
  --tasks mmlu_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU-CoT
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8",dtype=auto,max_model_len=4064,max_gen_toks=1024,tensor_parallel_size=8 \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8",dtype=auto,max_model_len=3940,max_gen_toks=100,tensor_parallel_size=8 \
  --tasks arc_challenge_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=8 \
  --tasks gsm8k_cot_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 8 \
  --batch_size auto
```

#### Hellaswag
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks hellaswag \
  --num_fewshot 10 \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks winogrande \
  --num_fewshot 5 \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=8 \
  --tasks truthfulqa \
  --num_fewshot 0 \
  --batch_size auto
```

#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8",dtype=auto,max_model_len=4096,tensor_parallel_size=8,enable_chunked_prefill=True \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval and HumanEval+
##### Generation
```
python3 codegen/generate.py \
  --model neuralmagic/Meta-Llama-3.1-405B-Instruct-quantized.w8a8 \
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
  humaneval/neuralmagic--Meta-Llama-3.1-405B-Instruct-quantized.w8a8_vllm_temp_0.2
```
##### Evaluation
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/neuralmagic--Meta-Llama-3.1-405B-Instruct-quantized.w8a8_vllm_temp_0.2-sanitized
```
