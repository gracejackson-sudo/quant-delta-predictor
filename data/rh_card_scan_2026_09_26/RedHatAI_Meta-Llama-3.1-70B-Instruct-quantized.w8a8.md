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
base_model: meta-llama/Meta-Llama-3.1-70B-Instruct
---

# Meta-Llama-3.1-70B-Instruct-quantized.w8a8

## Model Overview
- **Model Architecture:** LlamaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 7/29/2024
- **Version:** 1.0
- **License(s):** Llama3.1
- **Model Developers:** Neural Magic

This model is a quantized version of [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model, including multiple-choice, math reasoning, and open-ended text generation.
Meta-Llama-3.1-70B-Instruct-quantized.w8a8 achieves 98.8% recovery for the Arena-Hard evaluation, 99.9% for OpenLLM v1 (using Meta's prompting when available), 100.0% for OpenLLM v2, 98.7% for HumanEval pass@1, and 98.9% for HumanEval+ pass@1.

### Model Optimizations

This model was obtained by quantizing the weights of [Meta-Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-70B-Instruct) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between INT8 and floating point representations for each output channel dimension.
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between INT8 and floating point representations.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.
GPTQ used a 10% damping factor and 256 sequences sequences taken from Neural Magic's [LLM compression calibration dataset](https://huggingface.co/datasets/neuralmagic/LLM_compression_calibration).


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8"
number_gpus = 2
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

This model was created by using the [llm-compressor](https://github.com/vllm-project/llm-compressor) library as presented in the code snipet below.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

model_stub = "meta-llama/Meta-Llama-3.1-70B-Instruct"
model_name = model_stub.split("/")[-1]

num_samples = 256
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_fn(example):
  return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.map(preprocess_fn)

recipe = GPTQModifier(
    targets="Linear",
    scheme="W8A8",
    ignore=["lm_head"],
    dampening_frac=0.1,
)

device_map = calculate_offload_device_map(
    model_stub,
    reserve_for_hessians=True,
    num_gpus=2,
    torch_dtype="auto",
)

model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    device_map="auto",
    dtype="auto",
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=max_seq_len,
    num_calibration_samples=num_samples,
)


save_path = model_name + "-quantized.w8a8
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")

```


## Evaluation

This model was evaluated on the well-known Arena-Hard, OpenLLM v1, OpenLLM v2, HumanEval, and HumanEval+ benchmarks.
In all cases, model outputs were generated with the [vLLM](https://docs.vllm.ai/en/stable/) engine.

Arena-Hard evaluations were conducted using the [Arena-Hard-Auto](https://github.com/lmarena/arena-hard-auto) repository.
The model generated a single answer for each prompt form Arena-Hard, and each answer was judged twice by GPT-4.
We report below the scores obtained in each judgement and the average.

OpenLLM v1 and v2 evaluations were conducted using Neural Magic's fork of [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness/tree/llama_3.1_instruct) (branch llama_3.1_instruct).
This version of the lm-evaluation-harness includes versions of MMLU, ARC-Challenge and GSM-8K that match the prompting style of [Meta-Llama-3.1-Instruct-evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-70B-Instruct-evals) and a few fixes to OpenLLM v2 tasks.

HumanEval and HumanEval+ evaluations were conducted using Neural Magic's fork of the [EvalPlus](https://github.com/neuralmagic/evalplus) repository.

Detailed model outputs are available as HuggingFace datasets for [Arena-Hard](https://huggingface.co/datasets/neuralmagic/quantized-llama-3.1-arena-hard-evals), [OpenLLM v2](https://huggingface.co/datasets/neuralmagic/quantized-llama-3.1-leaderboard-v2-evals), and [HumanEval](https://huggingface.co/datasets/neuralmagic/quantized-llama-3.1-humaneval-evals).

**Note:** Results have been updated after Meta modified the chat template.

### Accuracy

<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Meta-Llama-3.1-70B-Instruct </strong>
   </td>
   <td><strong>Meta-Llama-3.1-70B-Instruct-quantized.w8a8 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Arena Hard</strong>
   </td>
   <td>57.0 (55.8 / 58.2)
   </td>
   <td>56.3 (56.0 / 56.6)
   </td>
   <td>98.8%
   </td>
  </tr>
  <tr>
   <td><strong>OpenLLM v1</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>83.9
   </td>
   <td>83.7
   </td>
   <td>99.7%
   </td>
  </tr>
  <tr>
   <td>MMLU (CoT, 0-shot)
   </td>
   <td>86.2
   </td>
   <td>85.8
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (0-shot)
   </td>
   <td>93.3
   </td>
   <td>93.1
   </td>
   <td>99.7%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (CoT, 8-shot, strict-match)
   </td>
   <td>95.4
   </td>
   <td>94.2
   </td>
   <td>98.8%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>86.7
   </td>
   <td>86.7
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>85.3
   </td>
   <td>85.1
   </td>
   <td>100.1%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>60.7
   </td>
   <td>61.4
   </td>
   <td>101.3%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>84.5</strong>
   </td>
   <td><strong>84.3</strong>
   </td>
   <td><strong>99.9%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>OpenLLM v2</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU-Pro (5-shot)
   </td>
   <td>48.1
   </td>
   <td>47.1
   </td>
   <td>97.9%
   </td>
  </tr>
  <tr>
   <td>IFEval (0-shot)
   </td>
   <td>86.4
   </td>
   <td>86.6
   </td>
   <td>100.2%
   </td>
  </tr>
  <tr>
   <td>BBH (3-shot)
   </td>
   <td>55.8
   </td>
   <td>55.2
   </td>
   <td>98.9%
   </td>
  </tr>
  <tr>
   <td>Math-|v|-5 (4-shot)
   </td>
   <td>26.1
   </td>
   <td>23.9
   </td>
   <td>91.8%
   </td>
  </tr>
  <tr>
   <td>GPQA (0-shot)
   </td>
   <td>15.4
   </td>
   <td>13.6
   </td>
   <td>88.4%
   </td>
  </tr>
  <tr>
   <td>MuSR (0-shot)
   </td>
   <td>18.2
   </td>
   <td>16.8
   </td>
   <td>92.6%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>41.7</strong>
   </td>
   <td><strong>40.5</strong>
   </td>
   <td><strong>97.3%</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Coding</strong>
   </td>
  </tr>
  <tr>
   <td>HumanEval pass@1
   </td>
   <td>79.7
   </td>
   <td>78.7
   </td>
   <td>98.7%
   </td>
  </tr>
  <tr>
   <td>HumanEval+ pass@1
   </td>
   <td>74.8
   </td>
   <td>74.0
   </td>
   <td>98.9%
   </td>
  </tr>
</table>

### Reproduction

The results were obtained using the following commands:

#### MMLU
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
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
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",dtype=auto,max_model_len=4064,max_gen_toks=1024,tensor_parallel_size=1 \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",dtype=auto,max_model_len=3940,max_gen_toks=100,tensor_parallel_size=1 \
  --tasks arc_challenge_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
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
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks hellaswag \
  --num_fewshot 10 \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks winogrande \
  --num_fewshot 5 \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks truthfulqa \
  --num_fewshot 0 \
  --batch_size auto
```

#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8",dtype=auto,max_model_len=4096,tensor_parallel_size=1",enable_chunked_prefill=True \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval and HumanEval+
##### Generation
```
python3 codegen/generate.py \
  --model neuralmagic/Meta-Llama-3.1-70B-Instruct-quantized.w8a8 \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval
```
##### Sanitization
```
python3 evalplus/sanitize.py \
  humaneval/neuralmagic--Meta-Llama-3.1-70B-Instruct-quantized.w8a8_vllm_temp_0.2
```
##### Evaluation
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/neuralmagic--Meta-Llama-3.1-70B-Instruct-quantized.w8a8_vllm_temp_0.2-sanitized
```