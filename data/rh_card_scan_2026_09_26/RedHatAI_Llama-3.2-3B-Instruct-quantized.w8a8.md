---
license: llama3.2
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
tags:
- llama
- llama-3
- neuralmagic
- llmcompressor
base_model: meta-llama/Llama-3.2-3B-Instruct
---

# Llama-3.2-3B-Instruct-quantized.w8a8

## Model Overview
- **Model Architecture:** Llama-3
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** INT8
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 9/25/2024
- **Version:** 1.0
- **License(s):** Llama3.2
- **Model Developers:** Neural Magic

Quantized version of [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
It achieves scores within 1% of the scores of the unquantized model for MMLU, ARC-Challenge, GSM-8k, Hellaswag, Winogrande and TruthfulQA.

### Model Optimizations

This model was obtained by quantizing the weights of [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between INT8 and floating point representations for each output channel dimension.
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between INT8 and floating point representations.
The [SmoothQuant](https://arxiv.org/abs/2211.10438) algorithm is used to alleviate outliers in the activations, whereas rhe [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization.
Both algorithms are implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.
GPTQ used a 1% damping factor and 512 sequences sequences taken from Neural Magic's [LLM compression calibration dataset](https://huggingface.co/datasets/neuralmagic/LLM_compression_calibration).

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8"
number_gpus = 1
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
from transformers import AutoTokenizer
from datasets import load_dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, SmoothQuantModifier

model_id = "meta-llama/Llama-3.2-3B-Instruct"

num_samples = 512
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_fn(example):
  return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.shuffle().select(range(num_samples))
ds = ds.map(preprocess_fn)

recipe = [
  SmoothQuantModifier(
    smoothing_strength=0.7,
    mappings=[
      [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
      [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
      [["re:.*down_proj"], "re:.*up_proj"],
    ],
  ),
  GPTQModifier(
    sequential=True,
    targets="Linear",
    scheme="W8A8",
    ignore=["lm_head"],
    dampening_frac=0.01,
  )
]

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

model.save_pretrained("Llama-3.2-3B-Instruct-quantized.w8a8")
```


## Evaluation

The model was evaluated on MMLU, ARC-Challenge, GSM-8K, Hellaswag, Winogrande and TruthfulQA.
Evaluation was conducted using the Neural Magic fork of [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness/tree/llama_3.1_instruct) (branch llama_3.1_instruct) and the [vLLM](https://docs.vllm.ai/en/stable/) engine.
This version of the lm-evaluation-harness includes versions of MMLU, ARC-Challenge and GSM-8K that match the prompting style of [Meta-Llama-3.1-Instruct-evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals).

### Accuracy

#### Open LLM Leaderboard evaluation scores
<table>
  <tr>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Llama-3.2-3B-Instruct </strong>
   </td>
   <td><strong>Llama-3.2-3B-Instruct-quantized.w8a8 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td>MMLU (5-shot)
   </td>
   <td>62.98
   </td>
   <td>62.75
   </td>
   <td>99.6%
   </td>
  </tr>
  <tr>
   <td>MMLU (CoT, 0-shot)
   </td>
   <td>65.40
   </td>
   <td>65.05
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>ARC Challenge (0-shot)
   </td>
   <td>77.13
   </td>
   <td>76.45
   </td>
   <td>99.1%
   </td>
  </tr>
  <tr>
   <td>GSM-8K (CoT, 8-shot, strict-match)
   </td>
   <td>77.94
   </td>
   <td>77.56
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>73.62
   </td>
   <td>73.63
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>71.11
   </td>
   <td>71.90
   </td>
   <td>101.1%
   </td>
  </tr>
  <tr>
   <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>51.47
   </td>
   <td>51.38
   </td>
   <td>98.4%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>68.52</strong>
   </td>
   <td><strong>68.39</strong>
   </td>
   <td><strong>99.81%</strong>
   </td>
  </tr>
</table>

### Reproduction

The results were obtained using the following commands:

#### MMLU
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
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
  --model_args pretrained="neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4064,max_gen_toks=1024,tensor_parallel_size=1 \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=3940,max_gen_toks=100,tensor_parallel_size=1 \
  --tasks arc_challenge_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
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
  --model_args pretrained="neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks hellaswag \
  --num_fewshot 10 \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks winogrande \
  --num_fewshot 5 \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Llama-3.2-3B-Instruct-quantized.w8a8",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks truthfulqa \
  --num_fewshot 0 \
  --batch_size auto
```
