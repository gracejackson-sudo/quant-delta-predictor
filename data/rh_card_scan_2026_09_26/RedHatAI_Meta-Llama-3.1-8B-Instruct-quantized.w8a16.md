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
base_model: meta-llama/Meta-Llama-3.1-8B-Instruct
---

# Meta-Llama-3.1-8B-Instruct-quantized.w8a16

## Model Overview
- **Model Architecture:** Meta-Llama-3
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
- **Intended Use Cases:** Intended for commercial and research use multiple languages. Similarly to [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct), this models is intended for assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 7/23/2024
- **Version:** 1.0
- **License(s):** Llama3.1
- **Model Developers:** Neural Magic

Quantized version of [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct).
It achieves scores within 1% of the scores of the unquantized model for MMLU, ARC-Challenge, GSM-8k, Hellaswag, Winogrande and TruthfulQA.

### Model Optimizations

This model was obtained by quantizing the weights of [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) to INT8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights of the linear operators within transformers blocks are quantized. Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the INT8 and floating point representations of the quantized weights.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library. GPTQ used a 1% damping factor and 256 sequences of 8,192 random tokens.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
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
from datasets import Dataset
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import random

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

num_samples = 256
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_id)

max_token_id = len(tokenizer.get_vocab()) - 1
input_ids = [[random.randint(0, max_token_id) for _ in range(max_seq_len)] for _ in range(num_samples)]
attention_mask = num_samples * [max_seq_len * [1]]
ds = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})

recipe = GPTQModifier(
  targets="Linear",
  scheme="W8A16",
  ignore=["lm_head"],
  dampening_frac=0.01,
)

model = SparseAutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="auto",
  trust_remote_code=True,
)

oneshot(
  model=model,
  dataset=ds,
  recipe=recipe,
  max_seq_length=max_seq_len,
  num_calibration_samples=num_samples,
)
model.save_pretrained("Meta-Llama-3.1-8B-Instruct-quantized.w8a16")
```



## Evaluation

The model was evaluated on MMLU, ARC-Challenge, GSM-8K, Hellaswag, Winogrande and TruthfulQA.
Evaluation was conducted using the Neural Magic fork of [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness/tree/llama_3.1_instruct) (branch llama_3.1_instruct) and the [vLLM](https://docs.vllm.ai/en/stable/) engine.
This version of the lm-evaluation-harness includes versions of MMLU, ARC-Challenge and GSM-8K that match the prompting style of [Meta-Llama-3.1-Instruct-evals](https://huggingface.co/datasets/meta-llama/Meta-Llama-3.1-8B-Instruct-evals).

**Note:** Results have been updated after Meta modified the chat template.

### Accuracy

<table>
  <tr>
   <td><strong>Category</strong>
   </td>
   <td><strong>Benchmark</strong>
   </td>
   <td><strong>Meta-Llama-3.1-8B-Instruct </strong>
   </td>
   <td><strong>Meta-Llama-3.1-8B-Instruct-quantized.w8a16 (this model)</strong>
   </td>
   <td><strong>Recovery</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="5" ><strong>OpenLLM v1</strong>
   </td>
  </tr>
  <tr>
   <td>Hellaswag (10-shot)
   </td>
   <td>80.47
   </td>
   <td>80.48
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>Winogrande (5-shot)
   </td>
   <td>78.06
   </td>
   <td>77.51
   </td>
   <td>99.3%
   </td>
  </tr>
  <tr>
  <td>TruthfulQA (0-shot, mc2)
   </td>
   <td>54.48
   </td>
   <td>54.41
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>74.05</strong>
   </td>
   <td><strong>74.12</strong>
   </td>
   <td><strong>100.1%</strong>
   </td>
  </tr>
  <tr>
   <td rowspan="9" ><strong>Multilingual</strong>
   </td>
   <td>Portuguese MMLU (5-shot)
   </td>
   <td>59.96
   </td>
   <td>59.79
   </td>
   <td>99.8%
   </td>
  </tr>
  <tr>
   <td>Spanish MMLU (5-shot)
   </td>
   <td>60.25
   </td>
   <td>59.92
   </td>
   <td>99.4%
   </td>
  </tr>
  <tr>
   <td>Italian MMLU (5-shot)
   </td>
   <td>59.23
   </td>
   <td>59.25
   </td>
   <td>100.0%
   </td>
  </tr>
  <tr>
   <td>German MMLU (5-shot)
   </td>
   <td>58.63
   </td>
   <td>58.31
   </td>
   <td>99.5%
   </td>
  </tr>
  <tr>
   <td>French MMLU (5-shot)
   </td>
   <td>59.65
   </td>
   <td>59.57
   </td>
   <td>99.9%
   </td>
  </tr>
  <tr>
   <td>Hindi MMLU (5-shot)
   </td>
   <td>50.10
   </td>
   <td>49.97
   </td>
   <td>99.7%
   </td>
  </tr>
  <tr>
   <td>Thai MMLU (5-shot)
   </td>
   <td>49.12
   </td>
   <td>49.09
   </td>
   <td>99.9%
   </td>
  </tr>
</table>

### Reproduction

The results were obtained using the following commands:

#### MMLU
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
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
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=4064,max_gen_toks=1024,tensor_parallel_size=1 \
  --tasks mmlu_cot_0shot_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3940,max_gen_toks=100,tensor_parallel_size=1 \
  --tasks arc_challenge_llama_3.1_instruct \
  --apply_chat_template \
  --num_fewshot 0 \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=4096,max_gen_toks=1024,tensor_parallel_size=1 \
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
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks hellaswag \
  --num_fewshot 10 \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks winogrande \
  --num_fewshot 5 \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1 \
  --tasks truthfulqa \
  --num_fewshot 0 \
  --batch_size auto
```

#### MMLU Portuguese
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
  --tasks mmlu_pt_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU Spanish
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
  --tasks mmlu_es_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU Italian
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
  --tasks mmlu_it_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU German
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
  --tasks mmlu_de_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU French
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
  --tasks mmlu_fr_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU Hindi
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
  --tasks mmlu_hi_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 \
  --batch_size auto
```

#### MMLU Thai
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16",dtype=auto,max_model_len=3850,max_gen_toks=10,tensor_parallel_size=1 \
  --tasks mmlu_th_llama_3.1_instruct \
  --fewshot_as_multiturn \
  --apply_chat_template \
  --num_fewshot 5 \
  --batch_size auto
```