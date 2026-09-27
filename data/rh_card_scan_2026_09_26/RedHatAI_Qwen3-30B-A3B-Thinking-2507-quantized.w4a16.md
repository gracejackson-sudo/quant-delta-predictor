---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-30B-A3B-Thinking-2507
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT4
---

# Qwen3-30B-A3B-Thinking-2507.w4a16

## Model Overview
- **Model Architecture:** Qwen3ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Intended Use Cases:**
  - Reasoning.
  - Function calling.
  - Subject matter experts via fine-tuning.
  - Multilingual instruction following.
  - Translation.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 05/05/2025
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights of [Qwen/Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-group scheme, with group size 128.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-30B-A3B-Thinking-2507.w4a16"
number_gpus = 1
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=256)

messages = [
    {"role": "user", "content": prompt}
]

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


  ```python
  from llmcompressor.modifiers.quantization import GPTQModifier
  from llmcompressor.transformers import oneshot
  from transformers import AutoModelForCausalLM, AutoTokenizer
  
  # Load model
  model_stub = "Qwen/Qwen3-30B-A3B-Thinking"
  model_name = model_stub.split("/")[-1]

  num_samples = 1024
  max_seq_len = 8192

  model = AutoModelForCausalLM.from_pretrained(model_stub)

  tokenizer = AutoTokenizer.from_pretrained(model_stub)

  def preprocess_fn(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}
  
  ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
  ds = ds.map(preprocess_fn)

  # Configure the quantization algorithm and scheme
  recipe = GPTQModifier(
      ignore: ["lm_head"]
      config_groups={"group_0": {"targets": ["Linear"], "weights": { "num_bits": 4, "type": int, "strategy": "group", "group_size": 128, "symmetric": true, "dynamic": false, "observer": "mse" } } },
      dampening_frac=0.01,
  )

  # Apply quantization
  oneshot(
      model=model,
      dataset=ds, 
      recipe=recipe,
      max_seq_length=max_seq_len,
      num_calibration_samples=num_samples,
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-quantized.w4a16"
  model.save_pretrained(save_path)
  tokenizer.save_pretrained(save_path)
  print(f"Model and tokenizer saved to: {save_path}")
  ```
</details>
 



## Evaluation

The model was evaluated on the ifeval, mmlu_pro and gsm8k_platinum  using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), on reasoning tasks using [lighteval](https://github.com/neuralmagic/lighteval/tree/reasoning).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.


<details>
  <summary>Evaluation details</summary>

  Deploy using vllm to create an OpenAI-compatible API endpoint:

- vLLM:
    ```shell
    vllm serve RedHatAI/Qwen3-30B-A3B-Thinking-2507.w4a16 --max-model-len 262144 --reasoning-parser deepseek_r1
    ```

  **lm-evaluation-harness**
  ```
  lm_eval --model local-chat-completions \
    --tasks mmlu_pro_chat \
    --model_args "model=RedHatAI/Qwen3-30B-A3B-Thinking-2507.w4a16,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks ifeval \
    --model_args "model=RedHatAI/Qwen3-30B-A3B-Thinking-2507.w4a16,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks gsm8k_platinum_cot_llama \
    --model_args "model=RedHatAI/Qwen3-30B-A3B-Thinking-2507.w4a16,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-30B-A3B-Thinking-2507.w4a16
    dtype: auto
    gpu_memory_utilization: 0.9
    max_model_length: 40960
    generation_parameters:
      temperature: 0.6
      top_k: 20
      min_p: 0.0
      top_p: 0.95
      max_new_tokens: 32000
  ```

  ```
  lighteval endpoint litellm lighteval_model_arguments.yaml  \
    "aime25|0,math_500|0,gpqa:diamond|0"
  ```


</details>

### Accuracy


| Benchmark | Qwen3-30B-A3B Thinking | Qwen3-30B-A3B Thinking.w4a16 (this model) | Recovery (%) |
|--------|-------------|-------------------|--------------|
| GSM8k Platinum (0-shot) | 85.55 | 72.65 | 84.91 |
| MMLU-Pro (0-shot) | 80.71 | 79.06 | 97.96 |
| IfEval (0-shot) | 90.09 | 90.29 | 100.22 |
| AIME 2025 | 86.67 |  83.33 | 96.15 |
| GPQA diamond | 71.50 | 71.59 | 100.12 |
| Math 500 | 90.75 | 93.10 | 102.59 |
