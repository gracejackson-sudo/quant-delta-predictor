---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3-4B-Thinking-2507
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- INT8
---

# Qwen3-4B-Thinking-2507.w8a8

## Model Overview
- **Model Architecture:** Qwen3ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
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

This model was obtained by quantizing the weights of [Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, whereas activations are quantized with a symmetric dynamic per-token scheme.
A combination of the [SmoothQuant](https://arxiv.org/abs/2211.10438) and [GPTQ](https://arxiv.org/abs/2210.17323) algorithms is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Qwen3-4B-Thinking-2507.w8a8"
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
  from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
  from llmcompressor.transformers import oneshot
  from transformers import AutoModelForCausalLM, AutoTokenizer
  
  # Load model
  model_stub = "Qwen/Qwen3-4B-Thinking"
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
  recipe = [
    SmoothQuantModifier(
      smoothing_strength=0.9,
      mappings=[
          [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
      ],
    ),
    GPTQModifier(
          ignore: ["lm_head"]
          config_groups={"group_0": {"targets": ["Linear"], "weights": { "num_bits": 4, "type": int, "strategy": "group", "group_size": 128, "symmetric": true, "dynamic": false, "observer": "mse" } } },
          dampening_frac=0.1,
      )
  ]


  # Apply quantization
  oneshot(
      model=model,
      dataset=ds, 
      recipe=recipe,
      max_seq_length=max_seq_len,
      num_calibration_samples=num_samples,
  )
  
  # Save to disk in compressed-tensors format
  save_path = model_name + "-quantized.w8a8"
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
    vllm serve RedHatAI/Qwen3-4B-Thinking-2507.w8a8 --max-model-len 262144 --reasoning-parser deepseek_r1
    ```

  **lm-evaluation-harness**
  ```
  lm_eval --model local-chat-completions \
    --tasks mmlu_pro_chat \
    --model_args "model=RedHatAI/Qwen3-4B-Thinking-2507.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks ifeval \
    --model_args "model=RedHatAI/Qwen3-4B-Thinking-2507.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000
  ```

  ```
  lm_eval --model local-chat-completions \
    --tasks gsm8k_platinum_cot_llama \
    --model_args "model=RedHatAI/Qwen3-4B-Thinking-2507.w8a8,max_length=262144,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
    --num_fewshot 0 \
    --apply_chat_template \
    --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000
  ```

  **lighteval**
  
  lighteval_model_arguments.yaml
  ```yaml 
  model_parameters:
    model_name: RedHatAI/Qwen3-4B-Thinking-2507.w8a8
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


| Benchmark | Qwen3-4B Thinking | Qwen3-4B Thinking.w8a8 (this model) | Recovery (%) |
|--------|-------------|-------------------|--------------|
| GSM8k Platinum (0-shot) | 95.26 | 95.78 | 100.55 |
| MMLU-Pro (0-shot) | 73.63 | 72.71 | 98.74 |
| IfEval (0-shot) |90.17 | 89.81| 99.60 |
| AIME 2025 | 73.75 | 71.67 | 97.18 |
| GPQA diamond | 68.94 | 66.67 | 96.70 |
| Math 500 | 89.45 |  89.28| 99.80 |
