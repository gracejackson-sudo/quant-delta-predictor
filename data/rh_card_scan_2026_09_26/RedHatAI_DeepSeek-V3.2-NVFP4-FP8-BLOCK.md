---
license: mit
base_model:
- deepseek-ai/DeepSeek-V3.2
---

Compressed version of `deepseek-ai/DeepSeek-V3.2`, with MLP/expert layers compressed to NVFP4 format and attention modules left in FP8_BLOCK format.

## Creation

Creation script available at [llm-compressor/examples/disk_offloading/deepseek_v32_example.py](https://github.com/vllm-project/llm-compressor/blob/893081332ab3402017a159bae7efba539d5765f7/examples/disk_offloading/deepseek_v32_example.py).


## Running
To run on vllm, follow vllm & DeepGemm install instructions at https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html

Serve on 4 B200s:
```
vllm serve "RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK" --tensor-parallel-size 2 --data-parallel-size=2 --enable-expert-parallel --tokenizer-mode deepseek_v32 --reasoning-parser deepseek_v3
```

## Evals

### gsm8k
This checkpoint achieves over 99% recovery of the original `deepseek-ai/DeepSeek-V3.2` checkpoint's [0.9553 score on gsm8k](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html#gsm8k):
```
lm_eval --model local-completions --model_args "model=RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK,base_url=http://0.0.0.0:8000/v1/completions,max_length=8192,tokenized_requests=False,tokenizer_backend=None,num_concurrent=32" --tasks gsm8k --num_fewshot 5
Requesting API: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [04:01<00:00,  5.46it/s]
local-completions ({'model': 'RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK', 'base_url': 'http://0.0.0.0:8000/v1/completions', 'max_length': 8192, 'tokenized_requests': False, 'tokenizer_backend': None, 'num_concurrent': 32}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.953|±  |0.0058|
|     |       |strict-match    |     5|exact_match|↑  |0.953|±  |0.0058|
```

Checkpoint outperforms W4A16 (unpublished):
```
local-completions ({'model': '/mnt/data/brian-dellabetta/DeepSeek-V3.2-W4A16', 'base_url': 'http://0.0.0.0:8000/v1/completions', 'max_length': 4096, 'tokenized_requests': False, 'tokenizer_backend': None, 'num_concurrent': 4}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9469|±  |0.0062|
|     |       |strict-match    |     5|exact_match|↑  |0.9469|±  |0.0062|
```

### aime
This checkpoint achieves roughly 86% recovery of the original checkpoint's [0.933 score on aime25](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html#aime25)
```
lm_eval --model local-chat-completions --model_args "model=RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK,base_url=http://0.0.0.0:8000/v1/chat/completions,tokenized_requests=False,tokenizer_backend=None,num_concurrent=20,timeout=5000,max_length=72768" --tasks aime25 --apply_chat_template --gen_kwargs '{"temperature":1.0,"max_gen_toks":65536,"top_p":0.95,"chat_template_kwargs":{"thinking":true}}' --log_samples --output_path "aime25_ds32"
local-chat-completions ({'model': 'RedHatAI/DeepSeek-V3.2-NVFP4-FP8-BLOCK', 'base_url': 'http://0.0.0.0:8000/v1/chat/completions', 'tokenized_requests': False, 'tokenizer_backend': None, 'num_concurrent': 20, 'timeout': 5000, 'max_length': 72768}), gen_kwargs: ({'temperature': 1.0, 'max_gen_toks': 65536, 'top_p': 0.95, 'chat_template_kwargs': {'thinking': True}}), limit: None, num_fewshot: None, batch_size: 1
|Tasks |Version|Filter|n-shot|  Metric   |   |Value|   |Stderr|
|------|------:|------|-----:|-----------|---|----:|---|-----:|
|aime25|      0|none  |     0|exact_match|↑  |  0.8|±  |0.0743|
```
when running without `--tokenizer-mode`/`--reasoning-parser`. Achieves 0.7667 when using vllm serve command listed above. 

Results are consistent with [RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4](https://huggingface.co/RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4#red-hat-ai-evaluations), which similarly shows nearly full recovery with gsm8k but a drop on aime25.