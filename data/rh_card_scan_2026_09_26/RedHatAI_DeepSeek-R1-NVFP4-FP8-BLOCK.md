---
license: mit
base_model: 
- deepseek-ai/DeepSeek-R1
- nvidia/DeepSeek-R1-NVFP4
---

# Model Overview

## Description:
Model created from the `nvidia/DeepSeek-R1-NVFP4` checkpoint by:
- converting all layers targeted by modelopt NVFP4 format to compressed-tensors format
- applying FP8_BLOCK quantization to targeted attention layers

More information at https://github.com/vllm-project/llm-compressor/pull/2228

Runs successfully on 4 B200s:
```python
from vllm import LLM, SamplingParams

prompts = ["The Swiss Alps are", "Brad Marchand is", "The Toronto Maple Leafs are"]

# Create a sampling params object for greedy sampling
sampling_params = SamplingParams(
    temperature=0.80, top_p=0.95, max_tokens=40, min_tokens=10
)
llm = LLM(
    "RedHatAI/DeepSeek-R1-NVFP4-FP8-BLOCK",
    tensor_parallel_size=4,
    max_model_len=4096,
)
output = llm.generate(prompts, sampling_params)
for out in output:
    print(out.outputs[0].text)
```

## Evals
Results from running `vllm serve RedHatAI/DeepSeek-R1-NVFP4-FP8-BLOCK --tensor-parallel-size=4` on 4 B200s, with `python vllm/tests/evals/gsm8k/gsm8k_eval.py --port 8000`:

```
Running GSM8K evaluation: 1319 questions, 5-shot
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [01:49<00:00, 12.09it/s]

Results:
Accuracy: 0.952
Invalid responses: 0.000
Total latency: 109.097 s
Questions per second: 12.090
Total output tokens: 124914
Output tokens per second: 1144.985
```

Compare to results with `nvidia/DeepSeek-R1-NVFP4`
```
Running GSM8K evaluation: 1319 questions, 5-shot
Evaluating: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1319/1319 [01:52<00:00, 11.74it/s]

Results:
Accuracy: 0.954
Invalid responses: 0.000
Total latency: 112.357 s
Questions per second: 11.739
Total output tokens: 128126
Output tokens per second: 1140.344
```