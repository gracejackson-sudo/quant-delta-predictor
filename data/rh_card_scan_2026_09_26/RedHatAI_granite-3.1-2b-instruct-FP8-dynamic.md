---
tags:
- fp8
- vllm
license: apache-2.0
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
  - en
base_model: ibm-granite/granite-3.1-2b-instruct
library_name: transformers
---

# granite-3.1-2b-instruct-FP8-dynamic

## Model Overview
- **Model Architecture:** granite-3.1-2b-instruct
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 1/8/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [ibm-granite/granite-3.1-2b-instruct](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct).
It achieves an average score of 61.84 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 61.98.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [ibm-granite/granite-3.1-2b-instruct](https://huggingface.co/ibm-granite/granite-3.1-2b-instruct) to FP8 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%. Only the weights and activations of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 1
model_name = "neuralmagic/granite-3.1-2b-instruct-FP8-dynamic"
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

<details>
  <summary>Model Creation Code</summary>
  
```bash
python quantize.py --model_id ibm-granite/granite-3.1-2b-instruct --save_path "output_dir/"
```

```python
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import oneshot
import os

def main():
    parser = argparse.ArgumentParser(description='Quantize a transformer model to FP8')
    parser.add_argument('--model_id', type=str, required=True,
                        help='The model ID from HuggingFace (e.g., "meta-llama/Meta-Llama-3-2b-Instruct")')
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
</details>

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard), OpenLLM Leaderboard [V2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/) and on [HumanEval](https://github.com/neuralmagic/evalplus), using the following commands:

<details>
<summary>Evaluation Commands</summary>
  
OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/granite-3.1-2b-instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
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
  --model_args pretrained="neuralmagic/granite-3.1-2b-instruct-FP8-dynamic",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks leaderboard \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```

#### HumanEval
##### Generation
```
python3 codegen/generate.py \
  --model neuralmagic/granite-3.1-2b-instruct-FP8-dynamic \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval
```
##### Sanitization
```
python3 evalplus/sanitize.py \
  humaneval/neuralmagic--granite-3.1-2b-instruct-FP8-dynamic_vllm_temp_0.2
```
##### Evaluation
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/neuralmagic--granite-3.1-2b-instruct-FP8-dynamic_vllm_temp_0.2-sanitized
```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>ibm-granite/granite-3.1-2b-instruct</th>
      <th>neuralmagic/granite-3.1-2b-instruct-FP8-dynamic</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <!-- OpenLLM V1 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>55.63</td>
      <td>55.03</td>
      <td>98.92</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>60.96</td>
      <td>61.49</td>
      <td>100.87</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>75.21</td>
      <td>75.26</td>
      <td>100.07</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>54.38</td>
      <td>54.24</td>
      <td>99.74</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>55.93</td>
      <td>55.42</td>
      <td>99.09</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>69.67</td>
      <td>69.61</td>
      <td>99.91</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>61.98</b></td>
      <td><b>61.84</b></td>
      <td><b>99.78</b></td>
    </tr>
    <!-- OpenLLM Leaderboard V2 -->
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>67.99</td>
      <td>66.79</td>
      <td>98.24</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>44.11</td>
      <td>44.24</td>
      <td>100.29</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>8.66</td>
      <td>7.89</td>
      <td>91.12</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>28.30</td>
      <td>26.90</td>
      <td>95.06</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>35.12</td>
      <td>35.12</td>
      <td>100.00</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>26.87</td>
      <td>28.33</td>
      <td>105.42</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>35.17</b></td>
      <td><b>34.88</b></td>
      <td><b>99.16</b></td>
    </tr>
    <!-- HumanEval -->
    <tr>
      <td rowspan="2"><b>Coding</b></td>
      <td>HumanEval Pass@1</td>
      <td>53.40</td>
      <td>54.90</td>
      <td><b>102.81</b></td>
    </tr>
  </tbody>
</table>



## Inference Performance


This model achieves up to 1.2x speedup in single-stream deployment on L40 GPUs.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.6.6.post1, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/granite-3.1-2b-instruct-FP8-dynamic --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
```

</details>

### Single-stream performance (measured with vLLM version 0.6.6.post1)
<table>
  <tr>
    <td></td>
    <td></td>
    <td></td>
    <th style="text-align: center;" colspan="7" >Latency (s)</th>
  </tr>
  <tr>
    <th>GPU class</th>
    <th>Model</th>
    <th>Speedup</th>
    <th>Code Completion<br>prefill: 256 tokens<br>decode: 1024 tokens</th>
    <th>Docstring Generation<br>prefill: 768 tokens<br>decode: 128 tokens</th>
    <th>Code Fixing<br>prefill: 1024 tokens<br>decode: 1024 tokens</th>
    <th>RAG<br>prefill: 1024 tokens<br>decode: 128 tokens</th>
    <th>Instruction Following<br>prefill: 256 tokens<br>decode: 128 tokens</th>
    <th>Multi-turn Chat<br>prefill: 512 tokens<br>decode: 256 tokens</th>
    <th>Large Summarization<br>prefill: 4096 tokens<br>decode: 512 tokens</th>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >L40</td>
    <td>granite-3.1-2b-instruct</td>
    <td></td>
    <td>9.3</td>
    <td>1.2</td>
    <td>9.4</td>
    <td>1.2</td>
    <td>1.2</td>
    <td>2.3</td>
    <td>5.0</td>
  </tr>
  <tr>
    <td>granite-3.1-2b-instruct-FP8-dynamic<br>(this model)</td>
    <td>1.26</td>
    <td>7.3</td>
    <td>0.9</td>
    <td>7.4</td>
    <td>1.0</td>
    <td>0.9</td>
    <td>1.8</td>
    <td>4.1</td>
  </tr>
  <tr>
    <td>granite-3.1-2b-instruct-quantized.w4a16</td>
    <td>1.88</td>
    <td>4.8</td>
    <td>0.6</td>
    <td>4.9</td>
    <td>0.6</td>
    <td>0.6</td>
    <td>1.2</td>
    <td>2.8</td>
  </tr>
</table>