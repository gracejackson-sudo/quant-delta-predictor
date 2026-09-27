---
license: mit
tags:
- deepseek
- int8
- vllm
- llmcompressor
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
library_name: transformers
---

# DeepSeek-R1-Distill-Qwen-1.5B-quantized.w8a8

## Model Overview
- **Model Architecture:** Qwen2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
  - **Activation quantization:** INT8
- **Release Date:** 2/5/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B).


### Model Optimizations

This model was obtained by quantizing the weights and activations of [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) to INT8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-channel scheme, whereas quantizations are quantized using a symmetric per-token scheme.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

number_gpus = 1
model_name = "neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w8a8"

tokenizer = AutoTokenizer.from_pretrained(model_name)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])
llm = LLM(model=model_name, tensor_parallel_size=number_gpus, trust_remote_code=True)

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


```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.transformers import oneshot

# Load model
model_stub = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = model_stub.split("/")[-1]

num_samples = 2048
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_stub)

model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    device_map="auto",
    torch_dtype="auto",
)

def preprocess_fn(example):
  return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.map(preprocess_fn)

# Configure the quantization algorithm and scheme
recipe = [
    SmoothQuantModifier(smoothing_strength=0.9),
    QuantizationModifier(
        targets="Linear",
        scheme="W8A8",
        ignore=["lm_head"],
        dampening_frac=0.1,
    ),
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
save_path = model_name + "-quantized.w8a8
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model and tokenizer saved to: {save_path}")
```

## Evaluation

The model was evaluated on OpenLLM Leaderboard [V1](https://huggingface.co/spaces/open-llm-leaderboard-old/open_llm_leaderboard) and [V2](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/), using the following commands:

OpenLLM Leaderboard V1:
```
lm_eval \
  --model vllm \
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w8a8",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w8a8",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</th>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-FP8-dynamic</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td rowspan="4"><b>Reasoning</b></td>
<td>AIME 2024 (pass@1)</td>
<td>30.05</td>
<td>26.67</td>
<td>88.75%</td>
</tr>
<tr>
<td>MATH-500 (pass@1)</td>
<td>84.66</td>
<td>84.39</td>
<td>99.68%</td>
</tr>
<tr>
<td>GPQA Diamond (pass@1)</td>
<td>35.37</td>
<td>34.43</td>
<td>97.34%</td>
</tr>
<tr>
<td><b>Average Score</b></td>
<td><b>50.03</b></td>
<td><b>48.5</b></td>
<td><b>96.94%</b></td>
</tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>37.20</td>
      <td>37.03</td>
      <td>99.5%</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>69.98</td>
      <td>68.46</td>
      <td>97.8%</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>43.86</td>
      <td>43.91</td>
      <td>100.1%</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>37.38</td>
      <td>37.20</td>
      <td>99.5%</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>45.21</td>
      <td>45.59</td>
      <td>100.8%</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>54.30</td>
      <td>54.30</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>47.99</b></td>
      <td><b>47.75</b></td>
      <td><b>99.5%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>34.63</td>
      <td>35.70</td>
      <td>103.1%</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>3.06</td>
      <td>2.31</td>
      <td>---</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>---</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>1.01</td>
      <td>1.19</td>
      <td>---</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>0.78</td>
      <td>0.29</td>
      <td>---</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>1.32</td>
      <td>1.23</td>
      <td>---</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>6.80</b></td>
      <td><b>6.79</b></td>
      <td><b>---</b></td>
    </tr>
    <tr>
      <td rowspan="4"><b>Coding</b></td>
      <td>HumanEval (pass@1)</td>
      <td>37.90</td>
      <td>36.20</td>
      <td><b>95.5%</b></td>
    </tr>
    <tr>
      <td>HumanEval (pass@10)</td>
      <td>61.30</td>
      <td>63.10</td>
      <td>102.9%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>33.00</td>
      <td>32.70</td>
      <td>99.1%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>55.90</td>
      <td>58.40</td>
      <td>104.5%</td>
    </tr>
  </tbody>
</table>


## Inference Performance


This model achieves up to 1.2x speedup in single-stream deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w8a8 --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
```
</details>

### Single-stream performance (measured with vLLM version 0.7.2)
<table>
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th style="text-align: center;" colspan="2" >Instruction Following<br>256 / 128</th>
      <th style="text-align: center;" colspan="2" >Multi-turn Chat<br>512 / 256</th>
      <th style="text-align: center;" colspan="2" >Docstring Generation<br>768 / 128</th>
      <th style="text-align: center;" colspan="2" >RAG<br>1024 / 128</th>
      <th style="text-align: center;" colspan="2" >Code Completion<br>256 / 1024</th>
      <th style="text-align: center;" colspan="2" >Code Fixing<br>1024 / 1024</th>
      <th style="text-align: center;" colspan="2" >Large Summarization<br>4096 / 512</th>
      <th style="text-align: center;" colspan="2" >Large RAG<br>10240 / 1536</th>
    </tr>
    <tr>
      <th>Hardware</th>
      <th>Model</th>
      <th>Average cost reduction</th>
      <th>Latency (s)</th>
      <th>QPD</th>
      <th>Latency (s)</th>
      <th>QPD</th>
      <th>Latency (s)</th>
      <th>QPD</th>
      <th>Latency (s)</th>
      <th>QPD</th>
      <th>Latency (s)</th>
      <th>QPD</th>
      <th>Latency (s)</th>
      <th>QPD</th>
      <th>Latency (s)</th>
      <th>QPD</th>
      <th>Latency (s)</th>
      <th>QPD</th>
    </tr>
  </thead>
  <tbody style="text-align: center" >
    <tr>
      <th rowspan="3" valign="top">A6000x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</th>
      <td>---</td>
      <td>0.8</td>
      <td>5667</td>
      <td>1.6</td>
      <td>2776</td>
      <td>0.8</td>
      <td>5515</td>
      <td>0.8</td>
      <td>5466</td>
      <td>6.4</td>
      <td>705</td>
      <td>6.5</td>
      <td>697</td>
      <td>3.5</td>
      <td>1295</td>
      <td>18.3</td>
      <td>246</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w8a8</th>
      <td>1.14</td>
      <td>0.7</td>
      <td>6635</td>
      <td>1.3</td>
      <td>3340</td>
      <td>0.7</td>
      <td>6396</td>
      <td>0.7</td>
      <td>6343</td>
      <td>5.3</td>
      <td>845</td>
      <td>5.4</td>
      <td>832</td>
      <td>2.9</td>
      <td>1547</td>
      <td>21.3</td>
      <td>211</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a16</th>
      <td>1.38</td>
      <td>0.5</td>
      <td>8293</td>
      <td>1.1</td>
      <td>4184</td>
      <td>0.6</td>
      <td>7976</td>
      <td>0.6</td>
      <td>7504</td>
      <td>4.3</td>
      <td>1051</td>
      <td>4.4</td>
      <td>1033</td>
      <td>2.5</td>
      <td>1819</td>
      <td>21.1</td>
      <td>213</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</th>
      <td>---</td>
      <td>0.6</td>
      <td>3359</td>
      <td>1.2</td>
      <td>1654</td>
      <td>0.6</td>
      <td>3286</td>
      <td>0.6</td>
      <td>3241</td>
      <td>4.7</td>
      <td>424</td>
      <td>4.9</td>
      <td>411</td>
      <td>2.6</td>
      <td>778</td>
      <td>21.1</td>
      <td>95</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w8a8</th>
      <td>1.05</td>
      <td>0.6</td>
      <td>3531</td>
      <td>1.1</td>
      <td>1807</td>
      <td>0.6</td>
      <td>3427</td>
      <td>0.6</td>
      <td>3480</td>
      <td>4.5</td>
      <td>448</td>
      <td>4.5</td>
      <td>447</td>
      <td>2.4</td>
      <td>842</td>
      <td>23.5</td>
      <td>86</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a16</th>
      <td>1.03</td>
      <td>0.6</td>
      <td>3469</td>
      <td>1.1</td>
      <td>1751</td>
      <td>0.6</td>
      <td>3403</td>
      <td>0.6</td>
      <td>3407</td>
      <td>4.5</td>
      <td>447</td>
      <td>4.6</td>
      <td>435</td>
      <td>2.5</td>
      <td>815</td>
      <td>23.3</td>
      <td>86</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B</th>
      <td>---</td>
      <td>0.4</td>
      <td>2604</td>
      <td>0.8</td>
      <td>1299</td>
      <td>0.4</td>
      <td>2543</td>
      <td>0.4</td>
      <td>2551</td>
      <td>3.3</td>
      <td>330</td>
      <td>3.4</td>
      <td>326</td>
      <td>1.8</td>
      <td>612</td>
      <td>14.0</td>
      <td>78</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-FP8-dynamic</th>
      <td>1.04</td>
      <td>0.4</td>
      <td>2694</td>
      <td>0.8</td>
      <td>1364</td>
      <td>0.4</td>
      <td>2670</td>
      <td>0.4</td>
      <td>2639</td>
      <td>3.2</td>
      <td>347</td>
      <td>3.2</td>
      <td>341</td>
      <td>1.6</td>
      <td>673</td>
      <td>14.1</td>
      <td>78</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-1.5B-quantized.w4a16</th>
      <td>0.84</td>
      <td>0.5</td>
      <td>2111</td>
      <td>1.0</td>
      <td>1065</td>
      <td>0.5</td>
      <td>2068</td>
      <td>0.5</td>
      <td>2119</td>
      <td>4.1</td>
      <td>270</td>
      <td>4.1</td>
      <td>265</td>
      <td>2.1</td>
      <td>530</td>
      <td>15.1</td>
      <td>73</td>
    </tr>
  </tbody>
</table>

**Use case profiles: prompt tokens / generation tokens

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).