---
license: mit
tags:
- deepseek
- int4
- vllm
- llmcompressor
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
library_name: transformers
---

# DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16

## Model Overview
- **Model Architecture:** Qwen2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Release Date:** 2/4/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B).


### Model Optimizations

This model was obtained by quantizing the weights of [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.


Only the weights of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-group scheme, with group size 128.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

number_gpus = 1
model_name = "neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16"

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
model_stub = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
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
recipe = QuantizationModifier(
    targets="Linear",
    scheme="W4A16",
    ignore=["lm_head"],
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
save_path = model_name + "-quantized.w4a16
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</th>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td rowspan="4"><b>Reasoning</b></td>
<td>AIME 2024 (pass@1)</td>
<td>66.67</td>
<td>66.04</td>
<td>99.06%</td>
</tr>
<tr>
<td>MATH-500 (pass@1)</td>
<td>94.66</td>
<td>94.95</td>
<td>100.31%</td>
</tr>
<tr>
<td>GPQA Diamond (pass@1)</td>
<td>59.35</td>
<td>57.48</td>
<td>96.85%</td>
</tr>
<tr>
<td><b>Average Score</b></td>
<td><b>73.56</b></td>
<td><b>72.82</b></td>
<td><b>98.99%</b></td>
</tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>58.79</td>
      <td>58.28</td>
      <td>99.1%</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>87.04</td>
      <td>87.34</td>
      <td>100.4%</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>81.51</td>
      <td>80.42</td>
      <td>98.7%</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>74.46</td>
      <td>73.32</td>
      <td>98.5%</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>54.77</td>
      <td>55.29</td>
      <td>101.0%</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>69.38</td>
      <td>70.48</td>
      <td>101.6%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>70.99</b></td>
      <td><b>70.85</b></td>
      <td><b>99.8%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>43.05</td>
      <td>34.90</td>
      <td>81.1%</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>47.16</td>
      <td>45.36</td>
      <td>96.2%</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>---</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>35.07</td>
      <td>34.90</td>
      <td>99.5%</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>45.14</td>
      <td>44.20</td>
      <td>97.9%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>34.86</td>
      <td>35.09</td>
      <td>100.7%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>34.21</b></td>
      <td><b>32.41</b></td>
      <td><b>94.7%</b></td>
    </tr>
    <tr>
      <td rowspan="4"><b>Coding</b></td>
      <td>HumanEval (pass@1)</td>
      <td>78.90</td>
      <td>79.00</td>
      <td><b>100.1%</b></td>
    </tr>
    <tr>
      <td>HumanEval (pass@10)</td>
      <td>89.80</td>
      <td>89.70</td>
      <td>99.9%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>72.60</td>
      <td>72.80</td>
      <td>100.3%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>84.90</td>
      <td>84.00</td>
      <td>98.8%</td>
    </tr>
  </tbody>
</table>

## Inference Performance


This model achieves up to 2.8x speedup in single-stream deployment and up to 1.4x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16 --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</th>
      <td>---</td>
      <td>5.4</td>
      <td>837</td>
      <td>10.7</td>
      <td>419</td>
      <td>5.5</td>
      <td>813</td>
      <td>5.6</td>
      <td>805</td>
      <td>42.2</td>
      <td>107</td>
      <td>42.8</td>
      <td>105</td>
      <td>22.9</td>
      <td>197</td>
      <td>71.7</td>
      <td>63</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w8a8</th>
      <td>1.59</td>
      <td>3.3</td>
      <td>1345</td>
      <td>6.7</td>
      <td>673</td>
      <td>3.4</td>
      <td>1315</td>
      <td>3.5</td>
      <td>1296</td>
      <td>26.5</td>
      <td>170</td>
      <td>26.8</td>
      <td>168</td>
      <td>14.5</td>
      <td>310</td>
      <td>48.3</td>
      <td>93</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16</th>
      <td>2.51</td>
      <td>2.0</td>
      <td>2275</td>
      <td>4.0</td>
      <td>1127</td>
      <td>2.2</td>
      <td>2072</td>
      <td>2.3</td>
      <td>1945</td>
      <td>15.3</td>
      <td>294</td>
      <td>15.9</td>
      <td>283</td>
      <td>9.9</td>
      <td>456</td>
      <td>36.6</td>
      <td>123</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</th>
      <td>---</td>
      <td>2.6</td>
      <td>765</td>
      <td>5.2</td>
      <td>383</td>
      <td>2.7</td>
      <td>746</td>
      <td>2.7</td>
      <td>732</td>
      <td>20.8</td>
      <td>97</td>
      <td>21.2</td>
      <td>95</td>
      <td>11.3</td>
      <td>179</td>
      <td>36.7</td>
      <td>55</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w8a8</th>
      <td>1.34</td>
      <td>1.9</td>
      <td>1072</td>
      <td>3.8</td>
      <td>533</td>
      <td>1.9</td>
      <td>1045</td>
      <td>1.9</td>
      <td>1032</td>
      <td>14.8</td>
      <td>136</td>
      <td>15.2</td>
      <td>132</td>
      <td>8.1</td>
      <td>248</td>
      <td>39.6</td>
      <td>51</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16</th>
      <td>1.93</td>
      <td>1.2</td>
      <td>1627</td>
      <td>2.5</td>
      <td>810</td>
      <td>1.3</td>
      <td>1530</td>
      <td>1.4</td>
      <td>1474</td>
      <td>9.7</td>
      <td>208</td>
      <td>10.2</td>
      <td>197</td>
      <td>5.8</td>
      <td>348</td>
      <td>37.6</td>
      <td>53</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</th>
      <td>---</td>
      <td>1.6</td>
      <td>672</td>
      <td>3.3</td>
      <td>334</td>
      <td>1.7</td>
      <td>662</td>
      <td>1.7</td>
      <td>652</td>
      <td>12.8</td>
      <td>85</td>
      <td>13.0</td>
      <td>84</td>
      <td>7.0</td>
      <td>155</td>
      <td>25.2</td>
      <td>43</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic</th>
      <td>1.33</td>
      <td>1.2</td>
      <td>925</td>
      <td>2.3</td>
      <td>467</td>
      <td>1.2</td>
      <td>908</td>
      <td>1.2</td>
      <td>896</td>
      <td>9.3</td>
      <td>118</td>
      <td>9.5</td>
      <td>115</td>
      <td>5.2</td>
      <td>210</td>
      <td>23.9</td>
      <td>46</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16</th>
      <td>1.37</td>
      <td>1.2</td>
      <td>944</td>
      <td>2.3</td>
      <td>474</td>
      <td>1.2</td>
      <td>931</td>
      <td>1.2</td>
      <td>907</td>
      <td>9.1</td>
      <td>121</td>
      <td>9.2</td>
      <td>119</td>
      <td>5.1</td>
      <td>214</td>
      <td>22.5</td>
      <td>49</td>
    </tr>
  </tbody>
</table>

**Use case profiles: prompt tokens / generation tokens

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).


### Multi-stream asynchronous performance (measured with vLLM version 0.7.2)
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
      <th>Maximum throughput (QPS)</th>
      <th>QPD</th>
      <th>Maximum throughput (QPS)</th>
      <th>QPD</th>
      <th>Maximum throughput (QPS)</th>
      <th>QPD</th>
      <th>Maximum throughput (QPS)</th>
      <th>QPD</th>
      <th>Maximum throughput (QPS)</th>
      <th>QPD</th>
      <th>Maximum throughput (QPS)</th>
      <th>QPD</th>
      <th>Maximum throughput (QPS)</th>
      <th>QPD</th>
      <th>Maximum throughput (QPS)</th>
      <th>QPD</th>
    </tr>
  </thead>
  <tbody style="text-align: center" >
    <tr>
      <th rowspan="3" valign="top">A6000x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</th>
      <td>---</td>
      <td>13.7</td>
      <td>30785</td>
      <td>5.5</td>
      <td>12327</td>
      <td>6.5</td>
      <td>14517</td>
      <td>5.1</td>
      <td>11439</td>
      <td>2.0</td>
      <td>4434</td>
      <td>1.3</td>
      <td>2982</td>
      <td>0.6</td>
      <td>1462</td>
      <td>0.2</td>
      <td>371</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w8a8</th>
      <td>1.44</td>
      <td>21.4</td>
      <td>48181</td>
      <td>8.2</td>
      <td>18421</td>
      <td>9.8</td>
      <td>22051</td>
      <td>7.8</td>
      <td>17462</td>
      <td>2.8</td>
      <td>6281</td>
      <td>1.7</td>
      <td>3758</td>
      <td>1.0</td>
      <td>2335</td>
      <td>0.2</td>
      <td>419</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16</th>
      <td>0.98</td>
      <td>12.7</td>
      <td>28540</td>
      <td>5.7</td>
      <td>12796</td>
      <td>5.4</td>
      <td>12218</td>
      <td>3.7</td>
      <td>8401</td>
      <td>2.5</td>
      <td>5583</td>
      <td>1.3</td>
      <td>2987</td>
      <td>0.7</td>
      <td>1489</td>
      <td>0.2</td>
      <td>368</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</th>
      <td>---</td>
      <td>15.6</td>
      <td>31306</td>
      <td>7.1</td>
      <td>14192</td>
      <td>7.7</td>
      <td>15435</td>
      <td>6.0</td>
      <td>11971</td>
      <td>2.4</td>
      <td>4878</td>
      <td>1.6</td>
      <td>3298</td>
      <td>0.9</td>
      <td>1862</td>
      <td>0.2</td>
      <td>355</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w8a8</th>
      <td>1.31</td>
      <td>20.8</td>
      <td>41907</td>
      <td>9.3</td>
      <td>18724</td>
      <td>10.5</td>
      <td>21043</td>
      <td>8.4</td>
      <td>16886</td>
      <td>3.0</td>
      <td>5975</td>
      <td>1.9</td>
      <td>3917</td>
      <td>1.2</td>
      <td>2481</td>
      <td>0.2</td>
      <td>464</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16</th>
      <td>0.94</td>
      <td>14.0</td>
      <td>28146</td>
      <td>6.5</td>
      <td>13042</td>
      <td>6.5</td>
      <td>12987</td>
      <td>5.1</td>
      <td>10194</td>
      <td>2.6</td>
      <td>5269</td>
      <td>1.5</td>
      <td>2925</td>
      <td>0.9</td>
      <td>1849</td>
      <td>0.2</td>
      <td>382</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-14B</th>
      <td>---</td>
      <td>31.4</td>
      <td>34404</td>
      <td>14.1</td>
      <td>15482</td>
      <td>16.6</td>
      <td>18149</td>
      <td>13.3</td>
      <td>14572</td>
      <td>4.7</td>
      <td>5099</td>
      <td>2.6</td>
      <td>2849</td>
      <td>1.9</td>
      <td>2060</td>
      <td>0.3</td>
      <td>347</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-FP8-dynamic</th>
      <td>1.31</td>
      <td>40.9</td>
      <td>44729</td>
      <td>18.5</td>
      <td>20260</td>
      <td>22.1</td>
      <td>24165</td>
      <td>18.1</td>
      <td>19779</td>
      <td>5.7</td>
      <td>6246</td>
      <td>3.4</td>
      <td>3681</td>
      <td>2.5</td>
      <td>2746</td>
      <td>0.4</td>
      <td>474</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16</th>
      <td>1.12</td>
      <td>33.3</td>
      <td>36387</td>
      <td>15.0</td>
      <td>16453</td>
      <td>17.6</td>
      <td>19241</td>
      <td>14.2</td>
      <td>15576</td>
      <td>4.6</td>
      <td>5034</td>
      <td>3.0</td>
      <td>3292</td>
      <td>2.2</td>
      <td>2412</td>
      <td>0.4</td>
      <td>481</td>
    </tr>
  </tbody>
</table>

**Use case profiles: prompt tokens / generation tokens

**QPS: Queries per second.

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).