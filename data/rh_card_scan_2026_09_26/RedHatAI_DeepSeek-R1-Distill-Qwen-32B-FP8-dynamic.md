---
license: mit
tags:
- deepseek
- fp8
- vllm
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
library_name: transformers
---

# DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic

## Model Overview
- **Model Architecture:** Qwen2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2/5/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B).


### Model Optimizations

This model was obtained by quantizing the weights and activations of [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-channel scheme, whereas quantizations are quantized using a symmetric per-token scheme.
[LLM Compressor](https://github.com/vllm-project/llm-compressor) is used for quantization.


## Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

number_gpus = 1
model_name = "neuralmagic/DeepSeek-R1-Distill-Qwen-32B-dynamic"

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
from llmcompressor.transformers import oneshot
import os

# Load model
model_stub = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
model_name = model_stub.split("/")[-1]

model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_stub)

# Configure the quantization algorithm and scheme
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head"],
)

# Apply quantization
oneshot(
    model=model,
    recipe=recipe,
)

# Save to disk in compressed-tensors format
save_path = model_name + "-FP8-dynamic
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-32B</th>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td rowspan="4"><b>Reasoning</b></td>
<td>AIME 2024 (pass@1)</td>
<td>69.75</td>
<td>68.5</td>
<td>98.21%</td>
</tr>
<tr>
<td>MATH-500 (pass@1)</td>
<td>95.09</td>
<td>95.26</td>
<td>100.18%</td>
</tr>
<tr>
<td>GPQA Diamond (pass@1)</td>
<td>64.05</td>
<td>62.88</td>
<td>98.17%</td>
</tr>
<tr>
<td><b>Average Score</b></td>
<td><b>76.3</b></td>
<td><b>75.55</b></td>
<td><b>99.02%</b></td>
</tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>64.59</td>
      <td>64.42</td>
      <td>99.7%</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>82.71</td>
      <td>82.64</td>
      <td>99.9%</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>83.80</td>
      <td>83.77</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>81.12</td>
      <td>80.98</td>
      <td>99.8%</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>58.41</td>
      <td>58.30</td>
      <td>99.8%</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>76.40</td>
      <td>76.09</td>
      <td>99.6%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>74.51</b></td>
      <td><b>74.36</b></td>
      <td><b>99.8%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>42.87</td>
      <td>49.43</td>
      <td>99.2%</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>57.96</td>
      <td>58.38</td>
      <td>100.7%</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>---</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>26.95</td>
      <td>26.86</td>
      <td>99.7%</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>43.95</td>
      <td>44.22</td>
      <td>100.6%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>49.82</td>
      <td>49.43</td>
      <td>99.2%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>36.92</b></td>
      <td><b>36.86</b></td>
      <td><b>99.8%</b></td>
    </tr>
    <tr>
      <td rowspan="4"><b>Coding</b></td>
      <td>HumanEval (pass@1)</td>
      <td>86.00</td>
      <td>85.20</td>
      <td><b>99.1%</b></td>
    </tr>
    <tr>
      <td>HumanEval (pass@10)</td>
      <td>92.50</td>
      <td>92.20</td>
      <td>99.7%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>82.00</td>
      <td>80.90</td>
      <td>98.7%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>88.70</td>
      <td>88.70</td>
      <td>100.0%</td>
    </tr>
  </tbody>
</table>

## Inference Performance


This model achieves up to 1.5x speedup in single-stream deployment and up to 1.7x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
```
</details>

### Single-stream performance (measured with vLLM version 0.7.2)
<table>
  <thead>
    <tr>
      <th></th>
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
      <th>GPU class</th>
      <th>Number of GPUs</th>
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
      <th rowspan="3" valign="top">A6000</th>
      <td>2</td>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-32B</th>
      <td>---</td>
      <td>6.3</td>
      <td>359</td>
      <td>12.8</td>
      <td>176</td>
      <td>6.5</td>
      <td>347</td>
      <td>6.6</td>
      <td>342</td>
      <td>49.9</td>
      <td>45</td>
      <td>50.8</td>
      <td>44</td>
      <td>26.6</td>
      <td>85</td>
      <td>83.4</td>
      <td>27</td>
    </tr>
    <tr>
      <td>1</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8</th>
      <td>1.81</td>
      <td>6.9</td>
      <td>648</td>
      <td>13.8</td>
      <td>325</td>
      <td>7.2</td>
      <td>629</td>
      <td>7.2</td>
      <td>622</td>
      <td>54.8</td>
      <td>82</td>
      <td>55.6</td>
      <td>81</td>
      <td>30.0</td>
      <td>150</td>
      <td>94.8</td>
      <td>47</td>
    </tr>
    <tr>
      <td>1</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16</th>
      <td>3.07</td>
      <td>3.9</td>
      <td>1168</td>
      <td>7.8</td>
      <td>580</td>
      <td>4.3</td>
      <td>1041</td>
      <td>4.6</td>
      <td>975</td>
      <td>29.7</td>
      <td>151</td>
      <td>30.9</td>
      <td>146</td>
      <td>19.3</td>
      <td>233</td>
      <td>61.4</td>
      <td>73</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100</th>
      <td>1</td>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-32B</th>
      <td>---</td>
      <td>5.6</td>
      <td>361</td>
      <td>11.1</td>
      <td>180</td>
      <td>5.7</td>
      <td>350</td>
      <td>5.8</td>
      <td>347</td>
      <td>44.0</td>
      <td>46</td>
      <td>44.7</td>
      <td>45</td>
      <td>23.6</td>
      <td>85</td>
      <td>73.7</td>
      <td>27</td>
    </tr>
    <tr>
      <td>1</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8</th>
      <td>1.50</td>
      <td>3.7</td>
      <td>547</td>
      <td>7.3</td>
      <td>275</td>
      <td>3.8</td>
      <td>536</td>
      <td>3.8</td>
      <td>528</td>
      <td>29.0</td>
      <td>69</td>
      <td>29.5</td>
      <td>68</td>
      <td>15.7</td>
      <td>128</td>
      <td>53.1</td>
      <td>38</td>
    </tr>
    <tr>
      <td>1</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16</th>
      <td>2.30</td>
      <td>2.2</td>
      <td>894</td>
      <td>4.5</td>
      <td>449</td>
      <td>2.4</td>
      <td>831</td>
      <td>2.5</td>
      <td>798</td>
      <td>17.4</td>
      <td>116</td>
      <td>18.0</td>
      <td>112</td>
      <td>10.5</td>
      <td>191</td>
      <td>49.5</td>
      <td>41</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100</th>
      <td>1</td>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-32B</th>
      <td>---</td>
      <td>3.3</td>
      <td>327</td>
      <td>6.7</td>
      <td>163</td>
      <td>3.4</td>
      <td>320</td>
      <td>3.4</td>
      <td>317</td>
      <td>26.6</td>
      <td>41</td>
      <td>26.9</td>
      <td>41</td>
      <td>14.3</td>
      <td>77</td>
      <td>47.8</td>
      <td>23</td>
    </tr>
    <tr>
      <td>1</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic</th>
      <td>1.52</td>
      <td>2.2</td>
      <td>503</td>
      <td>4.3</td>
      <td>252</td>
      <td>2.2</td>
      <td>490</td>
      <td>2.3</td>
      <td>485</td>
      <td>17.3</td>
      <td>63</td>
      <td>17.5</td>
      <td>63</td>
      <td>9.5</td>
      <td>116</td>
      <td>33.4</td>
      <td>33</td>
    </tr>
    <tr>
      <td>1</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16</th>
      <td>1.61</td>
      <td>2.1</td>
      <td>532</td>
      <td>4.1</td>
      <td>268</td>
      <td>2.1</td>
      <td>516</td>
      <td>2.1</td>
      <td>513</td>
      <td>16.1</td>
      <td>68</td>
      <td>16.5</td>
      <td>66</td>
      <td>9.1</td>
      <td>120</td>
      <td>31.9</td>
      <td>34</td>
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
      <th rowspan="3" valign="top">A6000x2</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-32B</th>
      <td>---</td>
      <td>6.2</td>
      <td>13940</td>
      <td>1.9</td>
      <td>4348</td>
      <td>2.7</td>
      <td>6153</td>
      <td>2.1</td>
      <td>4778</td>
      <td>0.6</td>
      <td>1382</td>
      <td>0.4</td>
      <td>930</td>
      <td>0.3</td>
      <td>685</td>
      <td>0.1</td>
      <td>124</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8</th>
      <td>1.80</td>
      <td>8.7</td>
      <td>19492</td>
      <td>4.2</td>
      <td>9474</td>
      <td>4.1</td>
      <td>9290</td>
      <td>3.0</td>
      <td>6802</td>
      <td>1.2</td>
      <td>2734</td>
      <td>0.9</td>
      <td>1962</td>
      <td>0.5</td>
      <td>1177</td>
      <td>0.1</td>
      <td>254</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16</th>
      <td>1.30</td>
      <td>5.9</td>
      <td>13366</td>
      <td>2.5</td>
      <td>5733</td>
      <td>2.4</td>
      <td>5409</td>
      <td>1.6</td>
      <td>3525</td>
      <td>1.2</td>
      <td>2757</td>
      <td>0.7</td>
      <td>1663</td>
      <td>0.3</td>
      <td>676</td>
      <td>0.1</td>
      <td>214</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x2</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-32B</th>
      <td>---</td>
      <td>12.9</td>
      <td>13016</td>
      <td>5.8</td>
      <td>5848</td>
      <td>6.3</td>
      <td>6348</td>
      <td>5.1</td>
      <td>5146</td>
      <td>2.0</td>
      <td>1988</td>
      <td>1.5</td>
      <td>1463</td>
      <td>0.9</td>
      <td>869</td>
      <td>0.2</td>
      <td>192</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8</th>
      <td>1.52</td>
      <td>21.4</td>
      <td>21479</td>
      <td>8.9</td>
      <td>8948</td>
      <td>10.6</td>
      <td>10611</td>
      <td>8.2</td>
      <td>8197</td>
      <td>3.0</td>
      <td>3018</td>
      <td>2.0</td>
      <td>2054</td>
      <td>1.2</td>
      <td>1241</td>
      <td>0.3</td>
      <td>264</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16</th>
      <td>1.09</td>
      <td>13.5</td>
      <td>13568</td>
      <td>6.5</td>
      <td>6509</td>
      <td>6.0</td>
      <td>6075</td>
      <td>4.7</td>
      <td>4754</td>
      <td>2.8</td>
      <td>2790</td>
      <td>1.6</td>
      <td>1651</td>
      <td>0.9</td>
      <td>862</td>
      <td>0.2</td>
      <td>225</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x2</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-32B</th>
      <td>---</td>
      <td>25.5</td>
      <td>14392</td>
      <td>12.5</td>
      <td>7035</td>
      <td>14.0</td>
      <td>7877</td>
      <td>11.3</td>
      <td>6364</td>
      <td>3.6</td>
      <td>2041</td>
      <td>2.7</td>
      <td>1549</td>
      <td>1.9</td>
      <td>1057</td>
      <td>0.4</td>
      <td>200</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic</th>
      <td>1.46</td>
      <td>46.7</td>
      <td>25538</td>
      <td>20.3</td>
      <td>11082</td>
      <td>23.3</td>
      <td>12728</td>
      <td>18.4</td>
      <td>10049</td>
      <td>5.3</td>
      <td>2881</td>
      <td>3.7</td>
      <td>2097</td>
      <td>2.6</td>
      <td>1445</td>
      <td>0.5</td>
      <td>256</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16</th>
      <td>1.23</td>
      <td>36.9</td>
      <td>20172</td>
      <td>17.4</td>
      <td>9500</td>
      <td>18.0</td>
      <td>9822</td>
      <td>14.2</td>
      <td>7755</td>
      <td>5.3</td>
      <td>2900</td>
      <td>3.3</td>
      <td>1867</td>
      <td>2.3</td>
      <td>1265</td>
      <td>0.4</td>
      <td>241</td>
    </tr>
  </tbody>
</table>

**Use case profiles: prompt tokens / generation tokens

**QPS: Queries per second.

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).