---
license: mit
tags:
- deepseek
- fp8
- vllm
base_model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
library_name: transformers
---

# DeepSeek-R1-Distill-Llama-70B-FP8-dynamic

## Model Overview
- **Model Architecture:** LlamaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 2/1/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B).


### Model Optimizations

This model was obtained by quantizing the weights and activations of [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) to FP8 data type.
This optimization reduces the number of bits per parameter from 16 to 8, reducing the disk size and GPU memory requirements by approximately 50%.

Only the weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized using a symmetric per-channel scheme, whereas quantizations are quantized using a symmetric per-token scheme.
[LLM Compressor](https://github.com/vllm-project/llm-compressor) is used for quantization.


## Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

number_gpus = 2
model_name = "neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic"

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
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map
import os

# Load model
model_stub = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
model_name = model_stub.split("/")[-1]

device_map = calculate_offload_device_map(
    model_stub,
    reserve_for_hessians=True,
    num_gpus=2,
    torch_dtype="auto",
)

model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    device_map=device_map,
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True \
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True \
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-70B</th>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td rowspan="4"><b>Reasoning</b></td>
<td>AIME 2024 (pass@1)</td>
<td>67.83</td>
<td>69.17</td>
<td>101.98%</td>
</tr>
<tr>
<td>MATH-500 (pass@1)</td>
<td>95.29</td>
<td>95.14</td>
<td>99.84%</td>
</tr>
<tr>
<td>GPQA Diamond (pass@1)</td>
<td>65.57</td>
<td>65.15</td>
<td>99.36%</td>
</tr>
<tr>
<td><b>Average Score</b></td>
<td><b>76.23</b></td>
<td><b>76.49</b></td>
<td><b>100.34%</b></td>
</tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>63.65</td>
      <td>63.05</td>
      <td>99.1%</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>93.03</td>
      <td>93.03</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>84.85</td>
      <td>84.71</td>
      <td>99.8%</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>78.04</td>
      <td>77.45</td>
      <td>99.3%</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>56.67</td>
      <td>56.62</td>
      <td>99.9%</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>78.22</td>
      <td>78.45</td>
      <td>100.3%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>75.74</b></td>
      <td><b>75.55</b></td>
      <td><b>99.8%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>42.45</td>
      <td>42.11</td>
      <td>99.2%</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>21.26</td>
      <td>19.77</td>
      <td>93.0%</td>
    </tr>
    <tr>
      <td>Math-Hard (Exact-Match, 4-shot)</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>---</td>
    </tr>
    <tr>
      <td>GPQA (Acc-Norm, 0-shot)</td>
      <td>9.51</td>
      <td>6.97</td>
      <td>---</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>14.87</td>
      <td>14.60</td>
      <td>---</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>4.27</td>
      <td>5.76</td>
      <td>---</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>15.39</b></td>
      <td><b>14.87</b></td>
      <td><b>96.6%</b></td>
    </tr>
    <tr>
      <td rowspan="4"><b>Coding</b></td>
      <td>HumanEval (pass@1)</td>
      <td>81.10</td>
      <td>81.00</td>
      <td><b>99.9%</b></td>
    </tr>
    <tr>
      <td>HumanEval (pass@10)</td>
      <td>87.60</td>
      <td>88.60</td>
      <td>101.1%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>75.20</td>
      <td>75.50</td>
      <td>100.4%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>83.10</td>
      <td>84.30</td>
      <td>101.4%</td>
    </tr>
  </tbody>
</table>


## Inference Performance


This model achieves up to 1.4x speedup in single-stream deployment and up to 3.0x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
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
      <td>4</td>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-70B</th>
      <td>---</td>
      <td>7.4</td>
      <td>152</td>
      <td>14.9</td>
      <td>76</td>
      <td>7.5</td>
      <td>149</td>
      <td>7.7</td>
      <td>146</td>
      <td>57.2</td>
      <td>20</td>
      <td>58.9</td>
      <td>19</td>
      <td>31.9</td>
      <td>35</td>
      <td>98.4</td>
      <td>11</td>
    </tr>
    <tr>
      <td>2</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8</th>
      <td>1.93</td>
      <td>7.7</td>
      <td>292</td>
      <td>15.2</td>
      <td>148</td>
      <td>7.8</td>
      <td>287</td>
      <td>8.0</td>
      <td>282</td>
      <td>60.7</td>
      <td>37</td>
      <td>60.2</td>
      <td>37</td>
      <td>32.3</td>
      <td>70</td>
      <td>104.0</td>
      <td>22</td>
    </tr>
    <tr>
      <td>2</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16</th>
      <td>2.83</td>
      <td>4.9</td>
      <td>457</td>
      <td>10.0</td>
      <td>225</td>
      <td>5.5</td>
      <td>411</td>
      <td>5.8</td>
      <td>389</td>
      <td>38.9</td>
      <td>58</td>
      <td>39.2</td>
      <td>57</td>
      <td>23.7</td>
      <td>95</td>
      <td>76.6</td>
      <td>29</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100</th>
      <td>2</td>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-70B</th>
      <td>---</td>
      <td>6.4</td>
      <td>157</td>
      <td>12.8</td>
      <td>79</td>
      <td>6.6</td>
      <td>153</td>
      <td>6.7</td>
      <td>151</td>
      <td>50.4</td>
      <td>20</td>
      <td>50.8</td>
      <td>20</td>
      <td>27.0</td>
      <td>37</td>
      <td>85.4</td>
      <td>12</td>
    </tr>
    <tr>
      <td>2</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8</th>
      <td>1.48</td>
      <td>4.1</td>
      <td>245</td>
      <td>8.2</td>
      <td>123</td>
      <td>4.2</td>
      <td>238</td>
      <td>4.3</td>
      <td>235</td>
      <td>32.4</td>
      <td>31</td>
      <td>32.8</td>
      <td>31</td>
      <td>17.6</td>
      <td>57</td>
      <td>90.8</td>
      <td>11</td>
    </tr>
    <tr>
      <td>1</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16</th>
      <td>2.69</td>
      <td>4.6</td>
      <td>440</td>
      <td>9.2</td>
      <td>220</td>
      <td>4.9</td>
      <td>407</td>
      <td>5.2</td>
      <td>389</td>
      <td>35.3</td>
      <td>57</td>
      <td>36.3</td>
      <td>55</td>
      <td>21.2</td>
      <td>95</td>
      <td>68.1</td>
      <td>30</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100</th>
      <td>2</td>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-70B</th>
      <td>---</td>
      <td>3.8</td>
      <td>149</td>
      <td>7.6</td>
      <td>74</td>
      <td>3.9</td>
      <td>146</td>
      <td>3.9</td>
      <td>144</td>
      <td>30.0</td>
      <td>19</td>
      <td>30.4</td>
      <td>19</td>
      <td>16.1</td>
      <td>35</td>
      <td>56.5</td>
      <td>10</td>
    </tr>
    <tr>
      <td>2</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic</th>
      <td>1.39</td>
      <td>2.7</td>
      <td>210</td>
      <td>5.3</td>
      <td>106</td>
      <td>2.7</td>
      <td>207</td>
      <td>2.8</td>
      <td>203</td>
      <td>21.1</td>
      <td>27</td>
      <td>21.4</td>
      <td>26</td>
      <td>11.5</td>
      <td>49</td>
      <td>47.2</td>
      <td>12</td>
    </tr>
    <tr>
      <td>1</td>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16</th>
      <td>1.83</td>
      <td>4.0</td>
      <td>277</td>
      <td>7.9</td>
      <td>138</td>
      <td>4.1</td>
      <td>266</td>
      <td>4.2</td>
      <td>262</td>
      <td>31.2</td>
      <td>35</td>
      <td>31.8</td>
      <td>34</td>
      <td>17.8</td>
      <td>61</td>
      <td>61.4</td>
      <td>18</td>
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
      <th rowspan="3" valign="top">A6000x4</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-70B</th>
      <td>---</td>
      <td>3.65</td>
      <td>4102</td>
      <td>1.56</td>
      <td>1757</td>
      <td>1.90</td>
      <td>2143</td>
      <td>1.48</td>
      <td>1665</td>
      <td>0.44</td>
      <td>493</td>
      <td>0.34</td>
      <td>380</td>
      <td>0.22</td>
      <td>245</td>
      <td>0.05</td>
      <td>55</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8</th>
      <td>1.76</td>
      <td>5.89</td>
      <td>6625</td>
      <td>2.94</td>
      <td>3307</td>
      <td>3.36</td>
      <td>3775</td>
      <td>2.59</td>
      <td>2916</td>
      <td>0.74</td>
      <td>828</td>
      <td>0.53</td>
      <td>601</td>
      <td>0.35</td>
      <td>398</td>
      <td>0.11</td>
      <td>120</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16</th>
      <td>1.48</td>
      <td>4.91</td>
      <td>5528</td>
      <td>2.01</td>
      <td>2259</td>
      <td>2.03</td>
      <td>2280</td>
      <td>1.12</td>
      <td>1255</td>
      <td>1.11</td>
      <td>1251</td>
      <td>0.76</td>
      <td>852</td>
      <td>0.24</td>
      <td>267</td>
      <td>0.07</td>
      <td>81</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x4</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-70B</th>
      <td>---</td>
      <td>10.41</td>
      <td>5235</td>
      <td>5.10</td>
      <td>2565</td>
      <td>5.50</td>
      <td>2766</td>
      <td>4.36</td>
      <td>2193</td>
      <td>1.49</td>
      <td>751</td>
      <td>1.21</td>
      <td>607</td>
      <td>0.89</td>
      <td>447</td>
      <td>0.19</td>
      <td>98</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8</th>
      <td>1.63</td>
      <td>18.11</td>
      <td>9103</td>
      <td>8.90</td>
      <td>4477</td>
      <td>9.41</td>
      <td>4730</td>
      <td>7.42</td>
      <td>3731</td>
      <td>2.44</td>
      <td>1229</td>
      <td>1.89</td>
      <td>948</td>
      <td>1.26</td>
      <td>631</td>
      <td>0.30</td>
      <td>149</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16</th>
      <td>1.12</td>
      <td>12.63</td>
      <td>6353</td>
      <td>5.32</td>
      <td>2673</td>
      <td>5.58</td>
      <td>2804</td>
      <td>4.27</td>
      <td>2144</td>
      <td>2.30</td>
      <td>1158</td>
      <td>1.45</td>
      <td>729</td>
      <td>0.76</td>
      <td>381</td>
      <td>0.22</td>
      <td>110</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x4</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-70B</th>
      <td>---</td>
      <td>14.04</td>
      <td>2113</td>
      <td>10.85</td>
      <td>1634</td>
      <td>12.25</td>
      <td>1844</td>
      <td>9.93</td>
      <td>1494</td>
      <td>3.68</td>
      <td>554</td>
      <td>2.82</td>
      <td>425</td>
      <td>1.81</td>
      <td>273</td>
      <td>0.35</td>
      <td>52</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic</th>
      <td>1.78</td>
      <td>41.44</td>
      <td>6236</td>
      <td>19.64</td>
      <td>2956</td>
      <td>21.03</td>
      <td>3166</td>
      <td>16.72</td>
      <td>2516</td>
      <td>6.01</td>
      <td>904</td>
      <td>4.46</td>
      <td>672</td>
      <td>2.55</td>
      <td>383</td>
      <td>0.49</td>
      <td>74</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16</th>
      <td>1.45</td>
      <td>36.61</td>
      <td>5509</td>
      <td>15.12</td>
      <td>2275</td>
      <td>16.24</td>
      <td>2443</td>
      <td>13.22</td>
      <td>1990</td>
      <td>5.48</td>
      <td>825</td>
      <td>3.01</td>
      <td>453</td>
      <td>2.07</td>
      <td>312</td>
      <td>0.43</td>
      <td>64</td>
    </tr>
  </tbody>
</table>

**Use case profiles: prompt tokens / generation tokens

**QPS: Queries per second.

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).

