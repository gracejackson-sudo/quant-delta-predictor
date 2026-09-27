---
license: mit
tags:
- deepseek
- int4
- vllm
- llmcompressor
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
library_name: transformers
---

# DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16

## Model Overview
- **Model Architecture:** Qwen2ForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Release Date:** 2/4/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).


### Model Optimizations

This model was obtained by quantizing the weights of [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) to INT4 data type.
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
model_name = "neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16"

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
model_stub = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</th>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td rowspan="4"><b>Reasoning</b></td>
<td>AIME 2024 (pass@1)</td>
<td>53.17</td>
<td>50.88</td>
<td>95.69%</td>
</tr>
<tr>
<td>MATH-500 (pass@1)</td>
<td>93.66</td>
<td>93.33</td>
<td>99.65%</td>
</tr>
<tr>
<td>GPQA Diamond (pass@1)</td>
<td>50.53</td>
<td>49.84</td>
<td>98.63%</td>
</tr>
<tr>
<td><b>Average Score</b></td>
<td><b>65.79</b></td>
<td><b>64.68</b></td>
<td><b>98.31%</b></td>
</tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>50.51</td>
      <td></td>
      <td>%</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>78.62</td>
      <td></td>
      <td>%</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>61.90</td>
      <td></td>
      <td>%</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>54.19</td>
      <td></td>
      <td>%</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>45.55</td>
      <td></td>
      <td>%</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>61.56</td>
      <td></td>
      <td>%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>58.72</b></td>
      <td><b></b></td>
      <td><b>%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>39.38</td>
      <td>39.38</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>6.97</td>
      <td>7.48</td>
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
      <td>1.81</td>
      <td>0.99</td>
      <td>---</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>4.68</td>
      <td>3.80</td>
      <td>---</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>1.66</td>
      <td>1.74</td>
      <td>---</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>9.08</b></td>
      <td><b>8.90</b></td>
      <td><b>---</b></td>
    </tr>
    <tr>
      <td rowspan="4"><b>Coding</b></td>
      <td>HumanEval (pass@1)</td>
      <td>40.80</td>
      <td>48.50</td>
      <td><b>118.9%</b></td>
    </tr>
    <tr>
      <td>HumanEval (pass@10)</td>
      <td>64.40</td>
      <td>67.10</td>
      <td>104.2%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>38.50</td>
      <td>45.00</td>
      <td>116.9%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>60.40</td>
      <td>64.10</td>
      <td>106.1%</td>
    </tr>
  </tbody>
</table>


## Inference Performance


This model achieves up to 2.6x speedup in single-stream deployment and up to 1.5x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16 --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</th>
      <td>---</td>
      <td>2.9</td>
      <td>1576</td>
      <td>5.7</td>
      <td>788</td>
      <td>2.9</td>
      <td>1535</td>
      <td>3.0</td>
      <td>1496</td>
      <td>22.6</td>
      <td>199</td>
      <td>23.2</td>
      <td>194</td>
      <td>12.1</td>
      <td>370</td>
      <td>38.5</td>
      <td>117</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8</th>
      <td>1.56</td>
      <td>1.8</td>
      <td>2495</td>
      <td>3.7</td>
      <td>1223</td>
      <td>1.9</td>
      <td>2384</td>
      <td>1.9</td>
      <td>2393</td>
      <td>14.3</td>
      <td>315</td>
      <td>14.8</td>
      <td>304</td>
      <td>7.9</td>
      <td>572</td>
      <td>25.3</td>
      <td>178</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16</th>
      <td>2.41</td>
      <td>1.1</td>
      <td>4086</td>
      <td>2.3</td>
      <td>1998</td>
      <td>1.2</td>
      <td>3783</td>
      <td>1.3</td>
      <td>3527</td>
      <td>8.6</td>
      <td>526</td>
      <td>8.8</td>
      <td>512</td>
      <td>5.2</td>
      <td>860</td>
      <td>22.7</td>
      <td>198</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</th>
      <td>---</td>
      <td>1.4</td>
      <td>1389</td>
      <td>2.9</td>
      <td>691</td>
      <td>1.5</td>
      <td>1358</td>
      <td>1.5</td>
      <td>1329</td>
      <td>11.5</td>
      <td>175</td>
      <td>11.6</td>
      <td>174</td>
      <td>6.2</td>
      <td>326</td>
      <td>21.5</td>
      <td>93</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8</th>
      <td>1.28</td>
      <td>1.1</td>
      <td>1850</td>
      <td>2.2</td>
      <td>905</td>
      <td>1.1</td>
      <td>1807</td>
      <td>1.1</td>
      <td>1750</td>
      <td>8.6</td>
      <td>233</td>
      <td>8.7</td>
      <td>230</td>
      <td>4.7</td>
      <td>431</td>
      <td>23.1</td>
      <td>87</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16</th>
      <td>1.72</td>
      <td>0.8</td>
      <td>2575</td>
      <td>1.5</td>
      <td>1298</td>
      <td>0.8</td>
      <td>2461</td>
      <td>0.8</td>
      <td>2382</td>
      <td>6.1</td>
      <td>331</td>
      <td>6.2</td>
      <td>323</td>
      <td>3.6</td>
      <td>566</td>
      <td>22.7</td>
      <td>89</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</th>
      <td>---</td>
      <td>0.9</td>
      <td>1161</td>
      <td>1.9</td>
      <td>579</td>
      <td>1.0</td>
      <td>1138</td>
      <td>1.0</td>
      <td>1121</td>
      <td>7.5</td>
      <td>146</td>
      <td>7.6</td>
      <td>145</td>
      <td>3.9</td>
      <td>279</td>
      <td>15.4</td>
      <td>71</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic</th>
      <td>1.34</td>
      <td>0.7</td>
      <td>1585</td>
      <td>1.4</td>
      <td>786</td>
      <td>0.7</td>
      <td>1577</td>
      <td>0.7</td>
      <td>1524</td>
      <td>5.3</td>
      <td>207</td>
      <td>5.5</td>
      <td>197</td>
      <td>2.9</td>
      <td>382</td>
      <td>14.3</td>
      <td>77</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16</th>
      <td>1.33</td>
      <td>0.7</td>
      <td>1590</td>
      <td>1.4</td>
      <td>793</td>
      <td>0.7</td>
      <td>1549</td>
      <td>0.7</td>
      <td>1509</td>
      <td>5.4</td>
      <td>201</td>
      <td>5.5</td>
      <td>198</td>
      <td>2.9</td>
      <td>381</td>
      <td>14.0</td>
      <td>78</td>
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</th>
      <td>---</td>
      <td>14.9</td>
      <td>67138</td>
      <td>7.1</td>
      <td>32094</td>
      <td>7.4</td>
      <td>33096</td>
      <td>5.9</td>
      <td>26480</td>
      <td>2.0</td>
      <td>9004</td>
      <td>1.5</td>
      <td>6639</td>
      <td>1.1</td>
      <td>4938</td>
      <td>0.3</td>
      <td>1151</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8</th>
      <td>1.36</td>
      <td>20.2</td>
      <td>90956</td>
      <td>8.8</td>
      <td>39786</td>
      <td>10.2</td>
      <td>45963</td>
      <td>8.1</td>
      <td>36596</td>
      <td>3.1</td>
      <td>13968</td>
      <td>2.1</td>
      <td>9629</td>
      <td>1.4</td>
      <td>6374</td>
      <td>0.3</td>
      <td>1429</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16</th>
      <td>1.00</td>
      <td>13.3</td>
      <td>59681</td>
      <td>6.1</td>
      <td>27633</td>
      <td>5.9</td>
      <td>26689</td>
      <td>4.7</td>
      <td>20944</td>
      <td>2.9</td>
      <td>13108</td>
      <td>1.9</td>
      <td>8355</td>
      <td>1.0</td>
      <td>4362</td>
      <td>0.3</td>
      <td>1170</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</th>
      <td>---</td>
      <td>26.4</td>
      <td>53073</td>
      <td>13.0</td>
      <td>26213</td>
      <td>14.5</td>
      <td>29110</td>
      <td>11.4</td>
      <td>22936</td>
      <td>4.4</td>
      <td>8749</td>
      <td>3.3</td>
      <td>6680</td>
      <td>2.3</td>
      <td>4634</td>
      <td>0.5</td>
      <td>1105</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w8a8</th>
      <td>1.27</td>
      <td>34.3</td>
      <td>69009</td>
      <td>14.8</td>
      <td>29791</td>
      <td>19.0</td>
      <td>38214</td>
      <td>15.7</td>
      <td>31598</td>
      <td>5.6</td>
      <td>11186</td>
      <td>4.2</td>
      <td>8350</td>
      <td>3.0</td>
      <td>6020</td>
      <td>0.7</td>
      <td>1328</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16</th>
      <td>0.93</td>
      <td>23.9</td>
      <td>47993</td>
      <td>12.0</td>
      <td>24194</td>
      <td>12.5</td>
      <td>25239</td>
      <td>10.0</td>
      <td>20029</td>
      <td>4.5</td>
      <td>9055</td>
      <td>3.3</td>
      <td>6681</td>
      <td>2.1</td>
      <td>4156</td>
      <td>0.5</td>
      <td>1043</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Qwen-7B</th>
      <td>---</td>
      <td>54.3</td>
      <td>59410</td>
      <td>26.0</td>
      <td>28440</td>
      <td>32.1</td>
      <td>35154</td>
      <td>26.7</td>
      <td>29190</td>
      <td>8.0</td>
      <td>8700</td>
      <td>6.6</td>
      <td>7275</td>
      <td>5.2</td>
      <td>5669</td>
      <td>1.2</td>
      <td>1266</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-FP8-dynamic</th>
      <td>1.16</td>
      <td>62.9</td>
      <td>68818</td>
      <td>30.3</td>
      <td>33196</td>
      <td>39.4</td>
      <td>43132</td>
      <td>31.1</td>
      <td>34073</td>
      <td>9.2</td>
      <td>10058</td>
      <td>7.1</td>
      <td>7748</td>
      <td>6.1</td>
      <td>6714</td>
      <td>1.3</td>
      <td>1415</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16</th>
      <td>1.02</td>
      <td>56.2</td>
      <td>61483</td>
      <td>26.7</td>
      <td>29243</td>
      <td>32.5</td>
      <td>35592</td>
      <td>26.9</td>
      <td>29461</td>
      <td>8.3</td>
      <td>9072</td>
      <td>6.4</td>
      <td>7027</td>
      <td>5.2</td>
      <td>5731</td>
      <td>1.2</td>
      <td>1291</td>
    </tr>
  </tbody>
</table>

**Use case profiles: prompt tokens / generation tokens

**QPS: Queries per second.

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).