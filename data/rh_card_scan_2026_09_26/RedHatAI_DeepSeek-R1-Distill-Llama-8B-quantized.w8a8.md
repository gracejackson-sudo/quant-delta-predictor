---
license: mit
tags:
- deepseek
- int8
- vllm
- llmcompressor
base_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
library_name: transformers
---

# DeepSeek-R1-Distill-Llama-8B-quantized.w8a8

## Model Overview
- **Model Architecture:** LlamaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT8
  - **Activation quantization:** INT8
- **Release Date:** 2/1/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B).


### Model Optimizations

This model was obtained by quantizing the weights and activations of [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) to INT8 data type.
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
model_name = "neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8"

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
model_stub = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_name = model_stub.split("/")[-1]

num_samples = 1024
max_seq_len = 8192

tokenizer = AutoTokenizer.from_pretrained(model_stub)

model = AutoModelForCausalLM.from_pretrained(
    model_stub,
    device_map=device_map,
    torch_dtype="auto",
)

def preprocess_fn(example):
  return {"text": tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.map(preprocess_fn)

# Configure the quantization algorithm and scheme
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
  --model_args pretrained="neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8",dtype=auto,max_model_len=4096,tensor_parallel_size=1,enable_chunked_prefill=True \
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</th>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4"><b>Reasoning</b></td>
      <td>AIME 2024 (pass@1)</td>
      <td>49.25</td>
      <td>49.07</td>
      <td>99.63%</td>
    </tr>
    <tr>
      <td>MATH-500 (pass@1)</td>
      <td>90.18</td>
      <td>90.04</td>
      <td>99.84%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (pass@1)</td>
      <td>49.27</td>
      <td>48.9</td>
      <td>98.25%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>62.9</b></td>
      <td><b>62.67</b></td>
      <td><b>99.63%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>45.05</td>
      <td>45.22</td>
      <td>100.4%</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>62.77</td>
      <td>62.09</td>
      <td>98.9%</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>76.78</td>
      <td>76.80</td>
      <td>100.0%</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>55.65</td>
      <td>55.53</td>
      <td>99.8%</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>50.55</td>
      <td>49.89</td>
      <td>98.7%</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>68.51</td>
      <td>67.40</td>
      <td>98.4%</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>59.88</b></td>
      <td><b>59.49</b></td>
      <td><b>99.3%</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>IFEval (Inst Level Strict Acc, 0-shot)</td>
      <td>38.37</td>
      <td>38.40</td>
      <td>100.1%</td>
    </tr>
    <tr>
      <td>BBH (Acc-Norm, 3-shot)</td>
      <td>7.43</td>
      <td>7.66</td>
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
      <td>1.51</td>
      <td>1.47</td>
      <td>---</td>
    </tr>
    <tr>
      <td>MUSR (Acc-Norm, 0-shot)</td>
      <td>1.86</td>
      <td>1.27</td>
      <td>---</td>
    </tr>
    <tr>
      <td>MMLU-Pro (Acc, 5-shot)</td>
      <td>1.61</td>
      <td>1.48</td>
      <td>---</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>8.47</b></td>
      <td><b>8.38</b></td>
      <td><b>---</b></td>
    </tr>
    <tr>
      <td rowspan="4"><b>Coding</b></td>
      <td>HumanEval (pass@1)</td>
      <td>49.90</td>
      <td>50.90</td>
      <td><b>102.0%</b></td>
    </tr>
    <tr>
      <td>HumanEval (pass@10)</td>
      <td>68.90</td>
      <td>68.70</td>
      <td>99.7%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>44.10</td>
      <td>46.70</td>
      <td>105.9%</td>
    </tr>
    <tr>
      <td>HumanEval+ (pass@10)</td>
      <td>62.90</td>
      <td>64.30</td>
      <td>102.2%</td>
    </tr>
  </tbody>
</table>

## Inference Performance


This model achieves up to 1.6x speedup in single-stream deployment and up to 1.4x speedup in multi-stream asynchronous deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.7.2, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8 --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</th>
      <td>---</td>
      <td>3.0</td>
      <td>1511</td>
      <td>6.0</td>
      <td>755</td>
      <td>3.0</td>
      <td>1483</td>
      <td>3.1</td>
      <td>1462</td>
      <td>23.6</td>
      <td>191</td>
      <td>24.0</td>
      <td>188</td>
      <td>12.7</td>
      <td>353</td>
      <td>41.1</td>
      <td>110</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8</th>
      <td>1.53</td>
      <td>1.9</td>
      <td>2356</td>
      <td>3.8</td>
      <td>1175</td>
      <td>2.0</td>
      <td>2291</td>
      <td>2.0</td>
      <td>2207</td>
      <td>15.2</td>
      <td>297</td>
      <td>15.5</td>
      <td>290</td>
      <td>8.5</td>
      <td>531</td>
      <td>28.6</td>
      <td>157</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16</th>
      <td>2.35</td>
      <td>1.2</td>
      <td>3870</td>
      <td>2.3</td>
      <td>1918</td>
      <td>1.3</td>
      <td>3492</td>
      <td>1.3</td>
      <td>3335</td>
      <td>9.1</td>
      <td>492</td>
      <td>9.5</td>
      <td>472</td>
      <td>5.8</td>
      <td>771</td>
      <td>22.7</td>
      <td>198</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</th>
      <td>---</td>
      <td>1.5</td>
      <td>1308</td>
      <td>3.1</td>
      <td>657</td>
      <td>1.6</td>
      <td>1274</td>
      <td>1.6</td>
      <td>1263</td>
      <td>12.1</td>
      <td>166</td>
      <td>12.4</td>
      <td>162</td>
      <td>6.5</td>
      <td>308</td>
      <td>25.6</td>
      <td>78</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8</th>
      <td>1.30</td>
      <td>1.1</td>
      <td>1763</td>
      <td>2.3</td>
      <td>882</td>
      <td>1.2</td>
      <td>1716</td>
      <td>1.2</td>
      <td>1698</td>
      <td>9.0</td>
      <td>223</td>
      <td>9.2</td>
      <td>218</td>
      <td>4.9</td>
      <td>409</td>
      <td>25.7</td>
      <td>78</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16</th>
      <td>1.76</td>
      <td>0.8</td>
      <td>2501</td>
      <td>1.6</td>
      <td>1236</td>
      <td>0.9</td>
      <td>2350</td>
      <td>0.9</td>
      <td>2287</td>
      <td>6.4</td>
      <td>316</td>
      <td>6.6</td>
      <td>306</td>
      <td>3.7</td>
      <td>544</td>
      <td>24.7</td>
      <td>82</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</th>
      <td>---</td>
      <td>1.0</td>
      <td>1146</td>
      <td>1.9</td>
      <td>574</td>
      <td>1.0</td>
      <td>1128</td>
      <td>1.0</td>
      <td>1111</td>
      <td>7.6</td>
      <td>144</td>
      <td>7.7</td>
      <td>142</td>
      <td>4.1</td>
      <td>266</td>
      <td>16.3</td>
      <td>67</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic</th>
      <td>1.25</td>
      <td>0.7</td>
      <td>1567</td>
      <td>1.4</td>
      <td>758</td>
      <td>0.7</td>
      <td>1484</td>
      <td>0.7</td>
      <td>1462</td>
      <td>5.7</td>
      <td>191</td>
      <td>5.8</td>
      <td>189</td>
      <td>3.2</td>
      <td>347</td>
      <td>22.5</td>
      <td>49</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16</th>
      <td>1.30</td>
      <td>0.7</td>
      <td>1527</td>
      <td>1.4</td>
      <td>768</td>
      <td>0.7</td>
      <td>1495</td>
      <td>0.7</td>
      <td>1463</td>
      <td>5.6</td>
      <td>194</td>
      <td>5.7</td>
      <td>190</td>
      <td>3.1</td>
      <td>350</td>
      <td>14.7</td>
      <td>74</td>
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
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</th>
      <td>---</td>
      <td>12.6</td>
      <td>56742</td>
      <td>5.7</td>
      <td>25687</td>
      <td>6.5</td>
      <td>29349</td>
      <td>5.2</td>
      <td>23259</td>
      <td>1.6</td>
      <td>7250</td>
      <td>1.2</td>
      <td>5181</td>
      <td>0.8</td>
      <td>3445</td>
      <td>0.1</td>
      <td>616</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8</th>
      <td>1.34</td>
      <td>17.4</td>
      <td>78101</td>
      <td>7.6</td>
      <td>34351</td>
      <td>8.8</td>
      <td>39790</td>
      <td>7.0</td>
      <td>31532</td>
      <td>2.3</td>
      <td>10405</td>
      <td>1.5</td>
      <td>6960</td>
      <td>1.0</td>
      <td>4355</td>
      <td>0.2</td>
      <td>785</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16</th>
      <td>0.91</td>
      <td>10.9</td>
      <td>48964</td>
      <td>5.1</td>
      <td>22989</td>
      <td>4.8</td>
      <td>21791</td>
      <td>3.8</td>
      <td>17039</td>
      <td>2.2</td>
      <td>9726</td>
      <td>1.2</td>
      <td>5434</td>
      <td>0.6</td>
      <td>2544</td>
      <td>0.1</td>
      <td>578</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">A100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</th>
      <td>---</td>
      <td>24.5</td>
      <td>49296</td>
      <td>11.3</td>
      <td>22657</td>
      <td>13.0</td>
      <td>26047</td>
      <td>10.5</td>
      <td>21020</td>
      <td>3.5</td>
      <td>7029</td>
      <td>2.5</td>
      <td>4995</td>
      <td>1.7</td>
      <td>3503</td>
      <td>0.3</td>
      <td>659</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w8a8</th>
      <td>1.27</td>
      <td>30.8</td>
      <td>62042</td>
      <td>14.1</td>
      <td>28419</td>
      <td>17.2</td>
      <td>34554</td>
      <td>13.8</td>
      <td>27719</td>
      <td>4.6</td>
      <td>9299</td>
      <td>3.1</td>
      <td>6215</td>
      <td>2.2</td>
      <td>4331</td>
      <td>0.4</td>
      <td>807</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16</th>
      <td>0.97</td>
      <td>22.7</td>
      <td>45708</td>
      <td>10.5</td>
      <td>21216</td>
      <td>11.1</td>
      <td>22353</td>
      <td>8.9</td>
      <td>17939</td>
      <td>3.9</td>
      <td>7758</td>
      <td>2.6</td>
      <td>5241</td>
      <td>1.6</td>
      <td>3196</td>
      <td>0.4</td>
      <td>718</td>
    </tr>
    <tr>
      <th rowspan="3" valign="top">H100x1</th>
      <th>deepseek-ai/DeepSeek-R1-Distill-Llama-8B</th>
      <td>---</td>
      <td>49.0</td>
      <td>53593</td>
      <td>22.6</td>
      <td>24750</td>
      <td>28.3</td>
      <td>30971</td>
      <td>22.9</td>
      <td>25035</td>
      <td>7.2</td>
      <td>7912</td>
      <td>5.1</td>
      <td>5561</td>
      <td>3.6</td>
      <td>3939</td>
      <td>0.6</td>
      <td>703</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic</th>
      <td>1.14</td>
      <td>57.1</td>
      <td>62517</td>
      <td>26.0</td>
      <td>28440</td>
      <td>34.5</td>
      <td>37781</td>
      <td>28.7</td>
      <td>31360</td>
      <td>7.2</td>
      <td>7877</td>
      <td>5.4</td>
      <td>5923</td>
      <td>4.3</td>
      <td>4697</td>
      <td>0.7</td>
      <td>782</td>
    </tr>
    <tr>
      <th>neuralmagic/DeepSeek-R1-Distill-Llama-8B-quantized.w4a16</th>
      <td>1.01</td>
      <td>49.8</td>
      <td>54452</td>
      <td>22.9</td>
      <td>25035</td>
      <td>28.5</td>
      <td>31162</td>
      <td>23.0</td>
      <td>25200</td>
      <td>6.8</td>
      <td>7493</td>
      <td>5.0</td>
      <td>5431</td>
      <td>3.7</td>
      <td>4079</td>
      <td>0.7</td>
      <td>787</td>
    </tr>
  </tbody>
</table>

**Use case profiles: prompt tokens / generation tokens

**QPS: Queries per second.

**QPD: Queries per dollar, based on on-demand cost at [Lambda Labs](https://lambdalabs.com/service/gpu-cloud) (observed on 2/18/2025).