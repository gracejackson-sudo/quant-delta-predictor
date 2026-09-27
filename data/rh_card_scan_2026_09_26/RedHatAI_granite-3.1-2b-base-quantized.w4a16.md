---
tags:
- w4a16
- int4
- vllm
license: apache-2.0
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
  - en
base_model: ibm-granite/granite-3.1-2b-base
library_name: transformers
---

# granite-3.1-2b-base-quantized.w4a16

## Model Overview
- **Model Architecture:** granite-3.1-2b-base
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** INT4
- **Release Date:** 1/8/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [ibm-granite/granite-3.1-2b-base](https://huggingface.co/ibm-granite/granite-3.1-2b-base).
It achieves an average score of 61.54 on the [OpenLLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) benchmark (version 1), whereas the unquantized model achieves 61.98.

### Model Optimizations

This model was obtained by quantizing the weights of [ibm-granite/granite-3.1-2b-base](https://huggingface.co/ibm-granite/granite-3.1-2b-base) to INT4 data type, ready for inference with vLLM >= 0.5.2.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%. Only the weights of the linear operators within transformers blocks are quantized. 

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

max_model_len, tp_size = 4096, 1
model_name = "neuralmagic/granite-3.1-2b-base-quantized.w4a16"
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
python quantize.py --model_path ibm-granite/granite-3.1-2b-base --quant_path "output_dir/granite-3.1-2b-base-quantized.w4a16" --calib_size 1024 --dampening_frac 0.01 --observer mse
```


```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot, apply
import argparse
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--quant_path', type=str)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.1) 
parser.add_argument('--observer', type=str, default="minmax")
parser.add_argument('--group_size', type=int, default="128")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype="auto",
    use_cache=False,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)


NUM_CALIBRATION_SAMPLES = args.calib_size
DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"
ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

def preprocess(example):
    return {"text": example["text"]}

ds = ds.map(preprocess)

def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=["lm_head"],
        scheme="w4a16",
        dampening_frac=args.dampening_frac,
        observer=args.observer,
    )
]
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=args.calib_size,
    max_seq_length=8196,
)

# Save to disk compressed.
model.save_pretrained(quant_path, save_compressed=True)
tokenizer.save_pretrained(quant_path)
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
  --model_args pretrained="neuralmagic/granite-3.1-2b-base-quantized.w4a16",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=1,gpu_memory_utilization=0.8,enable_chunked_prefill=True,trust_remote_code=True \
  --tasks openllm \
  --write_out \
  --batch_size auto \
  --output_path output_dir \
  --show_config
```

#### HumanEval
##### Generation
```
python3 codegen/generate.py \
  --model neuralmagic/granite-3.1-2b-base-quantized.w4a16 \
  --bs 16 \
  --temperature 0.2 \
  --n_samples 50 \
  --root "." \
  --dataset humaneval
```
##### Sanitization
```
python3 evalplus/sanitize.py \
  humaneval/neuralmagic--granite-3.1-2b-base-quantized.w4a16_vllm_temp_0.2
```
##### Evaluation
```
evalplus.evaluate \
  --dataset humaneval \
  --samples humaneval/neuralmagic--granite-3.1-2b-base-quantized.w4a16_vllm_temp_0.2-sanitized
```
</details>

### Accuracy

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>ibm-granite/granite-3.1-2b-base</th>
      <th>neuralmagic/granite-3.1-2b-base-quantized.w4a16</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="7"><b>OpenLLM V1</b></td>
      <td>ARC-Challenge (Acc-Norm, 25-shot)</td>
      <td>53.75</td>
      <td>51.96</td>
      <td>96.67</td>
    </tr>
    <tr>
      <td>GSM8K (Strict-Match, 5-shot)</td>
      <td>47.84</td>
      <td>42.53</td>
      <td>88.89</td>
    </tr>
    <tr>
      <td>HellaSwag (Acc-Norm, 10-shot)</td>
      <td>77.94</td>
      <td>75.38</td>
      <td>96.71</td>
    </tr>
    <tr>
      <td>MMLU (Acc, 5-shot)</td>
      <td>52.88</td>
      <td>51.09</td>
      <td>96.61</td>
    </tr>
    <tr>
      <td>TruthfulQA (MC2, 0-shot)</td>
      <td>39.04</td>
      <td>41.35</td>
      <td>105.93</td>
    </tr>
    <tr>
      <td>Winogrande (Acc, 5-shot)</td>
      <td>74.43</td>
      <td>74.27</td>
      <td>99.78</td>
    </tr>
    <tr>
      <td><b>Average Score</b></td>
      <td><b>57.65</b></td>
      <td><b>56.10</b></td>
      <td><b>97.31</b></td>
    </tr>
    <tr>
      <td rowspan="2"><b>Coding</b></td>
      <td>HumanEval Pass@1</td>
      <td>30.00</td>
      <td>29.80</td>
      <td><b>99.33</b></td>
    </tr>
  </tbody>
</table>



## Inference Performance


This model achieves up to 1.9x speedup in single-stream deployment, depending on hardware and use-case scenario.
The following performance benchmarks were conducted with [vLLM](https://docs.vllm.ai/en/latest/) version 0.6.6.post1, and [GuideLLM](https://github.com/neuralmagic/guidellm).

<details>
<summary>Benchmarking Command</summary>

```
guidellm --model neuralmagic/granite-3.1-2b-base-quantized.w4a16 --target "http://localhost:8000/v1" --data-type emulated --data "prompt_tokens=<prompt_tokens>,generated_tokens=<generated_tokens>" --max seconds 360 --backend aiohttp_server
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
    <td style="vertical-align: middle;" rowspan="3" >A5000</td>
    <td>granite-3.1-2b-base</td>
    <td></td>
    <td>10.9</td>
    <td>1.4</td>
    <td>11.0</td>
    <td>1.5</td>
    <td>1.4</td>
    <td>2.8</td>
    <td>6.1</td>
  </tr>
  <tr>
    <td>granite-3.1-2b-base-quantized.w8a8</td>
    <td>1.37</td>
    <td>7.9</td>
    <td>1.0</td>
    <td>8.0</td>
    <td>1.1</td>
    <td>1.0</td>
    <td>2.0</td>
    <td>4.7</td>
  </tr>
  <tr>
    <td>granite-3.1-2b-base-quantized.w4a16<br>(this model)</td>
    <td>1.94</td>
    <td>5.4</td>
    <td>0.7</td>
    <td>5.5</td>
    <td>0.8</td>
    <td>0.7</td>
    <td>1.4</td>
    <td>3.4</td>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >A6000</td>
    <td>granite-3.1-2b-base</td>
    <td></td>
    <td>9.8</td>
    <td>1.3</td>
    <td>10.0</td>
    <td>1.3</td>
    <td>1.3</td>
    <td>2.6</td>
    <td>5.4</td>
  </tr>
  <tr>
    <td>granite-3.1-2b-base-quantized.w8a8</td>
    <td>1.31</td>
    <td>7.8</td>
    <td>1.0</td>
    <td>7.6</td>
    <td>1.0</td>
    <td>0.9</td>
    <td>1.9</td>
    <td>4.5</td>
  </tr>
  <tr>
    <td>granite-3.1-2b-base-quantized.w4a16<br>(this model)</td>
    <td>1.87</td>
    <td>5.1</td>
    <td>0.7</td>
    <td>5.2</td>
    <td>0.7</td>
    <td>0.7</td>
    <td>1.3</td>
    <td>3.1</td>
  </tr>
  <tr>
    <td style="vertical-align: middle;" rowspan="3" >L40</td>
    <td>granite-3.1-2b-base</td>
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
    <td>granite-3.1-2b-base-FP8-dynamic</td>
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
    <td>granite-3.1-2b-base-quantized.w4a16<br>(this model)</td>
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
