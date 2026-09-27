---
library_name: transformers
license: llama3.1
base_model: RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4
datasets:
- trl-lib/tldr
---

# Sparse-Llama-3.1-8B-tldr-2of4-FP8-dynamic

## Model Overview
- **Model Architecture:** LlamaForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Sparsity:** 2:4 
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 06/04/2025
- **Version:** 1.0
- **Intended Use Cases:** This model is finetuned to summarize text in the style of Reddit posts.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in any other way that is prohibited by the Acceptable Use Policy and Llama 3.1 Community License.
- **Model Developers:** Red Hat (Neural Magic)

This model is a quantized version of [RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4](https://huggingface.co/RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4), which is fine-tuned on the [trl-lib/tldr](https://huggingface.co/datasets/trl-lib/tldr) dataset.
This sparse-quantized model recovers 100% of the BERTScore (0.366) obtained by the dense model [RedHatAI/Llama-3.1-8B-tldr](https://huggingface.co/RedHatAI/Llama-3.1-8B-tldr) while providing up to 1.6x speedup. 


## Deployment

This model can be deployed efficiently using [vLLM](https://docs.vllm.ai/en/latest/), as shown in the example below.

Run the following command to start the vLLM server:
```bash
vllm serve RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4-FP8-dynamic
```

Once your server is started, you can query the model using the OpenAI API:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

post="""
SUBREDDIT: r/AI

TITLE: Training sparse LLMs

POST: Now you can use the llm-compressor integration to axolotl to train sparse LLMs!

It's super easy to use. See the example in https://huggingface.co/RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4.

And there's more. You can run 2:4 sparse models on vLLM and get significant speedupts on Hopper GPUs!
"""

prompt = f"Give a TL;DR of the following Reddit post.\n<|user|>{post}\nTL;DR:\n<|assistant|>\n"

completion = client.completions.create(
  model="RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4-FP8-dynamic",
  prompt=prompt,
  max_tokens=256,
)
print("Completion result:", completion)
```

## Quantization

<details><summary>Quantization details</summary>

This model was created by applying [llm-compressor](https://github.com/vllm-project/llm-compressor), as presented in the code snipet below.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.transformers import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])

model_stub = "RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4"
model_name = model_stub.split("/")[-1]

model = AutoModelForCausalLM.from_pretrained(
    model_stub, torch_dtype="auto", device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_stub),

output_dir = f"./{model_name}-FP8-dynamic"

oneshot(
    model=model,
    recipe=recipe,
)

model.save_pretrained(output_dir, save_compressed=True, skip_sparsity_compression_stats=False)
tokenizer.save_pretrained(output_dir)
```
</details>


## Evaluation

The model was evaluated on the test split of [trl-lib/tldr](https://huggingface.co/datasets/trl-lib/tldr) using the Neural Magic fork of [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness/tree/tldr) (tldr branch).
One can reproduce these results by using the following command:

```bash
lm_eval --model vllm --model_args "pretrained=RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4-FP8-dynamic,dtype=auto,add_bos_token=True" --batch-size auto --tasks tldr
```

<table>
  <tr>
   <th>Metric
   </th>
   <th>Llama-3.1-8B-Instruct
   </th>
   <th>Llama-3.1-8B-tldr
   </th>
   <th>Sparse-Llama-3.1-8B-tldr-2of4-FP8-dynamic<br>(this model)
   </th>
  </tr>
  <tr>
   <td>BERTScore
   </td>
   <td>-0.230
   </td>
   <td>0.366
   </td>
   <td>0.366
   </td>
  </tr>
  <tr>
   <td>ROUGE-1
   </td>
   <td>0.059
   </td>
   <td>0.362
   </td>
   <td>0.354
   </td>
  </tr>
  <tr>
   <td>ROUGE-2
   </td>
   <td>0.018
   </td>
   <td>0.144
   </td>
   <td>0.140
   </td>
  </tr>
  <tr>
   <td>ROUGE-Lsum
   </td>
   <td>0.051
   </td>
   <td>0.306
   </td>
   <td>0.302
   </td>
  </tr>
</table>

## Inference Performance

We evaluated the inference performance of this model using the first 1,000 samples from the training set of the [trl-lib/tldr](https://huggingface.co/datasets/trl-lib/tldr) dataset.
Benchmarking was conducted with [vLLM](https://docs.vllm.ai/en/latest/) version `0.9.0.1` and [GuideLLM](https://github.com/neuralmagic/guidellm) version `0.2.1`.

The figure below presents the **mean end-to-end latency per request** across varying request rates.
Results are shown for this model, as well as two variants:
- **Dense:** [Llama-3.1-8B-tldr](https://huggingface.co/RedHatAI/Llama-3.1-8B-tldr)
- **Dense-quantized:** [Llama-3.1-8B-tldr-FP8-dynamic](https://huggingface.co/RedHatAI/Llama-3.1-8B-tldr-FP8-dynamic)

![Latency](./inference_performance/latency.png)


<details><summary><strong>Reproduction instructions</strong></summary>

To replicate the benchmark:

1. Generate a JSON file containing the first 1,000 training samples:
```python
from datasets import load_dataset
ds = load_dataset("trl-lib/tldr", split="train").take(1000)
ds.to_json("tldr_1000.json")
```

2. Start a vLLM server using your target model:
```bash
vllm serve RedHatAI/Sparse-Llama-3.1-8B-tldr-2of4-FP8-dynamic
```

3. Run the benchmark with GuideLLM:
```
GUIDELLM__OPENAI__MAX_OUTPUT_TOKENS=128 guidellm benchmark --target "http://localhost:8000" --rate-type sweep --data tldr_1000.json
```
> The average output length is approximately 30 tokens per sample. We capped the generation at 128 tokens to reduce performance skew from rare, unusually verbose completions.

</details>



