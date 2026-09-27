---
tags:
- fp4
- vllm
- llm-compressor
- compressed-tensors
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-31B-it
provider: Google
name: RedHatAI/gemma-4-31B-it-NVFP4
description: NVFP4 variant of gemma-4-31B-it.
readme: https://huggingface.co/RedHatAI/gemma-4-31B-it-NVFP4/blob/main/README.md
tool_calling_supported: true
required_cli_args: ['--reasoning-parser gemma4', '--enable-prefix-caching']
default-chat-template-kwargs: '{"enable_thinking": true}'
chat_template_file_name: None
chat_template_path: None
tool_call_parser: gemma4
validated_tasks:
- tool-calling
tasks:
- text-to-text
- text-generation
- tool-calling
validated_on:
  - RHOAI 3.5
  - RHAIIS 3.5
  - vLLM 0.24.0
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  gemma-4-31B-it-NVFP4
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>

## Model Overview
- **Model Architecture:** Gemma4ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Release Date:** 2026-04-04
- **Version:** 1.0
- **Model Developers:** RedHatAI

This model is a quantized version of [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it).
It was evaluated on several tasks to assess its quality in comparison to the unquantized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) to FP4 data type using the NVFP4 format, ready for inference with vLLM.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Weights are quantized with FP4 (group_size=16), and activations are quantized with FP4 using local per-group scaling. Only the weights and activations of the linear operators within transformer blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor). Vision tower, embedding, and output head layers are kept in their original precision.

## Deployment

### Use with vLLM

This model can be deployed using [vLLM](https://docs.vllm.ai/en/latest/).
For detailed instructions including multi-GPU deployment, multimodal inference, thinking mode, function calling, and benchmarking, see the [Gemma 4 vLLM usage guide](https://recipes.vllm.ai/Google/gemma-4-31B-it).

1. Start the vLLM server:
```
vllm serve RedHatAI/gemma-4-31B-it-NVFP4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
  --async-scheduling
```

> **Tip:** For text-only workloads, pass `--limit-mm-per-prompt '{"image": 0, "audio": 0}'` to skip vision encoder memory allocation and free up GPU memory for a longer context window.

2. Send requests to the server:

```python
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://<your-server-host>:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

model = "RedHatAI/gemma-4-31B-it-NVFP4"

messages = [
    {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
]

outputs = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

generated_text = outputs.choices[0].message.content
print(generated_text)
```

## Creation

This model was created by applying NVFP4 quantization with [LLM Compressor](https://github.com/vllm-project/llm-compressor), as presented in the code snippet below.

<details>

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import apply
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = "google/gemma-4-31B-it"
SAVE_DIR = MODEL_ID.split("/")[1] + "-NVFP4"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

ds = load_dataset("mgoin/ultrachat_200k_s3", split="train_sft")
calibration_data = [ex["prompt"] for ex in ds.select(range(512))]

recipe = QuantizationModifier(
    targets=["Linear"],
    ignore=["re:.*vision.*", "re:.*audio.*", "lm_head", "re:.*embed.*"],
    scheme="NVFP4",
)

apply(model=model, tokenizer=tokenizer, recipe=recipe, calibration_data=calibration_data)
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)
```

</details>

## Evaluation

This model was evaluated on GSM8K Platinum, MMLU-Pro, IFEval, MATH-500, AIME 2025, GPQA Diamond, LiveCodeBench v6, and [BFCLv4](https://gorilla.cs.berkeley.edu/leaderboard.html) (function calling) using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness), [lighteval](https://github.com/neuralmagic/lighteval), and BFCL — all served with [vLLM](https://docs.vllm.ai/en/latest/) (OpenAI-compatible API). Accuracy results are reported both without and with thinking enabled; BFCLv4 was evaluated with thinking enabled.

### Accuracy

#### Without thinking

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>google/gemma-4-31B-it</th>
      <th>RedHatAI/gemma-4-31B-it-NVFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Instruction Following</b></td>
      <td>IFEval (0-shot, prompt-level strict)</td>
      <td>90.70</td>
      <td>89.77</td>
      <td>99.0%</td>
    </tr>
    <tr>
      <td>IFEval (0-shot, inst-level strict)</td>
      <td>93.45</td>
      <td>93.05</td>
      <td>99.6%</td>
    </tr>
    <tr>
      <td rowspan="5"><b>Reasoning</b></td>
      <td>GSM8K Platinum (0-shot, strict-match)</td>
      <td>95.78</td>
      <td>95.70</td>
      <td>99.9%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (0-shot, custom-extract)</td>
      <td>85.41</td>
      <td>84.50</td>
      <td>98.9%</td>
    </tr>
    <tr>
      <td>MATH-500 (0-shot, pass@1)</td>
      <td>89.40</td>
      <td>85.07</td>
      <td>95.2%</td>
    </tr>
    <tr>
      <td>AIME 2025 (0-shot, pass@1)</td>
      <td>65.83</td>
      <td>65.00</td>
      <td>98.7%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (0-shot, pass@1)</td>
      <td>77.44</td>
      <td>76.60</td>
      <td>98.9%</td>
    </tr>
    <tr>
      <td><b>Coding</b></td>
      <td>LiveCodeBench v6 (0-shot, pass@1)</td>
      <td>71.43</td>
      <td>70.67</td>
      <td>98.9%</td>
    </tr>
  </tbody>
</table>

#### With thinking

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Benchmark</th>
      <th>google/gemma-4-31B-it</th>
      <th>RedHatAI/gemma-4-31B-it-NVFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>Instruction Following</b></td>
      <td>IFEval (0-shot, prompt-level strict)</td>
      <td>94.58</td>
      <td>93.96</td>
      <td>99.3%</td>
    </tr>
    <tr>
      <td>IFEval (0-shot, inst-level strict)</td>
      <td>96.44</td>
      <td>95.96</td>
      <td>99.5%</td>
    </tr>
    <tr>
      <td rowspan="5"><b>Reasoning</b></td>
      <td>GSM8K Platinum (0-shot, strict-match)</td>
      <td>96.11</td>
      <td>95.95</td>
      <td>99.8%</td>
    </tr>
    <tr>
      <td>MMLU-Pro (0-shot, custom-extract)</td>
      <td>87.05</td>
      <td>86.38</td>
      <td>99.2%</td>
    </tr>
    <tr>
      <td>MATH-500 (0-shot, pass@1)</td>
      <td>87.93</td>
      <td>87.00</td>
      <td>98.9%</td>
    </tr>
    <tr>
      <td>AIME 2025 (0-shot, pass@1)</td>
      <td>87.08</td>
      <td>86.25</td>
      <td>99.0%</td>
    </tr>
    <tr>
      <td>GPQA Diamond (0-shot, pass@1)</td>
      <td>86.36</td>
      <td>87.04</td>
      <td>100.8%</td>
    </tr>
    <tr>
      <td><b>Coding</b></td>
      <td>LiveCodeBench v6 (0-shot, pass@1)</td>
      <td>74.29</td>
      <td>77.90</td>
      <td>104.9%</td>
    </tr>
    <tr>
      <td rowspan="4"><b>Tool Calling</b></td>
      <td>BFCLv4 Overall</td>
      <td>70.77%</td>
      <td>71.24%</td>
      <td>100.7%</td>
    </tr>
    <tr>
      <td>BFCLv4 Single Turn</td>
      <td>84.99%</td>
      <td>84.92%</td>
      <td>99.9%</td>
    </tr>
    <tr>
      <td>BFCLv4 Multi-Turn</td>
      <td>66.25%</td>
      <td>68.50%</td>
      <td>103.4%</td>
    </tr>
    <tr>
      <td>BFCLv4 Agentic</td>
      <td>65.82%</td>
      <td>64.50%</td>
      <td>98.0%</td>
    </tr>
  </tbody>
</table>

### Reproduction

The results were obtained using the following commands:

<details>

Each benchmark was run 3 times with different random seeds (1234, 2345, 3456) and the scores were averaged; AIME 2025 used 8 seeds.

**vLLM server:**
```
vllm serve RedHatAI/gemma-4-31B-it-NVFP4 \
  --served-model-name gemma-4-31b-it-NVFP4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --async-scheduling \
  --default-chat-template-kwargs '{"enable_thinking": true}'
```

> **Note:** To reproduce the results without thinking, remove `--default-chat-template-kwargs '{"enable_thinking": true}'`. To run without tool calling, remove `--enable-auto-tool-choice`, `--tool-call-parser gemma4`, and `--reasoning-parser gemma4`.

#### GSM8K Platinum (lm-eval, 0-shot, 3 repetitions)
```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=gemma-4-31b-it-NVFP4,max_length=32768,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_gsm8k_platinum.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=32000,seed=1234"
```

#### MMLU-Pro (lm-eval, 0-shot, 3 repetitions)
```
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=gemma-4-31b-it-NVFP4,max_length=32768,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_mmlu_pro.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=32000,seed=1234"
```

#### IFEval (lm-eval, 0-shot, 3 repetitions)
```
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=gemma-4-31b-it-NVFP4,max_length=32768,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results_ifeval.json \
  --seed 1234 \
  --gen_kwargs "do_sample=True,temperature=1.0,top_p=0.95,top_k=64,max_gen_toks=32000,seed=1234"
```

#### MATH-500, AIME 2025, GPQA Diamond, LiveCodeBench v6 (lighteval, 3 repetitions; 8 for AIME 2025)

**litellm_config.yaml:**
```yaml
model_parameters:
  provider: hosted_vllm
  model_name: hosted_vllm/gemma-4-31b-it-NVFP4
  base_url: http://0.0.0.0:8000/v1
  api_key: ''
  timeout: 3600
  concurrent_requests: 32
  generation_parameters:
    temperature: 1.0
    max_new_tokens: 65536
    top_p: 0.95
    top_k: 64
    seed: 1234
```

Run once per seed (changing `seed` in the config each time):
```
lighteval endpoint litellm litellm_config.yaml 'math_500|0' \
  --output-dir results/ --save-details

lighteval endpoint litellm litellm_config.yaml 'aime25|0' \
  --output-dir results/ --save-details

lighteval endpoint litellm litellm_config.yaml 'gpqa:diamond|0' \
  --output-dir results/ --save-details

lighteval endpoint litellm litellm_config.yaml 'lcb:codegeneration_v6|0' \
  --output-dir results/ --save-details
```

#### BFCLv4

BFCL requires the model to be registered in the leaderboard codebase before running evaluation.

**Step 1 — Register the model in `bfcl_eval/constants/model_config.py`**

Add the following entry to `api_inference_model_map`:

```python
"gemma-4-31b-it-NVFP4": ModelConfig(
    model_name="gemma-4-31b-it-NVFP4",
    display_name="Gemma-4-31b-it-NVFP4 (FC)",
    url="https://huggingface.co/RedHatAI/gemma-4-31B-it-NVFP4",
    org="Google",
    license="Apache 2.0",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,
),
```

**Step 2 — Add the key to `bfcl_eval/constants/supported_models.py`**

Add `"gemma-4-31b-it-NVFP4"` to the `SUPPORTED_MODELS` list.

**Step 3 — Start the vLLM server** (use the command at the top of this section; the `--served-model-name` flag ensures BFCL can find the model by its registered slug).

**Step 4 — Generate responses and evaluate**
```
bfcl generate --model gemma-4-31b-it-NVFP4 --test-category all
bfcl evaluate --model gemma-4-31b-it-NVFP4 --test-category all
```

</details>
