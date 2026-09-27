---
tags:
- FP8
- vllm
- audio
license: apache-2.0
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
  - en
base_model: openai/whisper-large-v2
library_name: transformers
---

# whisper-large-v2-FP8-Dynamic

## Model Overview
- **Model Architecture:** whisper-large-v2
  - **Input:** Audio-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 04/16/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2).

### Model Optimizations

This model was obtained by quantizing the weights of [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) to FP8 data type, ready for inference with vLLM >= 0.5.2.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.audio import AudioAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="neuralmagic/whisper-large-v2-FP8-Dynamic",
    max_model_len=448,
    max_num_seqs=400,
    limit_mm_per_prompt={"audio": 1},
)

# prepare inputs
inputs = {  # Test explicit encoder/decoder prompt
    "encoder_prompt": {
        "prompt": "",
        "multi_modal_data": {
            "audio": AudioAsset("winning_call").audio_and_sample_rate,
        },
    },
    "decoder_prompt": "<|startoftranscript|>",
}

# generate response
print("========== SAMPLE GENERATION ==============")
outputs = llm.generate(inputs, SamplingParams(temperature=0.0, max_tokens=64))
print(f"PROMPT  : {outputs[0].prompt}")
print(f"RESPONSE: {outputs[0].outputs[0].text}")
print("==========================================")
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 

<details>
  <summary>Model Creation Code</summary>

```bash
python quantize.py \
    --model_path openai/whisper-large-v2 \
    --quant_path output_dir/whisper-large-v2-FP8-Dynamic
```


```python
import argparse
import torch
import os
from datasets import load_dataset
from transformers import WhisperProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.tracing import TraceableWhisperForConditionalGeneration
from compressed_tensors.quantization import QuantizationType

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--quant_path', type=str, required=True)
parser.add_argument('--observer', type=str, default="minmax")
args = parser.parse_args()

# --- Load Model ---
model = TraceableWhisperForConditionalGeneration.from_pretrained(
    args.model_path,
    device_map="auto",
    torch_dtype="auto",
)
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(args.model_path)

# --- Recipe (FP8 Dynamic) ---
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        sequential_targets=["WhisperEncoderLayer", "WhisperDecoderLayer"],
        ignore=["re:.*lm_head"],
    )
]

# --- Run oneshot ---
oneshot(
    model=model,
    recipe=recipe,
    trust_remote_code_model=True,
)

# --- Save ---
os.makedirs(args.quant_path, exist_ok=True)
model.save_pretrained(args.quant_path, save_compressed=True)
processor.save_pretrained(args.quant_path)


```
</details>

## Evaluation

The model was evaluated on [LibriSpeech](https://huggingface.co/datasets/lmms-lab/librispeech) and [Fleurs](https://huggingface.co/datasets/lmms-lab/fleurs) datasets using [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval), via the following commands:

<details>
<summary>Evaluation Commands</summary>
  
Librispeech:
```
lmms-eval \
    --model=whisper_vllm \
    --model_args="pretrained=neuralmagic-ent/whisper-large-v2-FP8-Dynamic" \
    --batch_size 64 \
    --output_path <output_file_path> \
    --tasks librispeech
```

Fleurs:
```
lmms-eval \
    --model=whisper_vllm \
    --model_args="pretrained=neuralmagic-ent/whisper-large-v2-FP8-Dynamic" \
    --batch_size 64 \
    --output_path <output_file_path> \
    --tasks fleurs
```
</details>

<table>
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>Split</th>
      <th>BF16</th>
      <th>w8a8</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>LibriSpeech (WER)</b></td>
      <td>test-clean</td>
      <td>3.1437</td>
      <td>2.857</td>
      <td>110.04%</td>
    </tr>
    <tr>
      <td>test-other</td>
      <td>5.2362</td>
      <td>5.1813</td>
      <td>101.06%</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Fleurs (X→en, WER)</b></td>
      <td>cmn_hans_cn</td>
      <td>15.2148</td>
      <td>14.7614</td>
      <td>103.07%</td>
    </tr>
    <tr>
      <td>en</td>
      <td>4.0717</td>
      <td>4.0648</td>
      <td>100.17%</td>
    </tr>
    <tr>
      <td>yue_hant_hk</td>
      <td>8.5106</td>
      <td>8.4175</td>
      <td>1001.10%</td>
    </tr>
</table>
