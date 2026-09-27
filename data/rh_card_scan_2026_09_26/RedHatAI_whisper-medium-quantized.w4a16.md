---
tags:
- w4a16
- int4
- vllm
- audio
license: apache-2.0
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
language:
  - en
base_model: openai/whisper-medium
library_name: transformers
---

# whisper-medium-quantized.w4a16

## Model Overview
- **Model Architecture:** whisper-medium
  - **Input:** Audio-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
- **Release Date:** 04/16/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [openai/whisper-medium](https://huggingface.co/openai/whisper-medium).

### Model Optimizations

This model was obtained by quantizing the weights of [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) to INT4 data type, ready for inference with vLLM >= 0.5.2.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.audio import AudioAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="neuralmagic/whisper-medium-quantized.w4a16",
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
python quantize.py --model_path openai/whisper-medium --quant_path "output_dir/whisper-medium-quantized.w4a16" --calib_size 1024 --group_size 64 --dampening_frac 0.01 --actorder weight
```


```python
import torch
import argparse
from datasets import load_dataset
from transformers import WhisperProcessor
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers.tracing import TraceableWhisperForConditionalGeneration
import os
from compressed_tensors.quantization import QuantizationArgs, QuantizationType, QuantizationStrategy, ActivationOrdering, QuantizationScheme
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--quant_path', type=str)
parser.add_argument('--calib_size', type=int, default=256)
parser.add_argument('--dampening_frac', type=float, default=0.1) 
parser.add_argument('--observer', type=str, default="minmax")
parser.add_argument('--actorder', type=str, default="dynamic")
parser.add_argument('--group_size', type=int, default=128)
parser.add_argument('--save_dir', type=str, required=True)


args = parser.parse_args()
model_id = args.model_path

model = TraceableWhisperForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
model.config.forced_decoder_ids = None
processor = WhisperProcessor.from_pretrained(model_id)

# Configure processor the dataset task.
processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

# Select number of samples for calibration. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.

NUM_CALIBRATION_SAMPLES = args.calib_size
MAX_SEQUENCE_LENGTH = 2048
dampening_frac=args.dampening_frac
actorder_arg=args.actorder
group_size=args.group_size

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    trust_remote_code=True,
)

def preprocess(example):
    return {
        "array": example["audio"]["array"],
        "sampling_rate": example["audio"]["sampling_rate"],
        "text": " " + example["text"].capitalize(),
    }

ds = ds.map(preprocess, remove_columns=ds.column_names)

# Process inputs.
def process(sample):
    inputs = processor(
        audio=sample["array"],
        sampling_rate=sample["sampling_rate"],
        text=sample["text"],
        add_special_tokens=True,
        return_tensors="pt",
    )

    inputs["input_features"] = inputs["input_features"].to(dtype=model.dtype)
    inputs["decoder_input_ids"] = inputs["labels"]
    del inputs["labels"]

    return inputs

ds = ds.map(process, remove_columns=ds.column_names)

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

ignore=["lm_head"]

# Recipe
recipe = GPTQModifier(
    targets="Linear",
    config_groups={
        "config_group": QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type=QuantizationType.INT,
                strategy=QuantizationStrategy.GROUP,
                group_size=group_size,
                symmetric=True,
                dynamic=False,
                actorder=getattr(ActivationOrdering, actorder_arg.upper()),
            ),
        ),
    },
    sequential_targets=["WhisperEncoderLayer", "WhisperDecoderLayer"],
    ignore=["re:.*lm_head"],
    update_size=NUM_CALIBRATION_SAMPLES,
    dampening_frac=dampening_frac
)

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
)


# Save to disk compressed.
save_name = f"{model_id.split('/')[-1]}-quantized.w4a16"
save_path = os.path.join(args.save_dir, save_name)
print("Saving model:", save_path)
model.save_pretrained(save_path, save_compressed=True)
processor.save_pretrained(save_path)
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
    --model_args="pretrained=neuralmagic-ent/whisper-medium-quantized.w4a16" \
    --batch_size 64 \
    --output_path <output_file_path> \
    --tasks librispeech
```

Fleurs:
```
lmms-eval \
    --model=whisper_vllm \
    --model_args="pretrained=neuralmagic-ent/whisper-medium-quantized.w4a16" \
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
      <th>W4A16</th>
      <th>Recovery (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><b>LibriSpeech (WER)</b></td>
      <td>test-clean</td>
      <td>2.8269</td>
      <td>2.9419</td>
      <td>96.10%</td>
    </tr>
    <tr>
      <td>test-other</td>
      <td>6.4445</td>
      <td>6.5561</td>
      <td>98.30%</td>
    </tr>
    <tr>
      <td rowspan="3"><b>Fleurs (X→en, WER)</b></td>
      <td>cmn_hans_cn</td>
      <td>13.3371</td>
      <td>18.4749</td>
      <td>72.2%</td>
    </tr>
    <tr>
      <td>en</td>
      <td>4.6004</td>
      <td>4.7926</td>
      <td>95.99%</td>
    </tr>
    <tr>
      <td>yue_hant_hk</td>
      <td>9.8107</td>
      <td>13.4349</td>
      <td>73.02%</td>
    </tr>
  </tbody>
</table>
