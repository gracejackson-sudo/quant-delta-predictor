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
base_model: openai/whisper-large-v2
library_name: transformers
---

# whisper-large-v2-quantized.w4a16

## Model Overview
- **Model Architecture:** whisper-large-v2
  - **Input:** Audio-Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4
  - **Activation quantization:** FP16
- **Release Date:** 1/31/2025
- **Version:** 1.0
- **Model Developers:** Neural Magic

Quantized version of [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2).

### Model Optimizations

This model was obtained by quantizing the weights of [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) to INT4 data type, ready for inference with vLLM >= 0.5.2.

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm.assets.audio import AudioAsset
from vllm import LLM, SamplingParams

# prepare model
llm = LLM(
    model="neuralmagic/whisper-large-v2-W4A16-G128",
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

This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below as part a multimodal announcement blog.

```python
import torch
from datasets import load_dataset
from transformers import WhisperProcessor

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.transformers import oneshot
from llmcompressor.transformers.tracing import TraceableWhisperForConditionalGeneration

# Select model and load it.
model_id = "openai/whisper-large-v2"
model = TraceableWhisperForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
)
processor = WhisperProcessor.from_pretrained(model_id)

# Configure processor the dataset task.
processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

# Select calibration dataset.
DATASET_ID = "MLCommons/peoples_speech"
DATASET_SUBSET = "test"
DATASET_SPLIT = "test"

# Select number of samples. 512 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(
    DATASET_ID,
    DATASET_SUBSET,
    split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    trust_remote_code=True,
)

# Preprocess and Tokenize inputs.
def preprocess_and_tokenize(example):
    audio = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]
    text = " " + example["text"].capitalize()

    audio_inputs = processor(
        audio=audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )

    text_inputs = processor(
        text=text,
        add_special_tokens=True,
        return_tensors="pt"
    )
    text_inputs["decoder_input_ids"] = text_inputs["input_ids"]
    del text_inputs["input_ids"]

    return dict(**audio_inputs, **text_inputs)

ds = ds.map(preprocess_and_tokenize, remove_columns=ds.column_names)

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}

# Recipe
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply algorithms.
SAVE_DIR = model_id.split("/")[1] + "-W4A16-G128"

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    data_collator=data_collator,
    output_dir=SAVE_DIR,
)

```


### BibTeX entry and citation info
```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}