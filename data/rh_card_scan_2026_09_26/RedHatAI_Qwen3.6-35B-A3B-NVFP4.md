---
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.6-35B-A3B
tags:
- fp4
- qwen
- nvfp4
- vllm
- llm-compressor
- compressed-tensors
name: RedHatAI/Qwen3.6-35B-A3B-NVFP4
provider: Alibaba Cloud
description: NVFP4 variant of Qwen 3.6 35B for reasoning and tool calling.
readme: https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4/blob/main/README.md
tool_calling_supported: true
required_cli_args: ['--reasoning-parser qwen3', '--enable-prefix-caching']
default-chat-template-kwargs: '{"enable_thinking": true}'
chat_template_file_name: None
chat_template_path: None
tool_call_parser: qwen3_coder
validated_tasks:
- tool-calling
tasks:
- text-to-text
- text-generation
- image-text-to-text
- tool-calling
license_name: apache-2.0
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/LICENSE
validated_on:
  - RHOAI 3.5
  - RHAIIS 3.5
  - vLLM 0.24.0
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Qwen3.6-35B-A3B-NVFP4
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>

# NVFP4 Quantized RedHatAI/Qwen3.6-35B-A3B-NVFP4

This is a preliminary version (and subject to change) of NVFP4 quantized [Qwen/Qwen3.6-35B-A3B ](https://huggingface.co/Qwen/Qwen3.6-35B-A3B ) model. 
The model has both weights and activations quantized to NVFP4 format with [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor).

It is compatible and tested against vllm main. Deploy it with: `vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 --reasoning-parser qwen3 --moe_backend flashinfer_cutlass`.

If you have hardware with more compute than memory bandwidth, you may prefer this MoE variant for performance reasons.

# Creation Script:

Run this script with LLM Compressor main and latest transformers.

<details>
  
```python
import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# NOTE: This example requires transformers >= v5

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"

# Load model.
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID, dtype="auto")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# No need to include mtp layers as they are not loaded
# through Qwen3_5MoeForConditionalGeneration
recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=[
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*mlp.gate$",
        "re:.*embed_tokens$",
        "re:.*shared_expert_gate$",
        "re:.*linear_attn.*",
    ],
)

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 4096

ds = load_dataset(
    "HuggingFaceH4/ultrachat_200k",
    split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]",
)
ds = ds.select_columns(["messages"])
ds = ds.shuffle(seed=42)


def preprocess_function(example):
    messages = [
        {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
        for m in example["messages"]
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        add_generation_prompt=False,
        processor_kwargs={
            "return_tensors": "pt",
            "padding": False,
            "truncation": True,
            "max_length": MAX_SEQUENCE_LENGTH,
            "add_special_tokens": False,
        },
    )


ds = ds.map(preprocess_function, batched=False, remove_columns=ds.column_names)


def data_collator(batch):
    assert len(batch) == 1
    return {key: torch.tensor(value) for key, value in batch[0].items()}


# Apply quantization.
oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    moe_calibrate_all_experts=True,
    data_collator=data_collator,
)

# Save to disk in compressed-tensors format.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

# MTP layers are excluded from the model through Qwen3_5MoeForConditionalGeneration
# Save them as-is from the original checkpoint into the quantized output.
save_mtp_tensors_to_checkpoint(source_model=MODEL_ID, dest_dir=SAVE_DIR)

```
</details>



## Evaluation

This model was evaluated on GSM8K-Platinum, MMLU-Pro, IFEval, Math 500, GPQA Diamond, AIME 25, and LiveCodeBench v6 using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [lighteval](https://github.com/huggingface/lighteval), served with [vLLM](https://github.com/vllm-project/vllm) using `--language-model-only`.

### Accuracy

| Benchmark | Qwen/Qwen3.6-35B-A3B | RedHatAI/Qwen3.6-35B-A3B-NVFP4 | Recovery (%) |
|-----------|-------------------|-------------------------|--------------|
| GSM8k Platinum (0-shot) | 95.73 | 96.08 | 100.37 |
| IfEval (0-shot) | 93.09 | 92.45 | 99.31 |
| AIME 2025 | 92.92 | 91.25 | 98.21 |
| GPQA diamond | 84.51 | 84.68 | 100.20 |
| Math 500 | 84.80 | 85.00 | 100.24 |
| Lcb Codegeneration V6 | 77.33 | 74.67 | 96.55 |
| MMLU Pro Chat | 85.32 | 84.70 | 99.28 |
| BFCLv4 Overall | 57.83 | 56.10 | 97.01 |
| BFCLv4 Single Turn | 53.81 | 53.45 | 99.34 |
| BFCLv4 Multi-Turn |62.25 | 58.13 |93.38  |
| BFCLv4 Agentic |49.91  | 49.31 | 98.80 |
| SWEBench Verified | 54.8 | 50.2 | 91.61 |


### Reproduction

The results were obtained using the following commands:

<details>

The model was served with vLLM using the following command:

```
vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 --reasoning-parser qwen3 --language-model-only --max-model-len 96000
```

Each benchmark was run 3 times with different seeds (42, 1234, 4158), except AIME 25 which used 8 seeds (42, 1234, 4158, 5322, 1356, 9843, 3344, 5678). Scores are averaged across all seeds.

#### lm-eval benchmarks

##### GSM8K-Platinum (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks gsm8k_platinum_cot_llama \
  --model_args "model=RedHatAI/Qwen3.6-35B-A3B-NVFP4,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=2400" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results.json \
  --seed 42 \
  --gen_kwargs "do_sample=true,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=42"
```

##### IFEval (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks ifeval \
  --model_args "model=RedHatAI/Qwen3.6-35B-A3B-NVFP4,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=2400" \
  --apply_chat_template \
  --output_path results.json \
  --seed 42 \
  --gen_kwargs "do_sample=true,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=42"
```

##### MMLU-Pro (0-shot)
```
lm_eval --model local-chat-completions \
  --tasks mmlu_pro_chat \
  --model_args "model=RedHatAI/Qwen3.6-35B-A3B-NVFP4,max_length=96000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=128,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=3600" \
  --num_fewshot 0 \
  --apply_chat_template \
  --output_path results.json \
  --seed 42 \
  --gen_kwargs "do_sample=true,temperature=1.0,top_p=0.95,top_k=20,min_p=0.0,max_gen_toks=64000,presence_penalty=1.5,repetition_penalty=1.0,seed=42"
```

#### lighteval benchmarks

**litellm_config.yaml:**
```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/RedHatAI/Qwen3.6-35B-A3B-NVFP4"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 2400
  concurrent_requests: 64
  generation_parameters:
    temperature: 1.0
    max_new_tokens: 64000
    top_p: 0.95
    top_k: 20
    min_p: 0.0
    presence_penalty: 1.5
    repetition_penalty: 1.0
    seed: 0
```

##### Math 500, GPQA Diamond, LiveCodeBench v6 (0-shot)
```
lighteval endpoint litellm litellm_config.yaml \
  "math_500|0,gpqa:diamond|0,lcb:codegeneration_v6|0" \
  --output-dir results \
  --save-details
```

##### AIME 25 (0-shot)
```
lighteval endpoint litellm litellm_config.yaml \
  "aime25|0" \
  --output-dir results \
  --save-details
```


#### BFCLv4

BFCL requires the model to be registered in the leaderboard codebase before running evaluation.

**Step 1 — Register the model in `bfcl_eval/constants/model_config.py`**

Add the following entry to `api_inference_model_map`:

```python
"Qwen3.6-35B-A3B-NVFP4": ModelConfig(
    model_name="Qwen3.6-35B-A3B-NVFP4",
    display_name="Qwen3.6-35B-A3B-NVFP4 (FC)",
    url="https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4",
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

Add `"Qwen3.6-35B-A3B-NVFP4"` to the `SUPPORTED_MODELS` list.

**Step 3 — Start the vLLM server** (use the command at the top of this section; the `--served-model-name` flag ensures BFCL can find the model by its registered slug).

**Step 4 — Generate responses and evaluate**
```
bfcl generate --model Qwen3.6-35B-A3B-NVFP4 --test-category all
bfcl evaluate --model Qwen3.6-35B-A3B-NVFP4 --test-category all
```


</details>

