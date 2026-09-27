---
language:
- en
base_model:
- deepreinforce-ai/Ornith-1.0-35B
pipeline_tag: text-generation
tags:
- qwen3_5
- moe
- nvfp4
- fp4
- vllm
- conversational
- text-generation-inference
license: mit
---

## Model Overview
- **Model Architecture:** Qwen3_5MoeForConditionalGeneration
  - **Input:** Text (+ image/video inputs supported by the base architecture)
  - **Output:** Text
- **Model Optimizations:**
  - **Activation quantization:** FP4
  - **Weight quantization:** FP4
- **Intended Use Cases:** Intended for commercial and research use. Similarly to the base model, this quantized version is intended for agentic coding and general assistant-like chat.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights and activations of [deepreinforce-ai/Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B) to NVFP4 data type.
This optimization reduces the number of bits used to represent weights from 16 to 4, significantly reducing GPU memory requirements (by approximately 75%) and increasing inference throughput.

Only weights and activations of the linear operators within the language-model transformer blocks are quantized.
The router/gate projections, shared-expert gates, token embeddings, vision tower, and the entire linear-attention (gated deltanet) sub-module are kept at full precision, since they are either not compute-bound `Linear` layers or are highly sensitive to quantization error.
Quantization is performed using the NVFP4 scheme (`QuantizationModifier`), which targets NVIDIA H100 and Blackwell (sm90+) GPUs, with calibration performed over 256 samples from `HuggingFaceH4/ultrachat_200k` and `moe_calibrate_all_experts=True` so that every one of the 256 experts per layer receives calibration signal, not just the experts that are routed to for the sampled tokens.
The [llm-compressor](https://github.com/vllm-project/llm-compressor) library is used for quantization.

**Hardware requirement:** H100 / Blackwell (sm90+) for inference.

## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Ornith-1.0-35B-NVFP4"
number_gpus = 1
sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, min_p=0, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)
outputs = llm.generate(prompts, sampling_params)
generated_text = outputs[0].outputs[0].text
print(generated_text)
```

## Creation

<details>
  <summary>Creation details</summary>

  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. This is the same calibration-based `oneshot` recipe used upstream for [Qwen/Qwen3.5-122B-A10B](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/qwen3_5_example.py) (the reference example for the `Qwen3_5MoeForConditionalGeneration` architecture that Ornith-1.0-35B shares), pointed at `deepreinforce-ai/Ornith-1.0-35B`. Note: unlike `Qwen/Qwen3.5-*`, the `deepreinforce-ai/Ornith-1.0-35B` checkpoint does not ship separate MTP (multi-token-prediction) tensors, so the `save_mtp_tensors_to_checkpoint` step used upstream is omitted here.

  ```python
  import torch
  from datasets import load_dataset
  from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

  from llmcompressor import oneshot
  from llmcompressor.modifiers.quantization import QuantizationModifier
  from llmcompressor.utils import load_context

  # NOTE: This example requires transformers >= v5

  MODEL_ID = "deepreinforce-ai/Ornith-1.0-35B"

  # Load model.
  with load_context(Qwen3_5MoeForConditionalGeneration):
      model = Qwen3_5MoeForConditionalGeneration.from_pretrained(MODEL_ID)
  processor = AutoProcessor.from_pretrained(MODEL_ID)

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
          return_tensors="pt",
          padding=False,
          truncation=True,
          max_length=MAX_SEQUENCE_LENGTH,
          tokenize=True,
          add_special_tokens=False,
          return_dict=True,
          add_generation_prompt=False,
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
  ```
</details>
