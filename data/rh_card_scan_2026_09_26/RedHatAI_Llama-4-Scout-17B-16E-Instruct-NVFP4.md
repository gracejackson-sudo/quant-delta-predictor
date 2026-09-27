---
tags:
- fp4
- vllm
language:
- en
- de
- fr
- it
- pt
- hi
- es
- th
pipeline_tag: text-generation
license: llama3.1
base_model: meta-llama/Llama-4-Scout-17B-16E-Instruct
---

# Llama-4-Scout-17B-16E-Instruct-NVFP4

## Model Overview
- **Model Architecture:** Meta-Llama-3.1
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Intended Use Cases:** Intended for commercial and research use in multiple languages. 
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English.
- **Release Date:** 7/15/25
- **Version:** 1.0
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** RedHatAI

This model is a quantized version of [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct).
It was evaluated on a several tasks to assess the its quality in comparison to the unquatized model.

### Model Optimizations

This model was obtained by quantizing the weights and activations of [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) to FP4 data type, ready for inference with vLLM>=0.9.1
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.

Only the weights of the linear operators within transformers blocks are quantized using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Deployment

### Use with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.
<details>
<summary>Model Usage Code</summary>
  
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4"
number_gpus = 2

sampling_params = SamplingParams(temperature=0.6, top_p=0.9, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```
</details>

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

This model was created by applying [LLM Compressor with calibration samples from neuralmagic/calibration dataset](https://github.com/vllm-project/llm-compressor/blob/main/examples/multimodal_vision/llama4_example.py), as presented in the code snipet below.

<details>
<summary>Model Creation Code</summary>
  
```python
from transformers import Llama4ForConditionalGeneration, Llama4Processor
from transformers.quantizers.quantizers_utils import get_module_from_name
import torch
from datasets import load_dataset

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.utils.dev import skip_weights_initialize
from transformers.models.llama4.modeling_llama4 import Llama4TextMLP
from llmcompressor.modifiers.quantization import QuantizationModifier
import gc
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

def convert_model_for_quantization(model):
    to_delete = []
    for name, module in model.named_modules():
        module_class_name = module.__class__.__name__
        if module_class_name == "Llama4TextMoe":
            parent_module, module_name = get_module_from_name(model, name)
            parent_module._modules[module_name] = SequentialLlama4TextMoe(
                model.config.get_text_config(),
                module,
            )
            to_delete.append(module)
            print(f"Patched {name} with SequentialLlama4TextMoe", flush=True)

    for module in to_delete:
        del module
        gc.collect()
        torch.cuda.empty_cache()


class SequentialLlama4TextMoe(torch.nn.Module):
    def __init__(self, config, original_moe):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.experts = SequentialLlama4TextExperts(config, original_moe.experts)
        self.router = original_moe.router
        self.shared_expert = original_moe.shared_expert

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = self.router(hidden_states)

        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=1)

        router_scores = (
            torch.full_like(router_logits, float("-inf")).scatter_(1, router_indices, router_top_value).transpose(0, 1)
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden_states.dtype)

        out = self.shared_expert(hidden_states)
        for i in range(self.num_experts):
            out += self.experts[i](hidden_states) * router_scores[i].reshape(-1, 1)

        return out, router_scores


class SequentialLlama4TextExperts(torch.nn.ModuleList):
    def __init__(self, config, original_experts):
        self.num_experts = original_experts.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__([Llama4TextMLP(config) for _ in range(self.num_experts)])

        intermediate_size = original_experts.down_proj.shape[1]

        for i in range(self.num_experts):
            gate_up = original_experts.gate_up_proj[i]
            down = original_experts.down_proj[i]

            gate_proj = gate_up[:, :intermediate_size]
            up_proj = gate_up[:, intermediate_size:]

            self[i].gate_proj.weight.data = gate_proj.t().clone().contiguous()
            self[i].up_proj.weight.data = up_proj.t().clone().contiguous()
            self[i].down_proj.weight.data = down.t().clone().contiguous()

        original_experts.gate_up_proj = None
        original_experts.down_proj = None
        gc.collect()
        torch.cuda.empty_cache()


model_id = "meta-llama/Llama-4-Scout-17B-16E"

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16  # load on cpu
)
processor = Llama4Processor.from_pretrained(model_id)

convert_model_for_quantization(model)

# Oneshot arguments
DATASET_ID = "neuralmagic/calibration"
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 8192

ds = load_dataset(DATASET_ID, name="LLM", split=f"train[:{NUM_CALIBRATION_SAMPLES}]")

def preprocess_function(example):
    messgages = []
    for message in example["messages"]:
        messgages.append(
            {
                "role": message["role"], 
                "content": [{"type": "text", "text": message["content"]}]
            }
        )
    
    return processor.apply_chat_template(
        messgages, 
        return_tensors="pt", 
        padding=False, 
        truncation=True, 
        max_length=MAX_SEQUENCE_LENGTH,
        tokenize=True,
        add_special_tokens=False,
        return_dict=True,
        add_generation_prompt=False,
    ).to("cuda:0")

ds = ds.map(
    preprocess_function,
    batched=False,
    remove_columns=ds.column_names
)

# Define a oneshot data collator for multimodal inputs.
def data_collator(batch):
    assert len(batch) == 1
    return {
        key: torch.tensor(value) if key != "pixel_values" else torch.tensor(value, dtype=torch.bfloat16).squeeze(0)
        for key, value in batch[0].items()
    }

# Recipe
recipe = QuantizationModifier(targets="Linear", scheme="NVFP4", 
            ignore=[
                're:.*lm_head',
                're:.*self_attn',
                're:.*router',
                're:.*vision_model',
                're:.*multi_modal_projector',
                're:.*multi_modal_projector',
                "Llama4TextAttention",
            ],
            sequential_targets=["Llama4TextMLP"],
        )

SAVE_DIR = f"{model_id.split('/')[1]}-{recipe.scheme}"

# Perform oneshot
oneshot(
    model=model,
    tokenizer=model_id,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
    data_collator=data_collator,
    output_dir=SAVE_DIR
)

# Save to disk compressed.
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

```
</details>

## Evaluation

This model was evaluated on the well-known OpenLLM v1, OpenLLM v2, HumanEval, and HumanEval_64 benchmarks. All evaluations were conducted using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness).
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Metric</th>
      <th>Llama-4-Scout-17B-16E-Instruct</th>
      <th>Llama-4-Scout-17B-16E-Instruct-NVFP4 (this model)</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8"><b>OpenLLM V1</b></td>
      <td>mmlu_llama</td>
      <td>81.06</td>
      <td>79.11</td>
      <td>97.59</td>
    </tr>
    <tr>
      <td>mmlu_cot_llama (0-shot)</td>
      <td>85.86</td>
      <td>84.07</td>
      <td>97.92</td>
    </tr>
    <tr>
      <td>arc_challenge_llama (0-shot)</td>
      <td>93.39</td>
      <td>92.02</td>
      <td>98.53</td>
    </tr>
    <tr>
      <td>gsm8k_llama (8-shot, strict-match)</td>
      <td>93.78</td>
      <td>93.78</td>
      <td>100.00</td>
    </tr>
    <tr>
      <td>hellaswag (10-shot)</td>
      <td>79.06</td>
      <td>78.63</td>
      <td>99.46</td>
    </tr>
    <tr>
      <td>winogrande (5-shot)</td>
      <td>74.43</td>
      <td>73.48</td>
      <td>98.72</td>
    </tr>
    <tr>
      <td>truthfulQA (0-shot, mc2)</td>
      <td>62.15</td>
      <td>60.63</td>
      <td>97.55</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>81.39</b></td>
      <td><b>80.25</b></td>
      <td><b>98.59</b></td>
    </tr>
    <tr>
      <td rowspan="7"><b>OpenLLM V2</b></td>
      <td>MMLU-Pro (5-shot)</td>
      <td>55.68</td>
      <td>53.05</td>
      <td>95.28</td>
    </tr>
    <tr>
      <td>IFEval (0-shot)</td>
      <td>89.09</td>
      <td>89.57</td>
      <td>100.54</td>
    </tr>
    <tr>
      <td>BBH (3-shot)</td>
      <td>65.11</td>
      <td>63.53</td>
      <td>97.57</td>
    </tr>
    <tr>
      <td>Math-|v|-5 (4-shot)</td>
      <td>57.70</td>
      <td>55.06</td>
      <td>95.42</td>
    </tr>
    <tr>
      <td>GPQA (0-shot)</td>
      <td>30.70</td>
      <td>31.04</td>
      <td>101.11</td>
    </tr>
    <tr>
      <td>MuSR (0-shot)</td>
      <td>42.59</td>
      <td>43.52</td>
      <td>102.18</td>
    </tr>
    <tr>
      <td><b>Average</b></td>
      <td><b>57.04</b></td>
      <td><b>56.54</b></td>
      <td><b>99.13</b></td>
    </tr>
    <tr>
      <td rowspan="1"><b>Coding</b></td>
      <td>HumanEval_64 pass@2</td>
      <td>83.83</td>
      <td>84.81</td>
      <td>101.17</td>
    </tr>
  </tbody>
</table>



### Reproduction

The results were obtained using the following commands:

<details>
<summary>Model Evaluation Commands</summary>

#### MMLU_LLAMA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks mmlu_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### MMLU_COT_LLAMA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks mmlu_cot_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### ARC-Challenge
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks arc_challenge_llama \
  --apply_chat_template \
  --batch_size auto
```

#### GSM-8K
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks gsm8k_llama \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### Hellaswag
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks hellaswag \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### Winogrande
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks winogrande \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### TruthfulQA
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,add_bos_token=True,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True \
  --tasks truthfulqa \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --batch_size auto
```

#### OpenLLM v2
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks leaderboard \
  --batch_size auto
```

#### HumanEval and HumanEval_64
```
lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_instruct \
  --batch_size auto


lm_eval \
  --model vllm \
  --model_args pretrained="RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4",dtype=auto,max_model_len=4096,tensor_parallel_size=2,enable_chunked_prefill=True,enforce_eager=True\
  --apply_chat_template \
  --fewshot_as_multiturn \
  --tasks humaneval_64_instruct \
  --batch_size auto
```
</details>
