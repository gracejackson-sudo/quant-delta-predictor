---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model:
- mistralai/Ministral-3-14B-Instruct-2512-BF16
tags:
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP4
---

# Ministral-3-14B-Instruct-2512-NVFP4

## Model Overview
- **Model Architecture:** MistralForCausalLM
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP4
  - **Activation quantization:** FP4
- **Intended Use Cases:**
  - Reasoning.
  - Function calling.
  - Subject matter experts via fine-tuning.
  - Multilingual instruction following.
  - Translation.
- **Out-of-scope:** Use in any manner that violates applicable laws or regulations (including trade compliance laws).
- **Release Date:** 05/21/2025
- **Version:** 1.0
- **Model Developers:** RedHat (Neural Magic)

### Model Optimizations

This model was obtained by quantizing the weights and activations of [mistralai/Ministral-3-14B-Instruct-2512-BF16](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-BF16) to FP4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.
Only the weights and activations of the linear operators within transformers blocks of the language model are quantized. 


## Deployment

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend, as shown in the example below.

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/Ministral-3-14B-Instruct-2512-NVFP4"
number_gpus = 1
sampling_params = SamplingParams(temperature=0.15, top_p=1.0, top_k=20, min_p=0, max_tokens=65536)

messages = [
    {"role": "user", "content": prompt}
]

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompts, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

## Creation

<details>
  <summary>Creation details</summary>
  This model was created with [llm-compressor](https://github.com/vllm-project/llm-compressor) by running the code snippet below. 


  ```python
  from datasets import load_dataset
  from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend
  from llmcompressor import oneshot
  from llmcompressor.modifiers.quantization import QuantizationModifier
  from llmcompressor.utils import dispatch_for_generation
  
  MODEL_ID = "mistralai/Ministral-3-14B-Instruct-2512-BF16"
  
  model = Mistral3ForConditionalGeneration.from_pretrained(MODEL_ID, device_map="auto")
  tokenizer = MistralCommonBackend.from_pretrained(MODEL_ID)

  recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    weight_observer="mse",
    ignore= ['re:.*lm_head', 're:.*vision_tower.*', 're:.*multi_modal_projector.*', 're:.*self_attn'],
  )
  
  # Apply quantization.
  oneshot(model=model, recipe=recipe)
  
  # Confirm generations of the quantized model look sane.
  print("========== SAMPLE GENERATION ==============")
  dispatch_for_generation(model)
  input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
      model.device
  )
  output = model.generate(input_ids, max_new_tokens=20)
  print(tokenizer.decode(output[0]))
  print("==========================================")
  
  
  # Save to disk in compressed-tensors format.
  SAVE_DIR = MODEL_ID.split("/")[1] + "-NVFP4"
  model.save_pretrained(SAVE_DIR, save_compressed = True)
  tokenizer.save_pretrained(SAVE_DIR)
  ```
</details>
 


## Evaluation

The model was evaluated on the ifeval and mmmu using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), on reasoning tasks using [lighteval](https://github.com/neuralmagic/lighteval/tree/reasoning).
[vLLM](https://docs.vllm.ai/en/stable/) was used for all evaluations.

<details>
  <summary>Evaluation details</summary>

  **lm-evaluation-harness**
  ```
  lm_eval \
    --model vllm \
    --model_args pretrained="RedHatAI/Ministral-3-14B-Instruct-2512-NVFP4",dtype=auto,gpu_memory_utilization=0.7,max_model_len=262144,enable_chunk_prefill=True,tensor_parallel_size=1 \
    --tasks ifeval,mmmu_val \
    --apply_chat_template\
    --fewshot_as_multiturn \
    --batch_size auto
  ```

  **lighteval**
  
  litellm_config.yaml
  ```yaml
  model_parameters:
    provider: "hosted_vllm"
    model_name: "hosted_vllm/RedHatAI/Ministral-3-14B-Instruct-2512-NVFP4"
    base_url: "http://0.0.0.0:8000/v1"
    api_key: ""
    timeout: 1200
    concurrent_requests: 16
    generation_parameters:
      temperature: 0.15
      max_new_tokens: 65536
      top_p: 0.95
      seed: 0
  ```

  ```
  lighteval endpoint litellm litellm_config.yaml "aime25"
  ```

  ```
  lighteval endpoint litellm litellm_config.yaml "math_500"
  ```

  ```
  lighteval endpoint litellm litellm_config.yaml "gpqa:diamond"
  ```

</details>

### Accuracy

<table>
  <tr>
   <th>Category
   </th>
   <th>Benchmark
   </th>
   <th>Ministral-3-14B-Instruct-2512-BF16
   </th>
   <th>Ministral-3-14B-Instruct-2512-NVFP4<br>(this model)
   </th>
   <th>Recovery
   </th>
  </tr>
  <tr>
   <td rowspan="1" ><strong>Vision</strong>
   </td>    
   <td>MMMU
   </td>
   <td>55.33
   </td>
   <td>52.37
   </td>
   <td>94.65%
   </td>
  </tr>
  <tr>
   <td rowspan="1" ><strong>OpenLLM v2</strong>
   </td>
   <td>IFEval
   </td>
   <td>77.34
   </td>
   <td>63.55
   </td>
   <td>82.17%
   </td>
  </tr>
  <tr>
   <td rowspan="4" ><strong>Reasoning<br>(generation)</strong>
   </td>
   <td>AIME 2025
   </td>
   <td>36.67
   </td>
   <td>32.5
   </td>
   <td>88.63%
   </td>
  </tr>
  <tr>
   <td>GPQA diamond
   </td>
   <td>58.59
   </td>
   <td>60.94
   </td>
   <td>104.02%
   </td>
  </tr>
  <tr>
   <td>Math-lvl-5
   </td>
   <td>88.6
   </td>
   <td>85.80
   </td>
   <td>93.84%
   </td>
  </tr>
    <tr>
   <td><strong>Average</strong>
   </td>
   <td><strong>61.29</strong>
   </td>
   <td><strong>59.75</strong>
   </td>
   <td><strong>97.49%</strong>
   </td>
  </tr>
</table>
