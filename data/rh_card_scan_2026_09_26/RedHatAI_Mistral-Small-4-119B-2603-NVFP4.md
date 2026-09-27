---
language:
- en
- fr
- de
- es
- pt
- it
- ja
- ko
- ru
- zh
- ar
- fa
- id
- ms
- ne
- pl
- ro
- sr
- sv
- tr
- uk
- vi
- hi
- bn
tags:
- vLLM
- compressed-tensors
- NVFP4
base_model:
- mistralai/Mistral-Small-4-119B-2603
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Mistral-Small-4-119B-2603-NVFP4
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md
readme: https://huggingface.co/RedHatAI/Mistral-Small-4-119B-2603-NVFP4/blob/main/README.md
description: Mistral Small 4 (NVFP4 variant) is a powerful hybrid model capable of acting as both a general instruction model and a reasoning model.
provider: Mistral AI
validated_on:
  - RHOAI 3.4 
  - RHAIIS 3.4
  - vLLM 0.18.0
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Mistral Small 4 119B A6B NVFP4
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

Mistral Small 4 is a powerful hybrid model capable of acting as both a general instruction model and a reasoning model. It unifies the capabilities of three different model families—**Instruct**, **Reasoning** (previously called Magistral), and **Devstral**—into a single, unified model.

With its multimodal capabilities, efficient architecture, and flexible mode switching, it is a powerful general-purpose model for any task. In a latency-optimized setup, Mistral Small 4 achieves a **40% reduction in end-to-end completion time**, and in a throughput-optimized setup, it handles **3x more requests per second** compared to Mistral Small 3.

> [!Tip]
> This checkpoint in particular is a post-training-activation quantized version of [mistralai/Mistral-Small-4-119B-2603](https://huggingface.co/mistralai/Mistral-Small-4-119B-2603).
> It was created using [llm-compressor](https://github.com/vllm-project/llm-compressor) as part of a collaboration with teams from vLLM & Red Hat.
> Special thanks goes out to Dipika Sikka.
> We also would like to thank NVIDIA for their expertise and contributions to SGLang as well as vLLM to ensure kernel optimizations.

## Key Features

Mistral Small 4 includes the following architectural choices:

- **MoE**: 128 experts, 4 active.
- **119B parameters**, with **6.5B activated per token**.
- **256k context length**.
- **Multimodal input**: Accepts both text and image input, with text output.
- **Instruct and Reasoning functionalities** with function calls (reasoning effort configurable per request).

**ModelCar Storage URI:** oci://registry.redhat.io/rhai/modelcar-mistral-small-4-119b-2603-nvfp4:3.0

**Validated on vLLM:** 0.18.0

**Validated on RHAIIS:** 3.4

**Validated on RHOAI:** 3.4


Mistral Small 4 offers the following capabilities:

- **Reasoning Mode**: Toggle between fast instant reply mode and reasoning mode, boosting performance with test-time compute when requested.
- **Vision**: Analyzes images and provides insights based on visual content, in addition to text.
- **Multilingual**: Supports dozens of languages, including English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, and Arabic.
- **System Prompt**: Strong adherence and support for system prompts.
- **Agentic**: Best-in-class agentic capabilities with native function calling and JSON output.
- **Speed-Optimized**: Delivers best-in-class performance and speed.
- **Apache 2.0 License**: Open-source license for both commercial and non-commercial use.
- **Large Context Window**: Supports a 256k context window.

## Recommended Settings

- **Reasoning Effort**:
  - `'none'` → Do not use reasoning
  - `'high'` → Use reasoning (recommended for complex prompts)
  Use `reasoning_effort="high"` for complex tasks
- **Temperature**: 0.7 for `reasoning_effort="high"`. Temp between 0.0 and 0.7 for `reasoning_effort="none"` depending on task.

## Use Cases

Mistral Small 4 is designed for general chat assistants, coding, agentic tasks, and reasoning tasks (with reasoning mode toggled). Its multimodal capabilities also enable document and image understanding for data extraction and analysis.

Its capabilities are ideal for:
- Developers interested in coding and agentic capabilities for SWE automation and codebase exploration.
- Enterprises seeking general chat assistants, agents, and document understanding.
- Researchers leveraging its math and research capabilities.

Mistral Small 4 is also well-suited for customization and fine-tuning for more specialized tasks.

### Examples
- General chat assistant
- Document parsing and extraction
- Coding agent
- Research assistant
- Customization & fine-tuning
- And more...

## Usage

You can find Mistral Small 4 support on multiple libraries for inference:
- [`vllm (recommended)`](https://github.com/vllm-project/vllm): See [here](#vllm-recommended).
- [`SGLang`](https://github.com/sgl-project/sglang): (WIP ⏳ – follow updates [here](https://github.com/sgl-project/sglang/pull/20708/))


## vLLM (Recommended)

We recommend using Mistral Small 4 with the [vLLM library](https://github.com/vllm-project/vllm) for production-ready inference.

### Installation

> [!Tip]
> Use our custom Docker image with fixes for tool calling and reasoning parsing in vLLM, and the latest Transformers version. We are working with the vLLM team to merge these fixes soon.

**Custom Docker**
Use the following Docker image: [`mistralllm/vllm-ms4:latest`](https://hub.docker.com/repository/docker/mistralllm/vllm-ms4/latest/):
```bash
docker pull mistralllm/vllm-ms4:latest
docker run -it mistralllm/vllm-ms4:latest
```

**Manual Install**
Alternatively, install `vllm` from this PR: [Add Mistral Guidance](https://github.com/vllm-project/vllm/pull/37081).

> **Note**: This PR is expected to be merged into `vllm` main in the next 1-2 weeks (as of 16.03.2026). Track updates [here](https://github.com/vllm-project/vllm/pull/37081).

1. Clone vLLM:
   ```bash
   git clone --branch fix_mistral_parsing https://github.com/juliendenize/vllm.git
   ```
2. Install with pre-compiled kernels:
   ```bash
   VLLM_USE_PRECOMPILED=1 pip install --editable .
   ```
3. Install `transformers` from main:
   ```bash
   uv pip install git+https://github.com/huggingface/transformers.git
   ```
   Ensure [`mistral_common >= 1.10.0`](https://github.com/mistralai/mistral-common/releases/tag/v1.10.0) is installed:
   ```bash
   python -c "import mistral_common; print(mistral_common.__version__)"
   ```

### Serve the Model

We recommend a server/client setup:
```bash
vllm serve mistralai/Mistral-Small-4-119B-2603-NVFP4 --max-model-len 262144 --tensor-parallel-size 2 --attention-backend TRITON_MLA \
  --tool-call-parser mistral --enable-auto-tool-choice --reasoning-parser mistral --max_num_batched_tokens 16384 --max_num_seqs 128 \
  --gpu_memory_utilization 0.8
```

### Ping the Server

<details>
  <summary>Instruction Following</summary>

Mistral Small 4 can follow your instructions to the letter.


```python
from datetime import datetime, timedelta

from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.1

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "Write me a sentence where every word starts with the next letter in the alphabet - start with 'a' and end with 'z'.",
    },
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    reasoning_effort="none",
)

assistant_message = response.choices[0].message.content
print(assistant_message)
```

</details>

<details>
  <summary>Tool Call</summary>

Let's solve some equations thanks to our simple Python calculator tool.


```python
import json
from datetime import datetime, timedelta

from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.1

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")

image_url = "https://math-coaching.com/img/fiche/46/expressions-mathematiques.jpg"


def my_calculator(expression: str) -> str:
    return str(eval(expression))


tools = [
    {
        "type": "function",
        "function": {
            "name": "my_calculator",
            "description": "A calculator that can evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite",
            "description": "Rewrite a given text for improved clarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The input text to rewrite",
                    }
                },
            },
        },
    },
]

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Thanks to your calculator, compute the results for the equations that involve numbers displayed in the image.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        ],
    },
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    tools=tools,
    tool_choice="auto",
    reasoning_effort="none",
)

tool_calls = response.choices[0].message.tool_calls

results = []
for tool_call in tool_calls:
    function_name = tool_call.function.name
    function_args = tool_call.function.arguments
    if function_name == "my_calculator":
        result = my_calculator(**json.loads(function_args))
        results.append(result)

messages.append({"role": "assistant", "tool_calls": tool_calls})
for tool_call, result in zip(tool_calls, results):
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": result,
        }
    )


response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    reasoning_effort="none",
)

print(response.choices[0].message.content)
```

</details>

<details>
  <summary>Vision Reasoning</summary>

Let's see if the Mistral Small 4 knows when to pick a fight !

```python
from datetime import datetime, timedelta

from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.7

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")
image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]


response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    reasoning_effort="high",
)

print(response.choices[0].message.content)
```

</details>

## License

This model is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).

*You must not use this model in a manner that infringes, misappropriates, or violates any third party’s rights, including intellectual property rights.*