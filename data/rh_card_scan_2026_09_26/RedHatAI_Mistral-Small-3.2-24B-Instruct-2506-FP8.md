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
license: apache-2.0
library_name: vllm
inference: false
base_model:
- mistralai/Mistral-Small-3.2-24B-Instruct-2506
extra_gated_description: >-
  If you want to learn more about how we process your personal data, please read
  our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
pipeline_tag: image-text-to-text
---

Deploy this FP8 quantized variant with vLLM using:

```
vllm serve RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-FP8 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10'
```

# Mistral-Small-3.2-24B-Instruct-2506

Mistral-Small-3.2-24B-Instruct-2506 is a minor update of [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503).

Small-3.2 improves in the following categories:
- **Instruction following**: Small-3.2 is better at following precise instructions
- **Repetition errors**: Small-3.2 produces less infinite generations or repetitive answers
- **Function calling**: Small-3.2's function calling template is more robust (see [here](https://github.com/mistralai/mistral-common/blob/535b4d0a0fc94674ea17db6cf8dc2079b81cbcfa/src/mistral_common/tokens/tokenizers/instruct.py#L778) and [examples](#function-calling))

In all other categories Small-3.2 should match or slightly improve compared to [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503).

## Key Features
- same as [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#key-features)

## Benchmark Results

We compare Mistral-Small-3.2-24B to [Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503).
For more comparison against other models of similar size, please check [Mistral-Small-3.1's Benchmarks'](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#benchmark-results)

### Text 

#### Instruction Following / Chat / Tone

| Model | Wildbench v2 | Arena Hard v2 | IF (Internal; accuracy) |
|-------|---------------|---------------|------------------------|
| Small 3.1 24B Instruct | 55.6% | 19.56% | 82.75% |
| **Small 3.2 24B Instruct** | **65.33%** | **43.1%** | **84.78%** |

#### Infinite Generations

Small 3.2 reduces infinite generations by 2x on challenging, long and repetitive prompts.

| Model | Infinite Generations (Internal; Lower is better) |
|-------|-------|
| Small 3.1 24B Instruct | 2.11% |
| **Small 3.2 24B Instruct** | **1.29%** |

#### STEM

| Model                          | MMLU      | MMLU Pro (5-shot CoT) | MATH                   | GPQA Main (5-shot CoT) | GPQA Diamond (5-shot CoT )| MBPP Plus - Pass@5 | HumanEval Plus - Pass@5 | SimpleQA (TotalAcc)|
|--------------------------------|-----------|-----------------------|------------------------|------------------------|---------------------------|--------------------|-------------------------|--------------------|
| Small 3.1 24B Instruct         | 80.62%    | 66.76%                | 69.30%                 | 44.42%                 | 45.96%                    | 74.63%             | 88.99%                  | 10.43%             |
| **Small 3.2 24B Instruct**     | 80.50%    | **69.06%**            | 69.42%                 | 44.22%                 | 46.13%                    | **78.33%**         | **92.90%**              | **12.10%**         |

### Vision

| Model                          | MMMU       | Mathvista | ChartQA   | DocVQA    | AI2D      |
|--------------------------------|------------|-----------|-----------|-----------|-----------|
| Small 3.1 24B Instruct         | **64.00%** | **68.91%**| 86.24%    | 94.08%    | 93.72%  | 
| **Small 3.2 24B Instruct**     | 62.50%     | 67.09%    | **87.4%** | 94.86%    | 92.91%  | 


## Usage

The model can be used with the following frameworks;
- [`vllm (recommended)`](https://github.com/vllm-project/vllm): See [here](#vllm-recommended)
- [`transformers`](https://github.com/huggingface/transformers): See [here](#transformers)

**Note 1**: We recommend using a relatively low temperature, such as `temperature=0.15`.

**Note 2**: Make sure to add a system prompt to the model to best tailor it to your needs. If you want to use the model as a general assistant, we recommend to use the one provided in the [SYSTEM_PROMPT.txt](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506/blob/main/SYSTEM_PROMPT.txt) file.

### vLLM (recommended)

We recommend using this model with [vLLM](https://github.com/vllm-project/vllm).

#### Installation

Make sure to install [`vLLM >= 0.9.1`](https://github.com/vllm-project/vllm/releases/tag/v0.9.1):

```
pip install vllm --upgrade
```

Doing so should automatically install [`mistral_common >= 1.6.2`](https://github.com/mistralai/mistral-common/releases/tag/v1.6.2).

To check:
```
python -c "import mistral_common; print(mistral_common.__version__)"
```

You can also make use of a ready-to-go [docker image](https://github.com/vllm-project/vllm/blob/main/Dockerfile) or on the [docker hub](https://hub.docker.com/layers/vllm/vllm-openai/latest/images/sha256-de9032a92ffea7b5c007dad80b38fd44aac11eddc31c435f8e52f3b7404bbf39).

#### Serve

We recommend that you use Mistral-Small-3.2-24B-Instruct-2506 in a server/client setting. 

1. Spin up a server:

```
vllm serve mistralai/Mistral-Small-3.2-24B-Instruct-2506 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --limit_mm_per_prompt 'image=10' --tensor-parallel-size 2
```

**Note:** Running Mistral-Small-3.2-24B-Instruct-2506 on GPU requires ~55 GB of GPU RAM in bf16 or fp16. 


2. To ping the client you can use a simple Python snippet. See the following examples.


#### Vision reasoning

Leverage the vision capabilities of Mistral-Small-3.2-24B-Instruct-2506 to make the best choice given a scenario, go catch them all !

<details>
  <summary>Python snippet</summary>

```py
from datetime import datetime, timedelta

from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 131072

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


model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
SYSTEM_PROMPT = load_system_prompt(model_id, "SYSTEM_PROMPT.txt")
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
    max_tokens=MAX_TOK,
)

print(response.choices[0].message.content)
# In this situation, you are playing a Pokémon game where your Pikachu (Level 42) is facing a wild Pidgey (Level 17). Here are the possible actions you can take and an analysis of each:

# 1. **FIGHT**:
#    - **Pros**: Pikachu is significantly higher level than the wild Pidgey, which suggests that it should be able to defeat Pidgey easily. This could be a good opportunity to gain experience points and possibly items or money.
#    - **Cons**: There is always a small risk of Pikachu fainting, especially if Pidgey has a powerful move or a status effect that could hinder Pikachu. However, given the large level difference, this risk is minimal.

# 2. **BAG**:
#    - **Pros**: You might have items in your bag that could help in this battle, such as Potions, Poké Balls, or Berries. Using an item could help you capture the Pidgey or heal your Pikachu if needed.
#    - **Cons**: Using items might not be necessary given the level difference. It could be more efficient to just fight and defeat the Pidgey quickly.

# 3. **POKÉMON**:
#    - **Pros**: You might have another Pokémon in your party that is better suited for this battle or that you want to gain experience. Switching Pokémon could also be a strategic move if you want to train a lower-level Pokémon.
#    - **Cons**: Switching Pokémon might not be necessary since Pikachu is at a significant advantage. It could also waste time and potentially give Pidgey a turn to attack.

# 4. **RUN**:
#    - **Pros**: Running away could save time and conserve your Pokémon's health and resources. If you are in a hurry or do not need the experience or items, running away is a safe option.
#    - **Cons**: Running away means you miss out on the experience points and potential items or money that you could gain from defeating the Pidgey. It also means you do not get the chance to capture the Pidgey if you wanted to.

# ### Recommendation:
# Given the significant level advantage, the best action is likely to **FIGHT**. This will allow you to quickly defeat the Pidgey, gain experience points, and potentially earn items or money. If you are concerned about Pikachu's health, you could use an item from your **BAG** to heal it before or during the battle. Running away or switching Pokémon does not seem necessary in this situation.
```
</details>

#### Function calling

Mistral-Small-3.2-24B-Instruct-2506 is excellent at function / tool calling tasks via vLLM. *E.g.:*

<details>
  <summary>Python snippet - easy</summary>

```py
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 131072

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
    return system_prompt

model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
SYSTEM_PROMPT = load_system_prompt(model_id, "SYSTEM_PROMPT.txt")

image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_population",
            "description": "Get the up-to-date population of a given country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "The country to find the population of.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "The unit for the population.",
                        "enum": ["millions", "thousands"],
                    },
                },
                "required": ["country", "unit"],
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
        "content": "Could you please make the below article more concise?\n\nOpenAI is an artificial intelligence research laboratory consisting of the non-profit OpenAI Incorporated and its for-profit subsidiary corporation OpenAI Limited Partnership.",
    },
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "bbc5b7ede",
                "type": "function",
                "function": {
                    "name": "rewrite",
                    "arguments": '{"text": "OpenAI is an artificial intelligence research laboratory consisting of the non-profit OpenAI Incorporated and its for-profit subsidiary corporation OpenAI Limited Partnership."}',
                },
            }
        ],
    },
    {
        "role": "tool",
        "content": '{"action":"rewrite","outcome":"OpenAI is a FOR-profit company."}',
        "tool_call_id": "bbc5b7ede",
        "name": "rewrite",
    },
    {
        "role": "assistant",
        "content": "---\n\nOpenAI is a FOR-profit company.",
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Can you tell me what is the biggest country depicted on the map?",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            },
        ],
    }
]

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    max_tokens=MAX_TOK,
    tools=tools,
    tool_choice="auto",
)

assistant_message = response.choices[0].message.content
print(assistant_message)
# The biggest country depicted on the map is Russia.

messages.extend([
    {"role": "assistant", "content": assistant_message},
    {"role": "user", "content": "What is the population of that country in millions?"},
])

response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=TEMP,
    max_tokens=MAX_TOK,
    tools=tools,
    tool_choice="auto",
)

print(response.choices[0].message.tool_calls)
# [ChatCompletionMessageToolCall(id='3e92V6Vfo', function=Function(arguments='{"country": "Russia", "unit": "millions"}', name='get_current_population'), type='function')]
```

</details>

<details>
  <summary>Python snippet - complex</summary>

```python
import json
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 131072

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
    return system_prompt


model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
SYSTEM_PROMPT = load_system_prompt(model_id, "SYSTEM_PROMPT.txt")

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
                "text": "Can you calculate the results for all the equations displayed in the image? Only compute the ones that involve numbers.",
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
    max_tokens=MAX_TOK,
    tools=tools,
    tool_choice="auto",
)

tool_calls = response.choices[0].message.tool_calls
print(tool_calls)
# [ChatCompletionMessageToolCall(id='CyQBSAtGh', function=Function(arguments='{"expression": "6 + 2 * 3"}', name='my_calculator'), type='function'), ChatCompletionMessageToolCall(id='KQqRCqvzc', function=Function(arguments='{"expression": "19 - (8 + 2) + 1"}', name='my_calculator'), type='function')]

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
    max_tokens=MAX_TOK,
)

print(response.choices[0].message.content)
# Here are the results for the equations that involve numbers:

# 1. \( 6 + 2 \times 3 = 12 \)
# 3. \( 19 - (8 + 2) + 1 = 10 \)

# For the other equations, you need to substitute the variables with specific values to compute the results.
```

</details>

#### Instruction following

Mistral-Small-3.2-24B-Instruct-2506 will follow your instructions down to the last letter ! 

<details>
  <summary>Python snippet</summary>

```python
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 131072

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
    return system_prompt


model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
SYSTEM_PROMPT = load_system_prompt(model_id, "SYSTEM_PROMPT.txt")

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
    max_tokens=MAX_TOK,
)

assistant_message = response.choices[0].message.content
print(assistant_message)

# Here's a sentence where each word starts with the next letter of the alphabet, starting from 'a' and ending with 'z':

# "Always brave cats dance elegantly, fluffy giraffes happily ignore jungle kites, lovingly munching nuts, observing playful quails racing swiftly, tiny unicorns vaulting while xylophones yodel zealously."

# This sentence follows the sequence from A to Z without skipping any letters.
```
</details>

### Transformers

You can also use Mistral-Small-3.2-24B-Instruct-2506 with `Transformers` !

To make the best use of our model with `Transformers` make sure to have [installed](https://github.com/mistralai/mistral-common) `mistral-common >= 1.6.2` to use our tokenizer.

```bash
pip install mistral-common --upgrade
```

Then load our tokenizer along with the model and generate:

<details>
  <summary>Python snippet</summary>

```python
from datetime import datetime, timedelta
import torch

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download
from transformers import Mistral3ForConditionalGeneration


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
SYSTEM_PROMPT = load_system_prompt(model_id, "SYSTEM_PROMPT.txt")

tokenizer = MistralTokenizer.from_hf_hub(model_id)

model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)

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

tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))

input_ids = torch.tensor([tokenized.tokens])
attention_mask = torch.ones_like(input_ids)
pixel_values = torch.tensor(tokenized.images[0], dtype=torch.bfloat16).unsqueeze(0)
image_sizes = torch.tensor([pixel_values.shape[-2:]])

output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_sizes=image_sizes,
    max_new_tokens=1000,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized.tokens) :])
print(decoded_output)
# In this situation, you are playing a Pokémon game where your Pikachu (Level 42) is facing a wild Pidgey (Level 17). Here are the possible actions you can take and an analysis of each:

# 1. **FIGHT**:
#    - **Pros**: Pikachu is significantly higher level than the wild Pidgey, which suggests that it should be able to defeat Pidgey easily. This could be a good opportunity to gain experience points and possibly items or money.
#    - **Cons**: There is always a small risk of Pikachu fainting, especially if Pidgey has a powerful move or a status effect that could hinder Pikachu. However, given the large level difference, this risk is minimal.

# 2. **BAG**:
#    - **Pros**: You might have items in your bag that could help in this battle, such as Potions, Poké Balls, or Berries. Using an item could help you capture Pidgey or heal Pikachu if needed.
#    - **Cons**: Using items might not be necessary given the level difference. It could be more efficient to just fight and defeat Pidgey quickly.

# 3. **POKÉMON**:
#    - **Pros**: You might have another Pokémon in your party that is better suited for this battle or that you want to gain experience. Switching Pokémon could also be strategic if you want to train a lower-level Pokémon.
#    - **Cons**: Switching Pokémon might not be necessary since Pikachu is at a significant advantage. It could also waste time and potentially give Pidgey a turn to attack.

# 4. **RUN**:
#    - **Pros**: Running away could be a quick way to avoid the battle altogether. This might be useful if you are trying to conserve resources or if you are in a hurry to get to another location.
#    - **Cons**: Running away means you miss out on the experience points, items, or money that you could gain from defeating Pidgey. It also might not be the most efficient use of your time if you are trying to train your Pokémon.

# ### Recommendation:
# Given the significant level advantage, the best action to take is likely **FIGHT**. This will allow you to quickly defeat Pidgey and gain experience points for Pikachu. If you are concerned about Pikachu's health, you could use the **BAG** to heal Pikachu before or during the battle. Running away or switching Pokémon does not seem necessary in this situation.
```

</details>