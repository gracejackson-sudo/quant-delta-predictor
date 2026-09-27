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
- mistralai/Magistral-Small-2506
extra_gated_description: >-
  If you want to learn more about how we process your personal data, please read
  our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
---

# Model Card for Magistral-Small-2506-FP8

This checkpoint is the FP8 W8A8 quantized version of mistralai/Magistral-Small-2506, compressed using the Mistral-format integration in LLM Compressor.

It is important to note that this is a Mistral-format checkpoint, so it must be run in vLLM with `--tokenizer-mode mistral --config-format mistral --load-format mistral`.

For instance, serve the model as follows:

```
vllm serve RedHatAI/Magistral-Small-2506-FP8 --tokenizer-mode mistral --config-format mistral --load-format mistral --tool-call-parser mistral --enable-auto-tool-choice
```

## Evaluation

GSM8k:

```
lm_eval --model local-completions --model_args model=RedHatAI/Magistral-Small-2506-FP8,base_url=http://0.0.0.0:9000/v1/completions,num_concurrent=500,tokenized_requests=False --tasks gsm8k --num_fewshot 5
local-completions (model=RedHatAI/Magistral-Small-2506-FP8,base_url=http://0.0.0.0:9000/v1/completions,num_concurrent=500,tokenized_requests=False), gen_kwargs: (None), limit: None, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.8923|±  |0.0085|
|     |       |strict-match    |     5|exact_match|↑  |0.8886|±  |0.0087|
```

# Original Model Card

Building upon Mistral Small 3.1 (2503), **with added reasoning capabilities**, undergoing SFT from Magistral Medium traces and RL on top, it's a small, efficient reasoning model with 24B parameters.

Magistral Small can be deployed locally, fitting within a single RTX 4090 or a 32GB RAM MacBook once quantized.

Learn more about Magistral in our [blog post](https://mistral.ai/news/magistral/).

## Key Features
- **Reasoning:** Capable of long chains of reasoning traces before providing an answer.
- **Multilingual:** Supports dozens of languages, including English, French, German, Greek, Hindi, Indonesian, Italian, Japanese, Korean, Malay, Nepali, Polish, Portuguese, Romanian, Russian, Serbian, Spanish, Swedish, Turkish, Ukrainian, Vietnamese, Arabic, Bengali, Chinese, and Farsi.
- **Apache 2.0 License:** Open license allowing usage and modification for both commercial and non-commercial purposes.
- **Context Window:** A 128k context window, **but** performance might degrade past **40k**. Hence we recommend setting the maximum model length to 40k.

## Benchmark Results

| Model | AIME24 pass@1 | AIME25 pass@1 | GPQA Diamond | Livecodebench (v5) |
|-------|-------------|-------------|--------------|-------------------|
| Magistral Medium | 73.59% | 64.95% | 70.83% | 59.36% |
| Magistral Small | 70.68% | 62.76% | 68.18% | 55.84% |

## Sampling parameters

Please make sure to use: 
- `top_p`: 0.95
- `temperature`: 0.7
- `max_tokens`: 40960

## Basic Chat Template

We highly recommend including the default system prompt used during RL for the best results, you can edit and customise it if needed for your specific use case.

```
<s>[SYSTEM_PROMPT]system_prompt

A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.

Problem:

[/SYSTEM_PROMPT][INST]user_message[/INST]<think>
reasoning_traces
</think>
assistant_response</s>[INST]user_message[/INST]
```
*`system_prompt`, `user_message` and `assistant_response` are placeholders.*

We invite you to choose, depending on your use case and requirements, between keeping reasoning traces during multi-turn interactions or keeping only the final assistant response.

***Please make sure to use [mistral-common](https://github.com/mistralai/mistral-common) as the source of truth***

## Usage

The model can be used with the following frameworks;

### Inference

- [`vllm (recommended)`](https://github.com/vllm-project/vllm): See [below](#vllm)

In addition the community has prepared quantized versions of the model that can be used with the following frameworks (*alphabetically sorted*):
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp): https://huggingface.co/mistralai/Magistral-Small-2506_gguf
- [`lmstudio` (llama.cpp, MLX)](https://lmstudio.ai/): https://lmstudio.ai/models/mistralai/magistral-small
- [`ollama` (llama.cpp)](https://ollama.com/): https://ollama.com/library/magistral
- [`unsloth` (llama.cpp)](https://huggingface.co/unsloth): https://huggingface.co/unsloth/Magistral-Small-2506-GGUF

### Training

Fine-tuning is possible with (*alphabetically sorted*):
- [`axolotl`](https://github.com/axolotl-ai-cloud/axolotl): https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples/magistral
- [`unsloth`](https://github.com/unslothai/unsloth): https://docs.unsloth.ai/basics/magistral

### Other

Also you can use Magistral with:
- [`kaggle`](https://www.kaggle.com/models/mistral-ai/magistral-small-2506): https://www.kaggle.com/models/mistral-ai/magistral-small-2506

### vLLM (recommended)

We recommend using this model with the [vLLM library](https://github.com/vllm-project/vllm)
to implement production-ready inference pipelines.

**_Installation_**

Make sure you install the latest [`vLLM`](https://github.com/vllm-project/vllm/) code:

```
pip install -U vllm \
    --pre \
    --extra-index-url https://wheels.vllm.ai/nightly
```

Doing so should automatically install [`mistral_common >= 1.6.0`](https://github.com/mistralai/mistral-common/releases/tag/v1.6.0).

To check:
```
python -c "import mistral_common; print(mistral_common.__version__)"
```

You can also make use of a ready-to-go [docker image](https://github.com/vllm-project/vllm/blob/main/Dockerfile) or on the [docker hub](https://hub.docker.com/layers/vllm/vllm-openai/latest/images/sha256-de9032a92ffea7b5c007dad80b38fd44aac11eddc31c435f8e52f3b7404bbf39).


Serve model as follows:

```
vllm serve mistralai/Magistral-Small-2506 --tokenizer_mode mistral --config_format mistral --load_format mistral --tool-call-parser mistral --enable-auto-tool-choice --tensor-parallel-size 2
```

Ping model as follows:

```py
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.7
TOP_P = 0.95
MAX_TOK = 40_960

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

SYSTEM_PROMPT = load_system_prompt(model, "SYSTEM_PROMPT.txt")

query = "Write 4 sentences, each with at least 8 words. Now make absolutely sure that every sentence has exactly one word less than the previous sentence."
# or try out other queries
# query = "Exactly how many days ago did the French Revolution start? Today is June 4th, 2025."
# query = "Think about 5 random numbers. Verify if you can combine them with addition, multiplication, subtraction or division to 133"
# query = "If it takes 30 minutes to dry 12 T-shirts in the sun, how long does it take to dry 33 T-shirts?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": query}
]
stream = client.chat.completions.create(
  model=model,
  messages=messages,
  stream=True,
  temperature=TEMP,
  top_p=TOP_P,
  max_tokens=MAX_TOK,
)

print("client: Start streaming chat completions...")
printed_content = False

for chunk in stream:
  content = None
  # Check the content is content
  if hasattr(chunk.choices[0].delta, "content"):
    content = chunk.choices[0].delta.content

  if content is not None:
    if not printed_content:
        printed_content = True
        print("\ncontent:", end="", flush=True)
    # Extract and print the content
    print(content, end="", flush=True)

# content:<think>
# Alright, I need to write 4 sentences where each one has at least 8 words and each subsequent sentence has one fewer word than the previous one.
# ...
# Final boxed answer (the four sentences):

# \[
# \boxed{
# \begin{aligned}
# &\text{1. The quick brown fox jumps over lazy dog and yells hello.} \\
# &\text{2. I saw the cat on the stair with my hat.} \\
# &\text{3. The man in the moon came down quickly today.} \\
# &\text{4. A cat sat on the mat today patiently.}
# \end{aligned}
# }
# \]
```