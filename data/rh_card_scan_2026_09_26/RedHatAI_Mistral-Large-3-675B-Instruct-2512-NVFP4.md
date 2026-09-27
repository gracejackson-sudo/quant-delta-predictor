---
library_name: vllm
language:
- en
- fr
- es
- de
- it
- pt
- nl
- zh
- ja
- ko
- ar
license: apache-2.0
license_name: apache-2.0
name: RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4
description: State-of-the-art general-purpose Multimodal granular Mixture-of-Experts model, fine-tuned for instruction tasks, making it ideal for chat, agentic and instruction based use cases. NVFP4 format, optimized for Nvidia Blackwell. 
readme: https://huggingface.co/RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4/main/README.md
tasks:
- text-to-text
- text-generation
- image-to-text
- tool-calling
inference: false
provider: MistralAI
license_link: https://www.apache.org/licenses/LICENSE-2.0
validated_on:
  - RHOAI 3.0
  - RHAIIS 3.2.5
  - vLLM 0.11.2
extra_gated_description: >-
  If you want to learn more about how we process your personal data, please read
  our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
base_model:
- mistralai/Mistral-Large-3-675B-Base-2512
tags:
- mistral-common
- compressed-tensors
---

<h1 align: center; style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Mistral Large 3 675B Instruct 2512-NVFP4
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>

From our family of large models, **Mistral Large 3** is a state-of-the-art general-purpose **Multimodal granular Mixture-of-Experts** model with **41B active parameters** and **675B total parameters** trained from the ground up with 3000 H200s.

This model is the instruct post-trained version, fine-tuned for instruction tasks, making it ideal for chat, agentic and instruction based use cases.  
Designed for reliability and long-context comprehension - It is engineered for production-grade assistants, retrieval-augmented systems, scientific workloads, and complex enterprise workflows.

> [!Tip]
> This checkpoint in particular is a post-training-activation quantized version of [Mistral-Large-3-675B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512).
> It was created using [llm-compressor](https://github.com/vllm-project/llm-compressor) as part of a collaboration with teams from vLLM & Red Hat.
> Special thanks goes out to Dipika Sikka, Kyle Sayers, Eldar Kurtić, and Tyler Michael Smith.

Mistral Large 3 is deployable on-premises in:
- [FP8](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512) on a single node of B200s or H200s.
- **NVFP4** on a single node of H100s or A100s.

- **ModelCar Storage URI:** oci://registry.redhat.io/rhai/modelcar-mistral-large-3-675b-instruct-2512-nvfp4:3.0
- **Validated on RHOAI 3.2:** quay.io/modh/vllm:rhoai-3.2-cuda
- **Validated on RHAIIS 3.2.5:** http://registry.redhat.io/rhaiis/vllm-cuda-rhel9:3.2.5-1766067105
- **Validated on vLLM:** 0.11.2

## Key Features
Mistral Large 3 consists of two main architectural components:
- **A Granular MoE Language Model with 673B params and 39B active**
- **A 2.5B Vision Encoder**

The Mistral Large 3 Instruct model offers the following capabilities:
- **Vision**: Enables the model to analyze images and provide insights based on visual content, in addition to text.
- **Multilingual**: Supports dozens of languages, including English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese, Japanese, Korean, Arabic.
- **System Prompt**: Maintains strong adherence and support for system prompts.
- **Agentic**: Offers best-in-class agentic capabilities with native function calling and JSON outputting.
- **Frontier**: Delivers best-in-class performance.
- **Apache 2.0 License**: Open-source license allowing usage and modification for both commercial and non-commercial purposes.
- **Large Context Window**: Supports a 256k context window.

## Use Cases
With powerful long-context performance, stable and consistent cross-domain behavior, Mistral Large 3 is perfect for:
- Long Document Understanding
- Powerful Daily-Driver AI Assistants
- State-of-the-Art Agentic and Tool-Use Capabilities
- Enterprise Knowledge Work
- General Coding Assistant

And enterprise-grade use cases requiring frontier capabilities.

## Recommended Settings

We recommend deploying Large 3 in a client-server configuration with the following best practices:

- **System Prompt**: Define a clear environment and use case, including guidance on how to effectively leverage tools in agentic systems.
- **Sampling Parameters**: Use a temperature below 0.1 for daily-driver and production environments ; Higher temperatures may be explored for creative use cases - developers are encouraged to experiment with alternative settings.
- **Tools**: Keep the set of tools well-defined and limit their number to the minimum required for the use case - Avoiding overloading the model with an excessive number of tools.
- **Vision**: When deploying with vision capabilities, we recommend maintaining an aspect ratio close to 1:1 (width-to-height) for images. Avoiding the use of overly thin or wide images - crop them as needed to ensure optimal performance.

### Known Issues / Limitations

- **Not a dedicated reasoning model**: Dedicated reasoning models can outperform Mistral Large 3 in strict reasoning use cases.
- **Behind vision-first models in multimodal tasks**: Mistral Large 3 can lag behind models optimized for vision tasks and use cases.
- **Complex deployment**: Due to its large size and architecture, the model can be challenging to deploy efficiently with constrained resources or at scale.

## Benchmark Results

> [!Warning]
> For tasks requiring context lengths less than 32k tokens, you should see no performance degradation as compared to [Mistral-Large-3-675B-Instruct-2512](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512),
>  for tasks requiring more context you might experience a small degradation in performance.

We compare Mistral Large 3 to similar sized models.

![image](https://cdn-uploads.huggingface.co/production/uploads/64161701107962562e9b1006/IrPlvUUD-5-Phwi9QSevh.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/64161701107962562e9b1006/fDFEymz4HZNsqFARB4u9Y.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/64161701107962562e9b1006/eMdaAPcjOo8VyoGyFKxrE.png)

## Usage

The model can be used with the following frameworks;
- [`vllm`](https://github.com/vllm-project/vllm): See [here](#vllm)
  
### vLLM

We recommend using this model with [vLLM](https://github.com/vllm-project/vllm).

#### Installation

Make sure to install **vllm >= 1.12.0**:

```
pip install vllm --upgrade
```

Doing so should automatically install [`mistral_common >= 1.8.6`](https://github.com/mistralai/mistral-common/releases/tag/v1.8.6).

To check:
```
python -c "import mistral_common; print(mistral_common.__version__)"
```

You can also make use of a ready-to-go [docker image](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) or on the [docker hub](https://hub.docker.com/layers/vllm/vllm-openai/latest).

#### Serve

We recommend to use this format if you plan to deploy Mistral Large 3 has it achieves performance similar to FP8 for less memory. However please note that for large context (`> 64k`) we observed a subsequent drop of performance. In such cases, please use the FP8 weights. Otherwise on B200 (Blackwell 200) we observe a significant speed-up and a minor regression on vision datasets probably due to the calibration that was performed mainly on text data.

To take full advantage of the NVFP4 format, consider using B200 GPUs that has a dedicated architecture to maximize its performance. For those who do not have access to this GPU generation, vLLM has the awesome feature to fall back to Marlin FP4 which allows you to still run the quantized model in older generations (A100, H100). You won't notice a speed-up in comparison with FP8 quantization but still benefit a memory gain.

A simple launch command is:

```bash
vllm serve mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4 \
  --max-model-len 262144 --tensor-parallel-size 8 \
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --enable-auto-tool-choice --tool-call-parser mistral
```

Key parameter notes:

* enable-auto-tool-choice: Required when enabling tool usage.
* tool-call-parser mistral: Required when enabling tool usage.


Additional flags:

* You can set `--max-model-len` to preserve memory. By default it is set to `262144` which is quite large but not necessary for most scenarios.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency.

<details>
  <summary>Deploy on <strong>Red Hat AI Inference Server</strong></summary>
  
```bash
podman run --rm -it --device nvidia.com/gpu=all -p 8000:8000 \
 --ipc=host \
--env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
--env "HF_HUB_OFFLINE=0" -v ~/.cache/vllm:/home/vllm/.cache \
--name=vllm \
registry.access.redhat.com/rhaiis/rh-vllm-cuda \
vllm serve \
--tensor-parallel-size 8 \
--max-model-len 32768  \
--enforce-eager --model RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4
```
</details>


<details>
  <summary>Deploy on <strong>Red Hat Openshift AI</strong></summary>
  
```python
# Setting up vllm server with ServingRuntime
# Save as: vllm-servingruntime.yaml
apiVersion: serving.kserve.io/v1alpha1
kind: ServingRuntime
metadata:
 name: vllm-cuda-runtime # OPTIONAL CHANGE: set a unique name
 annotations:
   openshift.io/display-name: vLLM NVIDIA GPU ServingRuntime for KServe
   opendatahub.io/recommended-accelerators: '["nvidia.com/gpu"]'
 labels:
   opendatahub.io/dashboard: 'true'
spec:
 annotations:
   prometheus.io/port: '8080'
   prometheus.io/path: '/metrics'
 multiModel: false
 supportedModelFormats:
   - autoSelect: true
     name: vLLM
 containers:
   - name: kserve-container
     image: quay.io/modh/vllm:rhoai-3.2-cuda # CHANGE if needed. If AMD: quay.io/modh/vllm:rhoai-3.2-rocm
     command:
       - python
       - -m
       - vllm.entrypoints.openai.api_server
     args:
       - "--port=8080"
       - "--model=/mnt/models"
       - "--served-model-name={{.Name}}"
     env:
       - name: HF_HOME
         value: /tmp/hf_home
     ports:
       - containerPort: 8080
         protocol: TCP
```

```python
# Attach model to vllm server. This is an NVIDIA template
# Save as: inferenceservice.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    openshift.io/display-name: Mistral-Large-3-675B-Instruct-2512-NVFP4 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: Mistral-Large-3-675B-Instruct-2512-NVFP4       # specify model name. This value will be used to invoke the model in the payload
  labels:
    opendatahub.io/dashboard: 'true'
spec:
  predictor:
    maxReplicas: 1
    minReplicas: 1
    model:
      modelFormat:
        name: vLLM
      name: ''
      resources:
        limits:
          cpu: '2'			# this is model specific
          memory: 8Gi		# this is model specific
          nvidia.com/gpu: '1'	# this is accelerator specific
        requests:			# same comment for this block
          cpu: '1'
          memory: 4Gi
          nvidia.com/gpu: '1'
      runtime: vllm-cuda-runtime	# must match the ServingRuntime name above
      storageUri: oci://registry.redhat.io/rhai/modelcar-mistral-large-3-675b-instruct-2512-nvfp4:3.0
    tolerations:
    - effect: NoSchedule
      key: nvidia.com/gpu
      operator: Exists
```

```bash
# make sure first to be in the project where you want to deploy the model
# oc project <project-name>

# apply both resources to run model

# Apply the ServingRuntime
oc apply -f vllm-servingruntime.yaml

```

```python
# Replace <inference-service-name> and <cluster-ingress-domain> below:
# - Run `oc get inferenceservice` to find your URL if unsure.

# Call the server using curl:
curl https://<inference-service-name>-predictor-default.<domain>/v1/chat/completions
        -H "Content-Type: application/json" \
        -d '{
    "model": "Mistral-Large-3-675B-Instruct-2512-NVFP4",
    "stream": true,
    "stream_options": {
        "include_usage": true
    },
    "max_tokens": 1,
    "messages": [
        {
            "role": "user",
            "content": "How can a bee fly when its wings are so small?"
        }
    ]
}'

```

See [Red Hat Openshift AI documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai/2025) for more details.
</details>

#### Usage of the model

Here we asumme that the model `mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` is served and you can ping it to the domain `localhost` with the port `8000` which is the default for vLLM.

<details>
  <summary>Vision Reasoning</summary>

Let's see if Mistral Large 3 knows when to pick a fight !

```python
from datetime import datetime, timedelta

from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 262144

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
    max_tokens=MAX_TOK,
)

print(response.choices[0].message.content)
```
</details>

<details>
  <summary>Function Calling</summary>

Let's solve some equations thanks to our simple Python calculator tool.

```python
import json
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 262144

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

image_url = "https://math-coaching.com/img/fiche/46/expressions-mathematiques.jpg"


def my_calculator(expression: str) -> str:
    return str(eval(expression))


tools = [
    {
        "type": "function",
        "function": {
            "name": "my_calculator",
            "description": "A calculator that can evaluate a mathematical equation and compute its results.",
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
    max_tokens=MAX_TOK,
    tools=tools,
    tool_choice="auto",
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
    max_tokens=MAX_TOK,
)

print(response.choices[0].message.content)
```

</details>

<details>
  <summary>Text-Only Request</summary>

Mistral Large 3 can follow your instructions down to the letter.

```python
from openai import OpenAI
from huggingface_hub import hf_hub_download

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

TEMP = 0.15
MAX_TOK = 262144

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
```

</details>

## Red Hat AI Evaluations

As part of the model validation effort, Red Hat conducted independent accuracy evaluations and the results are presented below.
The model was evaluated with [vLLM](https://vllm.ai/) version 0.12.0 and either [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) or
[lighteval](https://github.com/huggingface/lighteval) depending on the benchmark.

<details>
<summary>Evaluation commands</summary>
  
All evaluations were conducted using the vLLM server interface.
The server is first initialized with the following command on 4 B200 GPUs:
```bash
vllm serve RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4 \
  --max-model-len 64000 \
  --tensor-parallel-size 4 \
  --tokenizer_mode mistral \
  --config_format mistral \
  --load_format mistral \
  --limit-mm-per-prompt '{"image": 10}'
```

MMLU-Pro, IFEval and MMMU were evaluated using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) as follows.
```bash
lm_eval \
  --model local-chat-completions \
  --tasks mmlu_pro,ifeval,mmmu_val \
  --model_args "model=RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4,max_length=64000,base_url=http://0.0.0.0:8000/v1/chat/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer_backend=None,timeout=1200,max_images=10" \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path results_lmeval_mistral_large_3 \
  --gen_kwargs "do_sample=True,temperature=0.15,max_gen_toks=42000"
```

AIME25, GPQA Diamond and Math 500 were evaluated using [lighteval](https://github.com/huggingface/lighteval) as follows.

litellm_config.yaml
```yaml
model_parameters:
  provider: "hosted_vllm"
  model_name: "hosted_vllm/RedHatAI/Mistral-Large-3-675B-Instruct-2512-NVFP4"
  base_url: "http://0.0.0.0:8000/v1"
  api_key: ""
  timeout: 1200
  concurrent_requests: 64
  generation_parameters:
    temperature: 0.15
    max_new_tokens: 42000
```

```bash
lighteval endpoint litellm litellm_config.yaml \
"aime25|0,math_500|0,gpqa:diamond|0" \
--output-dir results_lighteval_mistral_large_3 \
--save-details
```

</details>

<table>
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>RedHatAI/Mistral-Small-3.2-24B-Instruct-2506</th>
      <th>RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4</th>
      <th>Recovery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MMLU-Pro</td>
      <td>50.60</td>
      <td>54.54</td>
      <td>107.8%</td>
    </tr>
    <tr>
      <td>IFEval</td>
      <td>85.37</td>
      <td>83.77</td>
      <td>98.1%</td>
    </tr>
    <tr>
      <td>MMMU</td>
      <td>59.33</td>
      <td>56.65</td>
      <td>95.5%</td>
    </tr>
    <tr>
      <td>AIME25</td>
      <td>43.75</td>
      <td>33.33</td>
      <td>76.2%</td>
    </tr>
    <tr>
      <td>GPQA Diamond</td>
      <td>69.02</td>
      <td>70.54</td>
      <td>102.2%</td>
    </tr>
    <tr>
      <td>MATH 500</td>
      <td>84.87</td>
      <td>77.47</td>
      <td>91.3%</td>
    </tr>
  </tbody>
</table>

## License

This model is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt).

*You must not use this model in a manner that infringes, misappropriates, or otherwise violates any third party’s rights, including intellectual property rights.*