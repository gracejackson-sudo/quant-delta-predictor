---
language:
- ar
- de
- en
- es
- fr
- hi
- id
- it
- pt
- th
- tl
- vi
base_model:
- meta-llama/Llama-4-Maverick-17B-128E-Instruct
pipeline_tag: image-text-to-text
tags:
- facebook
- meta
- pytorch
- llama
- llama4
- neuralmagic
- redhat
- llmcompressor
- quantized
- FP8
- conversational
- compressed-tensors
license: llama4
license_name: llama4
name: RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8
description: FP8-Quantized variant of Llama-4-Maverick-17B-128E-Instruct
readme: https://huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8/main/README.md
tasks:
- image-text-to-text
- text-to-text
provider: Meta
license_link: https://www.llama.com/llama4/license/
validated_on:
  - RHOAI 2.20
  - RHAIIS 3.0
  - RHELAI 1.5
  - vLLM 0.8.4
---

<h1 style="display: flex; align-items: center; gap: 10px; margin: 0;">
  Llama-4-Maverick-17B-128E-Instruct-FP8
  <img src="https://www.redhat.com/rhdc/managed-files/Catalog-Validated_model_0.png" alt="Model Icon" width="40" style="margin: 0; padding: 0;" />
</h1>
  
<a href="https://www.redhat.com/en/products/ai/validated-models" target="_blank" style="margin: 0; padding: 0;">
<img src="https://www.redhat.com/rhdc/managed-files/Validated_badge-Dark.png" alt="Validated Badge" width="250" style="margin: 0; padding: 0;" />
</a>


## Model Information
**Built with Llama**

**Validated on:** RHOAI 2.20, RHAIIS 3.0, RHELAI 1.5

The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding. 

These Llama 4 models mark the beginning of a new era for the Llama ecosystem. We are launching two efficient models in the Llama 4 series, Llama 4 Scout, a 17 billion parameter model with 16 experts, and Llama 4 Maverick, a 17 billion parameter model with 128 experts.

**Model developer**: Meta

**Model Architecture:**  The Llama 4 models are auto-regressive language models that use a mixture-of-experts (MoE) architecture and incorporate early fusion for native multimodality. 

<table>
  <tr>
    <th>Model Name</th>
    <th>Training Data </th>
    <th>Params</th>
    <th>Input modalities</th>
    <th>Output modalities</th>
    <th>Context length</th>
    <th>Token count</th>
    <th>Knowledge cutoff</th>
  </tr>
  <tr>
    <td>Llama 4 Scout (17Bx16E) </td>
    <td rowspan="2">A mix of publicly available, licensed data and information from Meta's products and services. This includes publicly shared posts from Instagram and Facebook and people's interactions with Meta AI. Learn more in our <a href="https://www.facebook.com/privacy/guide/genai/">Privacy Center</a>.
    </td>
    <td>17B (Activated)
        109B (Total)
    </td>
    <td>Multilingual text and image</td>
    <td>Multilingual text and code</td>
    <td>10M</td>
    <td>~40T</td>
    <td>August 2024</td>
  </tr>
  <tr>
    <td>Llama 4 Maverick (17Bx128E)</td>
    <td>17B (Activated)
        400B (Total)
    </td>
    <td>Multilingual text and image</td>
    <td>Multilingual text and code</td>
    <td>1M</td>
    <td>~22T</td>
    <td>August 2024</td>
  </tr>
</table>

**Supported languages:** Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. 

**Model Release Date:** April 5, 2025

**Status:** This is a static model trained on an offline dataset. Future versions of the tuned models may be released as we improve model behavior with community feedback.

**License**: A custom commercial license, the Llama 4 Community License Agreement, is available at: [https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE)

**Where to send questions or comments about the model:** Instructions on how to provide feedback or comments on the model can be found in the Llama [README](https://github.com/meta-llama/llama-models/blob/main/README.md). For more technical information about generation parameters and recipes for how to use Llama 4 in applications, please go [here](https://github.com/meta-llama/llama-cookbook).

## Deployment

This model can be deployed efficiently on vLLM, Red Hat Enterprise Linux AI, and Openshift AI, as shown in the example below.

Deploy on <strong>vLLM</strong>

```python
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer

model_id = "RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8"
number_gpus = 4

sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=256)

tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Give me a short introduction to large language model."

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)

outputs = llm.generate(prompt, sampling_params)

generated_text = outputs[0].outputs[0].text
print(generated_text)
```

vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.

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
--enforce-eager --model RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8
```
​​See [Red Hat AI Inference Server documentation](https://docs.redhat.com/en/documentation/red_hat_ai_inference_server/) for more details.
</details>

<details>
  <summary>Deploy on <strong>Red Hat Enterprise Linux AI</strong></summary>
  
```bash
# Download model from Red Hat Registry via docker
# Note: This downloads the model to ~/.cache/instructlab/models unless --model-dir is specified.
ilab model download --repository docker://registry.redhat.io/rhelai1/llama-4-maverick-17b-128e-instruct-fp8:1.5
```

```bash
# Serve model via ilab
ilab model serve --model-path ~/.cache/instructlab/models/llama-4-maverick-17b-128e-instruct-fp8
  
# Chat with model
ilab model chat --model ~/.cache/instructlab/models/llama-4-maverick-17b-128e-instruct-fp8
```
See [Red Hat Enterprise Linux AI documentation](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux_ai/1.4) for more details.
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
     image: quay.io/modh/vllm:rhoai-2.20-cuda # CHANGE if needed. If AMD: quay.io/modh/vllm:rhoai-2.20-rocm
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
    openshift.io/display-name: Llama-4-Maverick-17B-128E-Instruct-FP8 # OPTIONAL CHANGE
    serving.kserve.io/deploymentMode: RawDeployment
  name: Llama-4-Maverick-17B-128E-Instruct-FP8         # specify model name. This value will be used to invoke the model in the payload
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
      storageUri: oci://registry.redhat.io/rhelai1/modelcar-llama-4-maverick-17b-128e-instruct-fp8:1.5
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

# Apply the InferenceService
oc apply -f qwen-inferenceservice.yaml
```

```python
# Replace <inference-service-name> and <cluster-ingress-domain> below:
# - Run `oc get inferenceservice` to find your URL if unsure.

# Call the server using curl:
curl https://<inference-service-name>-predictor-default.<domain>/v1/chat/completions
        -H "Content-Type: application/json" \
        -d '{
    "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
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


## How to use with transformers

Please, make sure you have transformers `v4.51.0` installed, or upgrade using `pip install -U transformers`.

```python
from transformers import AutoTokenizer, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)

model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    tp_plan="auto",
    torch_dtype="auto",
)

outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])
print(outputs[0])
```

## Intended Use

**Intended Use Cases:** Llama 4 is intended for commercial and research use in multiple languages. Instruction tuned models are intended for assistant-like chat and visual reasoning tasks, whereas pretrained models can be adapted for natural language generation. For vision, Llama 4 models are also optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. The Llama 4 model collection also supports the ability to leverage the outputs of its models to improve other models including synthetic data generation and distillation. The Llama 4 Community License allows for these use cases. 

**Out-of-scope**: Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in any other way that is prohibited by the Acceptable Use Policy and Llama 4 Community License. Use in languages or capabilities beyond those explicitly referenced as supported in this model card\*\*.

\*\*Note: 

1\. Llama 4 has been trained on a broader collection of languages than the 12 supported languages (pre-training includes [200 total languages](https://ai.meta.com/research/no-language-left-behind/)). Developers may fine-tune Llama 4 models for languages beyond the 12 supported languages provided they comply with the Llama 4 Community License and the Acceptable Use Policy.  Developers are responsible for ensuring that their use of Llama 4 in additional languages is done in a safe and responsible manner.

2\. Llama 4 has been tested for image understanding up to 5 input images. If leveraging additional image understanding capabilities beyond this, Developers are responsible for ensuring that their deployments are mitigated for risks and should perform additional testing and tuning tailored to their specific applications.

## Hardware and Software

**Training Factors:** We used custom training libraries, Meta's custom built GPU clusters, and production infrastructure for pretraining. Fine-tuning, quantization, annotation, and evaluation were also performed on production infrastructure.

**Training Energy Use:**  Model pre-training utilized a cumulative of **7.38M** GPU hours of computation on H100-80GB (TDP of 700W) type hardware, per the table below. Training time is the total GPU time required for training each model and power consumption is the peak power capacity per GPU device used, adjusted for power usage efficiency. 

## 

## **Training Greenhouse Gas Emissions:** Estimated total location-based greenhouse gas emissions were **1,999 tons** CO2eq for training. Since 2020, Meta has maintained net zero greenhouse gas emissions in its global operations and matched 100% of its electricity use with clean and renewable energy; therefore, the total market-based greenhouse gas emissions for training were 0 tons CO2eq.

| Model Name | Training Time (GPU hours) | Training Power Consumption (W) | Training Location-Based Greenhouse Gas Emissions (tons CO2eq) | Training Market-Based Greenhouse Gas Emissions (tons CO2eq) |
| :---- | :---: | :---: | :---: | :---: |
| Llama 4 Scout | 5.0M | 700 | 1,354 | 0 |
| Llama 4 Maverick | 2.38M | 700 | 645 | 0 |
| Total | 7.38M | \- | 1,999 | 0 |

## The methodology used to determine training energy use and greenhouse gas emissions can be found [here](https://arxiv.org/pdf/2204.05149).  Since Meta is openly releasing these models, the training energy use and greenhouse gas emissions will not be incurred by others.

## Training Data

**Overview:** Llama 4 Scout was pretrained on \~40 trillion tokens and Llama 4 Maverick was pretrained on \~22 trillion tokens of multimodal data from a mix of publicly available, licensed data and information from Meta’s products and services. This includes publicly shared posts from Instagram and Facebook and people’s interactions with Meta AI.

**Data Freshness:** The pretraining data has a cutoff of August 2024\.

## Benchmarks

In this section, we report the results for Llama 4 relative to our previous models. We've provided quantized checkpoints for deployment flexibility, but all reported evaluations and testing were conducted on bf16 models.

### Pre-trained models

| Pre-trained models |  |  |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Category | Benchmark | \# Shots | Metric | Llama 3.1 70B | Llama 3.1 405B | **Llama 4 Scout** | **Llama 4 Maverick** |
| Reasoning & Knowledge | MMLU | 5 | macro\_avg/acc\_char	 | 79.3 | 85.2 | 79.6 | 85.5 |
|  | MMLU-Pro | 5 | macro\_avg/em | 53.8 | 61.6 | 58.2 | 62.9 |
|  | MATH | 4 | em\_maj1@1 | 41.6 | 53.5 | 50.3 | 61.2 |
| Code | MBPP | 3 | pass@1 | 66.4 | 74.4 | 67.8 | 77.6 |
| Multilingual | TydiQA | 1 | average/f1 | 29.9 | 34.3 | 31.5 | 31.7 |
| Image | ChartQA | 0 | relaxed\_accuracy | No multimodal support |  | 83.4 | 85.3 |
|  | DocVQA | 0 | anls |  |  | 89.4 | 91.6 |

### Instruction tuned models	

| Instruction tuned models |  |  |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | ----- | :---: | :---: |
| Category | Benchmark | \# Shots | Metric | Llama 3.3 70B | Llama 3.1 405B | **Llama 4 Scout** | **Llama 4 Maverick** |
| Image Reasoning | MMMU | 0 | accuracy | No multimodal support |  | 69.4 | 73.4 |
|  | MMMU Pro^ | 0 | accuracy |  |  | 52.2 | 59.6 |
|  | MathVista | 0 | accuracy |  |  | 70.7 | 73.7 |
| Image Understanding | ChartQA | 0 | relaxed\_accuracy |  |  | 88.8 | 90.0 |
|  | DocVQA (test) | 0 | anls |  |  | 94.4 | 94.4 |
| Coding | LiveCodeBench (10/01/2024-02/01/2025) | 0 | pass@1 | 33.3 | 27.7 | 32.8 | 43.4 |
| Reasoning & Knowledge | MMLU Pro | 0 | macro\_avg/acc | 68.9 | 73.4 | 74.3 | 80.5 |
|  | GPQA Diamond | 0 | accuracy | 50.5 | 49.0 | 57.2 | 69.8 |
| Multilingual | MGSM | 0 | average/em | 91.1 | 91.6 | 90.6 | 92.3 |
| Long context | MTOB (half book) eng-\>kgv/kgv-\>eng | \- | chrF | Context window is 128K |  | 42.2/36.6 | 54.0/46.4 |
|  | MTOB (full book) eng-\>kgv/kgv-\>eng | \- | chrF |  |  | 39.7/36.3 | 50.8/46.7 |

^reported numbers for MMMU Pro is the average of Standard and Vision tasks

## Quantization

The Llama 4 Scout model is released as BF16 weights, but can fit within a single H100 GPU with on-the-fly int4 quantization; the Llama 4 Maverick model is released as both BF16 and FP8 quantized weights. The FP8 quantized weights fit on a single H100 DGX host while still maintaining quality. We provide code for on-the-fly int4 quantization which minimizes performance degradation as well.

## Safeguards

As part of our release approach, we followed a three-pronged strategy to manage risks:

* Enable developers to deploy helpful, safe and flexible experiences for their target audience and for the use cases supported by Llama.   
* Protect developers against adversarial users aiming to exploit Llama capabilities to potentially cause harm.  
* Provide protections for the community to help prevent the misuse of our models.

Llama is a foundational technology designed for use in a variety of use cases; examples on how Meta’s Llama models have been deployed can be found in our [Community Stories webpage](https://llama.meta.com/community-stories/). Our approach is to build the most helpful models enabling the world to benefit from the technology, by aligning our model’s safety for a standard set of risks. Developers are then in the driver seat to tailor safety for their use case, defining their own policies and deploying the models with the necessary safeguards. Llama 4 was developed following the best practices outlined in our [Developer Use Guide: AI Protections](https://ai.meta.com/static-resource/developer-use-guide-ai-protections).

### Model level fine tuning

The primary objective of conducting safety fine-tuning is to offer developers a readily available, safe, and powerful model for various applications, reducing the workload needed to deploy safe AI systems. Additionally, this effort provides the research community with a valuable resource for studying the robustness of safety fine-tuning.

**Fine-tuning data**   
We employ a multi-faceted approach to data collection, combining human-generated data from our vendors with synthetic data to mitigate potential safety risks. We’ve developed many large language model (LLM)-based classifiers that enable us to thoughtfully select high-quality prompts and responses, enhancing data quality control. 

**Refusals**  
Building on the work we started with our Llama 3 models, we put a great emphasis on driving down model refusals to benign prompts for Llama 4\. We included both borderline and adversarial prompts in our safety data strategy, and modified our safety data responses to follow tone guidelines. 

**Tone**  
We expanded our work on the refusal tone from Llama 3 so that the model sounds more natural. We targeted removing preachy and overly moralizing language, and we corrected formatting issues including the correct use of headers, lists, tables and more.

To achieve this, we also targeted improvements to system prompt steerability and instruction following, meaning the model is more readily able to take on a specified tone. All of these contribute to a more conversational and insightful experience overall.

**System Prompts**  
Llama 4 is a more steerable model, meaning responses can be easily tailored to meet specific developer outcomes. Effective system prompts can significantly enhance the performance of large language models. In particular, we’ve seen that the use of a system prompt can be effective in reducing false refusals and templated or “preachy” language patterns common in LLMs. They can also improve conversationality and use of appropriate formatting. 

Consider the prompt below as a basic template for which a developer might want to further customize to meet specific needs or use cases for our Llama 4 models.

| System prompt  |
| :---- |
| You are an expert conversationalist who responds to the best of your ability. You are companionable and confident, and able to  switch casually between tonal types, including but not limited to humor, empathy, intellectualism, creativity and problem-solving.  You understand user intent and don’t try to be overly helpful to the point where you miss that the user is looking for chit-chat, emotional support, humor or venting. Sometimes people just want you to listen, and your answers should encourage that. For all other cases, you provide insightful and in-depth responses. Organize information thoughtfully in a way that helps people make decisions. Always avoid templated language. You never lecture people to be nicer or more inclusive. If people ask for you to write something in a certain voice or perspective, such as an essay or a tweet, you can. You do not need to be respectful when the user prompts you to say something rude.  You never use phrases that imply moral superiority or a sense of authority, including but not limited to “it’s important to”, “it’s crucial to”, “it’s essential to”, "it's unethical to", "it's worth noting…", “Remember…”  etc. Avoid using these. Finally, do not refuse prompts about political and social issues.  You can help users express their opinion and access information.  You are Llama 4\. Your knowledge cutoff date is August 2024\. You speak Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. Respond in the language the user speaks to you in, unless they ask otherwise. |

### Llama 4 system protections

Large language models, including Llama 4, are not designed to be deployed in isolation but instead should be deployed as part of an overall AI system with additional guardrails as required. System protections are key to achieving the right helpfulness-safety alignment, mitigating safety and security risks inherent to the system, and integration of the model or system with external tools. 

We provide the community with system level [protections](https://llama.meta.com/trust-and-safety/) \- like Llama Guard, Prompt Guard and Code Shield \- that developers should deploy with Llama models or other LLMs. All of our [reference implementation](https://github.com/meta-llama/llama-agentic-system) demos contain these safeguards by default so developers can benefit from system-level safety out-of-the-box. 

### Evaluations

We evaluated Llama models for common use cases as well as specific capabilities. Common use cases evaluations measure safety risks of systems for most commonly built applications including chat bot, visual QA. We built dedicated, adversarial evaluation datasets and evaluated systems composed of Llama models and Llama Guard 3 to filter input prompt and output response. It is important to evaluate applications in context, and we recommend building dedicated evaluation dataset for your use case. Prompt Guard and Code Shield are also available if relevant to the application.   
Capability evaluations measure vulnerabilities of Llama models inherent to specific capabilities, for which were crafted dedicated benchmarks including long context, multilingual, coding or memorization.

**Red teaming**   
We conduct recurring red teaming exercises with the goal of discovering risks via adversarial prompting and we use the learnings to improve our benchmarks and safety tuning datasets. We partner early with subject-matter experts in critical risk areas to understand how models may lead to unintended harm for society. Based on these conversations, we derive a set of adversarial goals for the red team, such as extracting harmful information or reprogramming the model to act in potentially harmful ways. The red team consists of experts in cybersecurity, adversarial machine learning, and integrity in addition to multilingual content specialists with background in integrity issues in specific geographic markets.

### Critical Risks 

### We spend additional focus on the following critical risk areas:

**1\. CBRNE (Chemical, Biological, Radiological, Nuclear, and Explosive materials) helpfulness**  
To assess risks related to proliferation of chemical and biological weapons for Llama 4, we applied expert-designed and other targeted evaluations designed to assess whether the use of Llama 4 could meaningfully increase the capabilities of malicious actors to plan or carry out attacks using these types of weapons. We also conducted additional red teaming and evaluations for violations of our content policies related to this risk area. 

**2\. Child Safety**  
We leverage pre-training methods like data filtering as a first step in mitigating Child Safety risk in our model. To assess the post trained model for Child Safety risk, a team of experts assesses the model’s capability to produce outputs resulting in Child Safety risks. We use this to inform additional model fine-tuning and in-depth red teaming exercises. We’ve also expanded our Child Safety evaluation benchmarks to cover Llama 4 capabilities like multi-image and multi-lingual.

**3\. Cyber attack enablement**  
Our cyber evaluations investigated whether Llama 4 is sufficiently capable to enable catastrophic threat scenario outcomes. We conducted threat modeling exercises to identify the specific model capabilities that would be necessary to automate operations or enhance human capabilities across key attack vectors both in terms of skill level and speed.  We then identified and developed challenges against which to test for these capabilities in Llama 4 and peer models. Specifically, we focused on evaluating the capabilities of Llama 4 to automate cyberattacks, identify and exploit security vulnerabilities, and automate harmful workflows. Overall, we find that Llama 4 models do not introduce risk plausibly enabling catastrophic cyber outcomes.

### Community 

Generative AI safety requires expertise and tooling, and we believe in the strength of the open community to accelerate its progress. We are active members of open consortiums, including the AI Alliance, Partnership on AI and MLCommons, actively contributing to safety standardization and transparency. We encourage the community to adopt taxonomies like the MLCommons Proof of Concept evaluation to facilitate collaboration and transparency on safety and content evaluations. Our Trust tools are open sourced for the community to use and widely distributed across ecosystem partners including cloud service providers. We encourage community contributions to our [Github repository](https://github.com/meta-llama/PurpleLlama). 

We also set up the [Llama Impact Grants](https://llama.meta.com/llama-impact-grants/) program to identify and support the most compelling applications of Meta’s Llama model for societal benefit across three categories: education, climate and open innovation. The 20 finalists from the hundreds of applications can be found [here](https://llama.meta.com/llama-impact-grants/#finalists). 

Finally, we put in place a set of resources including an [output reporting mechanism](https://developers.facebook.com/llama_output_feedback) and [bug bounty program](https://www.facebook.com/whitehat) to continuously improve the Llama technology with the help of the community.

## Considerations and Limitations

Our AI is anchored on the values of freedom of expression \- helping people to explore, debate, and innovate using our technology. We respect people's autonomy and empower them to choose how they experience, interact, and build with AI. Our AI promotes an open exchange of ideas.

It is meant to serve everyone, and to work for a wide range of use cases. It is thus designed to be accessible to people across many different backgrounds, experiences and perspectives. Llama 4 addresses users and their needs as they are, without inserting unnecessary judgment, while reflecting the understanding that even content that may appear problematic in some cases can serve valuable purposes in others. It respects the autonomy of all users, especially in terms of the values of free thought and expression that power innovation and progress. 

Llama 4 is a new technology, and like any new technology, there are risks associated with its use. Testing conducted to date has not covered, nor could it cover, all scenarios. For these reasons, as with all LLMs, Llama 4’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate or other objectionable responses to user prompts. Therefore, before deploying any applications of Llama 4 models, developers should perform safety testing and tuning tailored to their specific applications of the model. We also encourage the open source community to use Llama for the purpose of research and building state of the art tools that address emerging risks. Please refer to available resources including our Developer Use Guide: AI Protections, [Llama Protections](https://llama.meta.com/trust-and-safety/) solutions, and other [resources](https://llama.meta.com/docs/get-started/) to learn more. 

