---
base_model: neuralmagic/Llama-2-7b-pruned70-retrained-instruct
inference: false
model_type: llama
pipeline_tag: text-generation
datasets:
  - garage-bAInd/Open-Platypus
  - Open-Orca/OpenOrca
  - cognitivecomputations/dolphin
tags:
- sparse
- instruct
- deepsparse
---

# Llama-2-7b-pruned70-retrained-instruct-quant-ds

This repo contains a [70% sparse Llama 2 7B](https://huggingface.co/neuralmagic/Llama-2-7b-pruned70-retrained) finetuned for instruction-following tasks using a blend of the Platypus + Open Orca + Dolphin datasets.
It was then quantized to 8-bit weights + activations and exported to deploy with [DeepSparse](https://github.com/neuralmagic/deepsparse), a CPU inference runtime for sparse models.

Official model weights from [Enabling High-Sparsity Foundational Llama Models with Efficient Pretraining and Deployment](https://arxiv.org/abs/2405.03594).

**Authors**: Neural Magic, Cerebras

## Usage

Below we share some code snippets on how to get quickly started with running the model.

### Sparse Transfer

By leveraging a pre-sparsified model's structure, you can efficiently fine-tune on new data, leading to reduced hyperparameter tuning, training times, and computational costs. Learn about this process [here](https://neuralmagic.github.io/docs-v2/get-started/transfer).

### Running the model

For accelerated inference with sparsity on CPUs, deploy with [deepsparse](https://github.com/neuralmagic/deepsparse).

```python
# pip install deepsparse[llm]
from deepsparse import TextGeneration

model = TextGeneration(model_path="hf:neuralmagic/Llama-2-7b-pruned70-retrained-instruct-quant-ds")

input_text = "Write me a poem about Machine Learning."
outputs = model(input_text, max_new_tokens=100)
print(outputs.generations[0].text)
```

## Evaluation Benchmark Results

Model evaluation metrics and results.

| Benchmark                                      | Metric        | Llama-2-7b-instruct  | Llama-2-7b-pruned70-retrained-instruct-quant-ds |
|------------------------------------------------|---------------|-------------|-------------------------------|
| [MMLU](https://arxiv.org/abs/2009.03300)       | 5-shot        | 48.60%      | 41.21%                        |
| [HellaSwag](https://arxiv.org/abs/1905.07830)  | 10-shot       | 79.45%      | 76.88%                        |
| [WinoGrande](https://arxiv.org/abs/1907.10641) | 5-shot        | 75.69%      | 70.24%                        |
| [ARC-c](https://arxiv.org/abs/1911.01547)      | 25-shot       | 53.92%      | 47.61%                        |
| [TruthfulQA](https://arxiv.org/abs/2109.07958) | 0-shot        | 43.63%      | 42.04%                        |
| [GSM8K](https://arxiv.org/abs/2110.14168)      | 5-shot        | 15.92%      | 12.13%                        |

## Help

For further support, and discussions on these models and AI in general, join [Neural Magic's Slack Community](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)