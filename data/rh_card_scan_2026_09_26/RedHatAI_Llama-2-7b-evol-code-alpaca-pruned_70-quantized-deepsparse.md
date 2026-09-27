---
base_model: neuralmagic/Llama-2-7b-pruned70-retrained-evolcodealpaca
inference: false
model_type: llama
pipeline_tag: text-generation
datasets:
  - cerebras/SlimPajama-627B
  - theblackcat102/evol-codealpaca-v1
tags:
- sparse
- code
- deepsparse
---

# Llama-2-7b-pruned70-retrained-evolcodealpaca-quant-ds

This repo contains a [70% sparse Llama 2 7B](https://huggingface.co/neuralmagic/Llama-2-7b-pruned70-retrained-evolcodealpaca) finetuned for code generation tasks using the [Evolved CodeAlpaca](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) dataset.
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

model = TextGeneration(model_path="hf:neuralmagic/Llama-2-7b-pruned70-retrained-evolcodealpaca-quant-ds")

input_text = "def fibonacci(n):\n"
outputs = model(input_text, max_new_tokens=100)
print(outputs.generations[0].text)
```

## Evaluation Benchmark Results

Model evaluation metrics and results.

| Benchmark                                      | Metric        | Llama-2-7b-evolcodealpaca  | Llama-2-7b-pruned70-retrained-evolcodealpaca-quant-ds |
|------------------------------------------------|---------------|-------------|-------------------------------|
| [HumanEval](https://arxiv.org/abs/2107.03374)  | pass@1        | 32.03       | 35.02                         |

## Help

For further support, and discussions on these models and AI in general, join [Neural Magic's Slack Community](https://join.slack.com/t/discuss-neuralmagic/shared_invite/zt-q1a1cnvo-YBoICSIw3L1dmQpjBeDurQ)
