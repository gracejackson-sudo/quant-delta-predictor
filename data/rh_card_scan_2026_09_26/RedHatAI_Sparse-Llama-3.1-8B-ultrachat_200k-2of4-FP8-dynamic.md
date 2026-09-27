---
tags:
- vllm
- sparsity
pipeline_tag: text-generation
license: llama3.1
base_model: neuralmagic/Sparse-Llama-3.1-8B-ultrachat_200k-2of4
datasets:
- HuggingFaceH4/ultrachat_200k
language:
- en
---

# Sparse-Llama-3.1-8B-ultrachat_200k-2of4-FP8-dynamic

## Model Overview
- **Model Architecture:** Llama-3.1-8B
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Sparsity:** 2:4
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Release Date:** 11/15/2024
- **Version:** 1.0
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

This is a multi-turn conversational AI model obtained by fine-tuning the 2:4 sparse [Sparse-Llama-3.1-8B-2of4](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-2of4) on the [ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset, followed by quantization.
On the [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) benchmark (version 1), it achieves a score of 62.9, compared to 62.0 for the fine-tuned dense model [Llama-3.1-8B-ultrachat_200k](https://huggingface.co/neuralmagic/Llama-3.1-8B-ultrachat_200k) — demonstrating a **99.4% accuracy recovery**.


### Model Optimizations

This model was obtained by quantizing the weights of [Sparse-Llama-3.1-8B-ultrachat_200k-2of4](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-ultrachat_200k-2of4) to FP8 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between FP8 and BF16 representations for each output channel dimension.
Linear scaling factors are computed via by minimizing the mean squarred error (MSE).
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between FP8 and BF16 representations.


## Deployment with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend. vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.


## Evaluation

This model was evaluated on Neural Magic's fork of [AlpacaEval](https://github.com/neuralmagic/alpaca_eval) benchmark.
We adopt the same setup as in [Enabling High-Sparsity Foundational Llama Models with Efficient Pretraining and Deployment](https://arxiv.org/abs/2405.03594), using version 1 of the benchmark and [Llama-2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) as the annotator.

### Accuracy
#### AlpacaEval Benchmark
<table>
    <tr>
        <td><strong>Metric</strong></td>
        <td style="text-align: center"><strong>Llama-3.1-8B-ultrachat_200k</strong></td>
        <td style="text-align: center"><strong>Sparse-Llama-3.1-8B-ultrachat_200k-2of4</strong></td>
        <td style="text-align: center"><strong>Sparse-Llama-3.1-8B-ultrachat_200k-2of4-FP8-dynamic</strong></td>
    </tr>
    <tr>
        <td>Win rate</td>
        <td style="text-align: center">62.0</td>
        <td style="text-align: center">61.1</td>
        <td style="text-align: center">62.9</td>
    </tr>
</table>