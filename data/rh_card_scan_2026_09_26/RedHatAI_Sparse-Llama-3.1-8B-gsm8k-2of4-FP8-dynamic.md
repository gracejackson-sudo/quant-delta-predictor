---
tags:
- vllm
- sparsity
- quantized
pipeline_tag: text-generation
license: llama3.1
base_model: neuralmagic/Sparse-Llama-3.1-8B-gsm8k-2of4
datasets:
- openai/gsm8k
language:
- en
metrics:
- accuracy
---

# Sparse-Llama-3.1-8B-gsm8k-2of4-FP8-dynamic

## Model Overview
- **Model Architecture:** Llama-3.1-8B
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Sparsity:** 2:4
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
  - **Release Date:** 11/21/2024
- **Version:** 1.0
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

This is AI model especialized in grade-school math obtained by fine-tuning the 2:4 sparse [Sparse-Llama-3.1-8B-2of4](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-2of4) on the [GSM8k](https://huggingface.co/datasets/openai/gsm8k) dataset, followed by one-shot quantization.
It achieves 66.8% 0-shot accuracy on the test set of GSM8k, compared to 66.3% for the fine-tuned dense model [Llama-3.1-8B-gsm8k](https://huggingface.co/neuralmagic/Llama-3.1-8B-gsm8k) — demonstrating over **100.0% accuracy recovery**.
In constrast, the pretrained [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) achieves 50.7% 5-shot accuracy and the sparse foundational [Sparse-Llama-3.1-8B-2of4](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-2of4) model achieves 56.3% 5-shot accuracy.


### Model Optimizations

This model was obtained by quantizing the weights of [Sparse-Llama-3.1-8B-gsm8k-2of4](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-gsm8k-2of4) to INT4 data type.
This optimization reduces the number of bits used to represent weights and activations from 16 to 8, reducing GPU memory requirements (by approximately 50%) and increasing matrix-multiply compute throughput (by approximately 2x).
Weight quantization also reduces disk size requirements by approximately 50%.

Only weights and activations of the linear operators within transformers blocks are quantized.
Weights are quantized with a symmetric static per-channel scheme, where a fixed linear scaling factor is applied between FP8 and BF16 representations for each output channel dimension.
Linear scaling factors are computed via by minimizing the mean squarred error (MSE).
Activations are quantized with a symmetric dynamic per-token scheme, computing a linear scaling factor at runtime for each token between FP8 and BF16 representations.


## Deployment with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend. vLLM also supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.


## Evaluation

This model was evaluated on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Accuracy
#### GSM8k Benchmark
<table>
    <tr>
        <td><strong>Metric</strong></td>
        <td style="text-align: center"><strong>Llama-3.1-8B<br>(5-shot)</strong></td>
        <td style="text-align: center"><strong>Sparse-Llama-3.1-8B-2of4<br>(5-shot)</strong></td>
        <td style="text-align: center"><strong>Llama-3.1-8B-gsm8k<br>(0-shot)</strong></td>
        <td style="text-align: center"><strong>Sparse-Llama-3.1-8B-gsm8k-2of4<br>(0-shot)</strong></td>
        <td style="text-align: center"><strong>Sparse-Llama-3.1-8B-gsm8k-2of4-FP8-dynamic<br>(0-shot)</strong></td>
    </tr>
    <tr>
        <td>Accuracy</td>
        <td style="text-align: center">50.7%</td>
        <td style="text-align: center">56.3%</td>
        <td style="text-align: center">66.3%</td>
        <td style="text-align: center">66.9%</td>
        <td style="text-align: center">66.8%</td>
    </tr>
</table>