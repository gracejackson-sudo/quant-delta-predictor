---
tags:
- vllm
- sparsity
- quantization
- int4
pipeline_tag: text-generation
license: llama3.1
base_model: neuralmagic/Sparse-Llama-3.1-8B-evolcodealpaca-2of4
datasets:
- theblackcat102/evol-codealpaca-v1
language:
- en
---

# Sparse-Llama-3.1-8B-evolcodealpaca-2of4-quantized.w4a16

## Model Overview
- **Model Architecture:** Llama-3.1-8B
  - **Input:** Text
  - **Output:** Text
- **Model Optimizations:**
  - **Sparsity:** 2:4
  - **Weight quantization:** INT4
- **Release Date:** 11/21/2024
- **Version:** 1.0
- **License(s):** [llama3.1](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B/blob/main/LICENSE)
- **Model Developers:** Neural Magic

This is a code completion AI model obtained by fine-tuning the 2:4 sparse [Sparse-Llama-3.1-8B-2of4](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-2of4) on the [evol-codealpaca-v1](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) dataset, followed by quantization
On the [HumanEval](https://arxiv.org/abs/2107.03374) benchmark, it achieves a pass@1 of 50.6, compared to 48.5 for the fine-tuned dense model [Llama-3.1-8B-evolcodealpaca](https://huggingface.co/neuralmagic/Llama-3.1-8B-evolcodealpaca) — demonstrating over **100% accuracy recovery**.


### Model Optimizations

This model was obtained by quantizing the weights of [Sparse-Llama-3.1-8B-evolcodealpaca-2of4](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-evolcodealpaca-2of4) to INT4 data type.
This optimization reduces the number of bits per parameter from 16 to 4, reducing the disk size and GPU memory requirements by approximately 75%.
That is on top of the reduction of 50% of weights via 2:4 pruning employed on [Sparse-Llama-3.1-8B-evolcodealpaca-2of4](https://huggingface.co/neuralmagic/Sparse-Llama-3.1-8B-evolcodealpaca-2of4).

Only the weights of the linear operators within transformers blocks are quantized. Symmetric per-channel quantization is applied, in which a linear scaling per output dimension maps the INT4 and floating point representations of the quantized weights.
The [GPTQ](https://arxiv.org/abs/2210.17323) algorithm is applied for quantization, as implemented in the [llm-compressor](https://github.com/vllm-project/llm-compressor) library.


## Deployment with vLLM

This model can be deployed efficiently using the [vLLM](https://docs.vllm.ai/en/latest/) backend. vLLM aslo supports OpenAI-compatible serving. See the [documentation](https://docs.vllm.ai/en/latest/) for more details.


## Evaluation

This model was evaluated on Neural Magic's fork of [EvalPlus](https://github.com/neuralmagic/evalplus).

### Accuracy
#### Human Benchmark
<table>
    <tr>
        <td><strong>Metric</strong></td>
        <td style="text-align: center"><strong>Llama-3.1-8B-evolcodealpaca</strong></td>
        <td style="text-align: center"><strong>Sparse-Llama-3.1-8B-evolcodealpaca-2of4</strong></td>
        <td style="text-align: center"><strong>Sparse-Llama-3.1-8B-evolcodealpaca-2of4-quantized.w4a16</strong></td>
    </tr>
    <tr>
        <td>HumanEval pass@1</td>
        <td style="text-align: center">48.5</td>
        <td style="text-align: center">49.1</td>
        <td style="text-align: center">50.6</td>
    </tr>
    <tr>
        <td>HumanEval+ pass@1</td>
        <td style="text-align: center">44.2</td>
        <td style="text-align: center">46.3</td>
        <td style="text-align: center">48.0</td>
    </tr>
</table>