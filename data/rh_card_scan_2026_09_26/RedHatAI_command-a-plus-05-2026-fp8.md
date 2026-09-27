# Command A+ 05-2026 FP8
## Model Information

This is a quantized version of [CohereLabs/command-a-plus-05-2026-bf16](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16), quantized to FP8 for efficient deployment.

- **Model Architecture:** Decoder-only Sparse Mixture-of-Experts Transformer
  - **Input:** Text and images
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** FP8
  - **Activation quantization:** FP8
- **Active Parameters:** 25B (218B total)
- **Context Length:** 128K input, 64K output
- **Languages:** 48 languages including English, French, German, Spanish, Chinese, Japanese, Korean, Arabic, and many more
- **License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Model Developer:** [Cohere](https://cohere.com) (quantized by Red Hat)

Command A+ is an open-source model optimized for agentic, multilingual, and reasoning-heavy tasks with enterprise performance focus and vision input support. This FP8 quantization reduces GPU memory requirements by roughly 50% compared to BF16, enabling deployment on 4x H100 or 2x B200 GPUs.

For more details about the base model, including tool use and citation capabilities, see the [Cohere model card](https://huggingface.co/CohereLabs/command-a-plus-05-2026) and the [Command A+ blog post](http://cohere.com/blog/command-a-plus).

## Deployment

### Use with vLLM

Requires `vllm>=0.21.0` and Cohere's `melody` library for accurate response parsing.

```bash
pip install vllm>=0.21.0 transformers cohere_melody>=0.9.0
```

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

model_id = "RedHatAI/command-a-plus-05-2026-fp8"
number_gpus = 4

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=4096)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": "Give me a short introduction to large language models."}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

llm = LLM(model=model_id, tensor_parallel_size=number_gpus)
outputs = llm.generate(prompt, sampling_params)
generated_text = outputs[0].outputs[0].text
print(generated_text)
```

Serving with tool calling and reasoning support (requires `cohere_melody>=0.9.0`):

```bash
pip install cohere_melody>=0.9.0

vllm serve RedHatAI/command-a-plus-05-2026-fp8 \
  --tensor-parallel-size 4 \
  --tool-call-parser cohere_command4 \
  --reasoning-parser cohere_command4 \
  --enable-auto-tool-choice
```

## Evaluation

All evaluations were conducted using [vLLM](https://docs.vllm.ai/) as the inference backend with temperature=0.6 and top_p=0.95. Results are averaged across multiple random seeds.

- [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness/tree/mmlu-pro-chat-variant) (v0.4.13.dev0) with vLLM v0.22.1
- [lighteval](https://github.com/neuralmagic/lighteval/tree/eldar-fix-litellm) (v0.13.1.dev0) with vLLM v0.22.1
- [BFCL](https://github.com/neuralmagic/gorilla/tree/shubhra/bfcl-vllm-patches) with vLLM v0.24.0

### Accuracy

| Benchmark | CohereLabs/command-a-plus-05-2026-bf16 | RedHatAI/command-a-plus-05-2026-fp8 (this model) | Recovery |
|---|---|---|---|
| AIME 2025 (pass@1, 8 seeds) | 90.42 | 91.25 | 100.9% |
| MATH-500 (pass@1, 3 seeds) | 89.80 | 91.20 | 101.6% |
| GPQA Diamond (pass@1, 3 seeds) | 74.41 | 76.77 | 103.2% |
| MMLU (5-shot CoT, exact_match) | 82.75 | 84.33 | 101.9% |
| MMLU-Pro (0-shot, exact_match) | 77.76 | 79.31 | 102.0% |
| GSM8K Platinum (0-shot CoT, exact_match) | 94.98 | 95.12 | 100.1% |
| IFEval (0-shot, inst_level_strict_acc) | 86.49 | 93.96 | 108.6% |
| IFEval (0-shot, prompt_level_strict_acc) | 84.35 | 91.56 | 108.5% |
| BFCL Non-Live AST (Acc) | 86.40 | 86.17 | 99.7% |
| BFCL Multi-Turn (Acc) | 22.75 | 23.50 | 103.3% |
