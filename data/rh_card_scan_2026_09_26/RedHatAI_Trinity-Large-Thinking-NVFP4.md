---
base_model:
- arcee-ai/Trinity-Large-Thinking
tags:
- afmoe
- nvfp4
- vllm
- compressed-tensors
name: RedHatAI/Trinity-Large-Thinking-NVFP4
---

# NVFP4 Quantized RedHatAI/Trinity-Large-Thinking-NVFP4

This is a preliminary version (and subject to change) of NVFP4 quantized [arcee-ai/Trinity-Large-Thinking ](https://huggingface.co/arcee-ai/Trinity-Large-Thinking/tree/main ) model. 
The model has both weights and activations quantized to NVFP4 format with [vllm-project/llm-compressor](https://github.com/vllm-project/llm-compressor).

It is compatible and tested against vllm main. Run it with ```vllm serve RedHatAI/Trinity-Large-Thinking-NVFP4 --trust-remote-code```
