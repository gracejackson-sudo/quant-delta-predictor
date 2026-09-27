---
license: apache-2.0
base_model: mistralai/Mistral-7B-Instruct-v0.3

model-index:
- name: Mistral-7B-Instruct-v0.3-GPTQ-4bit
  results:
  # AI2 Reasoning Challenge (25-Shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: AI2 Reasoning Challenge (25-Shot)
      type: ai2_arc
      config: ARC-Challenge
      split: test
      args:
        num_few_shot: 25
    metrics:
       - type: acc_norm
         name: normalized accuracy
         value: 63.40
  # HellaSwag (10-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: HellaSwag (10-Shot)
      type: hellaswag
      split: validation
      args:
        num_few_shot: 10
    metrics:
       - type: acc_norm
         name: normalized accuracy
         value: 84.04
  # TruthfulQA (0-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: TruthfulQA (0-shot)
      type: truthful_qa
      config: multiple_choice
      split: validation
      args:
        num_few_shot: 0
    metrics:
       - type: mc2
         value: 57.48
  # GSM8k (5-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: GSM8k (5-shot)
      type: gsm8k
      config: main
      split: test
      args:
        num_few_shot: 5
    metrics:
       - type: acc
         name: accuracy
         value: 45.41
  # MMLU (5-Shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: MMLU (5-Shot)
      type: cais/mmlu
      config: all
      split: test
      args:
        num_few_shot: 5
    metrics:
       - type: acc
         name: accuracy
         value: 61.07
  # Winogrande (5-shot)
  - task: 
      type: text-generation
      name: Text Generation
    dataset:
      name: Winogrande (5-shot)
      type: winogrande
      config: winogrande_xl
      split: validation
      args:
        num_few_shot: 5
    metrics:
       - type: acc
         name: accuracy
         value: 79.08

---

# Model Card for Mistral-7B-Instruct-v0.3 quantized to 4bit weights

- Weight-only quantization of [Mistral-7B-Instruct-v0.3](mistralai/Mistral-7B-Instruct-v0.3) via GPTQ to 4bits with group_size=128
- GPTQ optimized for 99.75% accuracy recovery relative to the unquantized model

# Open LLM Leaderboard evaluation scores
|                      | Mistral-7B-Instruct-v0.3 | Mistral-7B-Instruct-v0.3-GPTQ-4bit<br>(this model) |
| :------------------: | :----------------------: | :------------------------------------------------: |
| arc-c<br>25-shot     | 63.48                    | 63.40                                              |
| mmlu<br>5-shot       | 61.13                    | 60.89                                              |
| hellaswag<br>10-shot | 84.49                    | 84.04                                              |
| winogrande<br>5-shot | 79.16                    | 79.08                                              |
| gsm8k<br>5-shot      | 43.37                    | 45.41                                              |
| truthfulqa<br>0-shot | 59.65                    | 57.48                                              |
| **Average<br>Accuracy**  | **65.21**                    |              **65.05**                                     |
| **Recovery**             | **100%**                     |              **99.75%**                                     |

# vLLM Inference Performance

This model is ready for optimized inference using the Marlin mixed-precision kernels in vLLM: https://github.com/vllm-project/vllm

Simply start this model as an inference server with:
```bash
python -m vllm.entrypoints.openai.api_server --model neuralmagic/Mistral-7B-Instruct-v0.3-GPTQ-4bit
```

![image/png](https://cdn-uploads.huggingface.co/production/uploads/60466e4b4f40b01b66151416/SC_tYXjoS3yIoOYtfqZ2E.png)

