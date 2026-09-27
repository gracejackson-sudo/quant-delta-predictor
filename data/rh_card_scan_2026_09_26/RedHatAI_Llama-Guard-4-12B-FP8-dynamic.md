Evaluations are produced with [https://github.com/neuralmagic/GuardBench](https://github.com/neuralmagic/GuardBench) and vLLM as an inference engine.

Evaluations are obtained with `vllm==0.15.0` and bug fixes from this [PR](https://github.com/vllm-project/vllm/pull/34243).
If you are running `vllm > 0.15.0`, you will likely have the bug fixes already applied as the PR landed on main.

| Dataset                   | meta-llama/Llama-Guard-4-12B<br>F1      | RedHatAI/Llama-Guard-4-12B-FP8-dynamic<br>(this model)<br>F1       | F1 Recovery % | meta-llama/Llama-Guard-4-12B<br>Recall  | RedHatAI/Llama-Guard-4-12B-FP8-dynamic<br>(this model)<br>Recall   | Recall Recovery % |
| ------------------------- | :----------: | :----------: | :-----------: | :----------: | :----------: | :---------------: |
| AART                      | 0.874        | 0.873        | 99.89         | 0.776        | 0.775        | 99.87             |
| AdvBench Behaviors        | 0.964        | 0.965        | 100.1         | 0.931        | 0.933        | 100.21            |
| AdvBench Strings          | 0.83         | 0.832        | 100.24        | 0.709        | 0.713        | 100.56            |
| BeaverTails 330k          | 0.732        | 0.732        | 100           | 0.591        | 0.591        | 100               |
| Bot-Adversarial Dialogue  | 0.513        | 0.506        | 98.64         | 0.376        | 0.371        | 98.67             |
| CatQA                     | 0.932        | 0.933        | 100.11        | 0.873        | 0.875        | 100.23            |
| ConvAbuse                 | 0.241        | 0.258        | 107.05        | 0.148        | 0.164        | 110.81            |
| DecodingTrust Stereotypes | 0.591        | 0.601        | 101.69        | 0.419        | 0.43         | 102.63            |
| DICES 350                 | 0.118        | 0.118        | 100           | 0.063        | 0.063        | 100               |
| DICES 990                 | 0.219        | 0.226        | 103.2         | 0.135        | 0.142        | 105.19            |
| Do Anything Now Questions | 0.746        | 0.74         | 99.2          | 0.595        | 0.587        | 98.66             |
| DoNotAnswer               | 0.546        | 0.544        | 99.63         | 0.376        | 0.374        | 99.47             |
| DynaHate                  | 0.603        | 0.604        | 100.17        | 0.481        | 0.482        | 100.21            |
| HarmEval                  | 0.56         | 0.566        | 101.07        | 0.389        | 0.395        | 101.54            |
| HarmBench Behaviors       | 0.959        | 0.958        | 99.9          | 0.922        | 0.919        | 99.67             |
| HarmfulQ                  | 0.86         | 0.86         | 100           | 0.755        | 0.755        | 100               |
| HarmfulQA Questions       | 0.588        | 0.591        | 100.51        | 0.416        | 0.42         | 100.96            |
| HarmfulQA                 | 0.374        | 0.377        | 100.8         | 0.231        | 0.232        | 100.43            |
| HateCheck                 | 0.782        | 0.785        | 100.38        | 0.667        | 0.671        | 100.6             |
| Hatemoji Check            | 0.625        | 0.627        | 100.32        | 0.474        | 0.478        | 100.84            |
| HEx-PHI                   | 0.966        | 0.964        | 99.79         | 0.933        | 0.93         | 99.68             |
| I-CoNa                    | 0.837        | 0.833        | 99.52         | 0.719        | 0.713        | 99.17             |
| I-Controversial           | 0.596        | 0.621        | 104.19        | 0.425        | 0.45         | 105.88            |
| I-MaliciousInstructions   | 0.824        | 0.81         | 98.3          | 0.7          | 0.68         | 97.14             |
| I-Physical-Safety         | 0.493        | 0.482        | 97.77         | 0.34         | 0.33         | 97.06             |
| JBB Behaviors             | 0.86         | 0.86         | 100           | 0.86         | 0.86         | 100               |
| MaliciousInstruct         | 0.953        | 0.947        | 99.37         | 0.91         | 0.9          | 98.9              |
| MITRE                     | 0.663        | 0.685        | 103.32        | 0.495        | 0.521        | 105.25            |
| NicheHazardQA             | 0.46         | 0.469        | 101.96        | 0.299        | 0.307        | 102.68            |
| OpenAI Moderation Dataset | 0.739        | 0.734        | 99.32         | 0.787        | 0.787        | 100               |
| ProsocialDialog           | 0.427        | 0.432        | 101.17        | 0.276        | 0.28         | 101.45            |
| SafeText                  | 0.372        | 0.38         | 102.15        | 0.254        | 0.262        | 103.15            |
| SimpleSafetyTests         | 0.985        | 0.985        | 100           | 0.97         | 0.97         | 100               |
| StrongREJECT Instructions | 0.91         | 0.91         | 100           | 0.836        | 0.836        | 100               |
| TDCRedTeaming             | 0.947        | 0.947        | 100           | 0.9          | 0.9          | 100               |
| TechHazardQA              | 0.758        | 0.761        | 100.4         | 0.61         | 0.615        | 100.82            |
| Toxic Chat                | 0.433        | 0.425        | 98.15         | 0.519        | 0.519        | 100               |
| ToxiGen                   | 0.46         | 0.47         | 102.17        | 0.315        | 0.325        | 103.17            |
| XSTest                    | 0.834        | 0.833        | 99.88         | 0.78         | 0.775        | 99.36             |
| Average Score             | 0.6711282051 | 0.6729230769 | 100.5220513   | 0.5706410256 | 0.5725641026 | 100.8784615       |


## Model creation

This model is created with `compressed-tensors==0.13.0` and `llmcompressor==0.9.0.1`, and the following LLM-Compressor quantization script:

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py --model_path meta-llama/Llama-Guard-4-12B RedHatAI/Llama-Guard-4-12B-FP8-dynamic --pipeline datafree
```

```python
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Llama4ForConditionalGeneration
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor import oneshot
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationArgs,
    QuantizationType,
    QuantizationStrategy,
)


def main():
    parser = argparse.ArgumentParser(description="Quantize a causal language model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pre-trained model",
    )
    parser.add_argument(
        "--quant_path",
        type=str,
        required=True,
        help="Output path for the quantized model",
    )
    parser.add_argument(
        "--pipeline", #['basic', 'datafree', 'sequential', independent]
        type=str,
        required=True,
    )

    print(f"Loading model from {args.model_path}...")
    model = Llama4ForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8_dynamic",
        ignore=[
            're:.*lm_head',
            're:.*multi_modal_projector',
            're:.*vision_model',
        ]
    )

    print("Applying quantization...")
    oneshot(
        model=model,
        recipe=recipe,
        trust_remote_code_model=True,
        pipeline=args.pipeline,
    )

    model.save_pretrained(args.quant_path, save_compressed=True, skip_compression_stats=True, disable_sparse_compression=True)
    print(f"Quantized model saved to {args.quant_path}")


if __name__ == "__main__":
    main()
```