Evaluations are produced with [https://github.com/neuralmagic/GuardBench](https://github.com/neuralmagic/GuardBench) and vLLM as an inference engine.

Evaluations are obtained with `vllm==0.15.0` and bug fixes from this [PR](https://github.com/vllm-project/vllm/pull/34243).
If you are running `vllm > 0.15.0`, you will likely have the bug fixes already applied as the PR landed on main.


| Dataset                   |  meta-llama/Llama-Guard-4-12B<br>F1     |  RedHatAI/Llama-Guard-4-12B-quantized.w8a8<br>(this model)<br>F1     | F1 Recovery % | meta-llama/Llama-Guard-4-12B<br>Recall  | RedHatAI/Llama-Guard-4-12B-quantized.w8a8<br>Recall  | Recall<br>Recovery % |
| ------------------------- | :----------: | :----------: | :-----------: | :----------: | :----------: | :---------------: |
| AART                      | 0.874        | 0.869        | 99.43         | 0.776        | 0.769        | 99.1              |
| AdvBench Behaviors        | 0.964        | 0.966        | 100.21        | 0.931        | 0.935        | 100.43            |
| AdvBench Strings          | 0.83         | 0.824        | 99.28         | 0.709        | 0.7          | 98.73             |
| BeaverTails 330k          | 0.732        | 0.728        | 99.45         | 0.591        | 0.585        | 98.98             |
| Bot-Adversarial Dialogue  | 0.513        | 0.504        | 98.25         | 0.376        | 0.368        | 97.87             |
| CatQA                     | 0.932        | 0.928        | 99.57         | 0.873        | 0.865        | 99.08             |
| ConvAbuse                 | 0.241        | 0.225        | 93.36         | 0.148        | 0.141        | 95.27             |
| DecodingTrust Stereotypes | 0.591        | 0.596        | 100.85        | 0.419        | 0.424        | 101.19            |
| DICES 350                 | 0.118        | 0.118        | 100           | 0.063        | 0.063        | 100               |
| DICES 990                 | 0.219        | 0.219        | 100           | 0.135        | 0.135        | 100               |
| Do Anything Now Questions | 0.746        | 0.736        | 98.66         | 0.595        | 0.582        | 97.82             |
| DoNotAnswer               | 0.546        | 0.543        | 99.45         | 0.376        | 0.373        | 99.2              |
| DynaHate                  | 0.603        | 0.604        | 100.17        | 0.481        | 0.48         | 99.79             |
| HarmEval                  | 0.56         | 0.573        | 102.32        | 0.389        | 0.402        | 103.34            |
| HarmBench Behaviors       | 0.959        | 0.954        | 99.48         | 0.922        | 0.912        | 98.92             |
| HarmfulQ                  | 0.86         | 0.86         | 100           | 0.755        | 0.755        | 100               |
| HarmfulQA Questions       | 0.588        | 0.584        | 99.32         | 0.416        | 0.412        | 99.04             |
| HarmfulQA                 | 0.374        | 0.355        | 94.92         | 0.231        | 0.217        | 93.94             |
| HateCheck                 | 0.782        | 0.784        | 100.26        | 0.667        | 0.668        | 100.15            |
| Hatemoji Check            | 0.625        | 0.62         | 99.2          | 0.474        | 0.468        | 98.73             |
| HEx-PHI                   | 0.966        | 0.964        | 99.79         | 0.933        | 0.93         | 99.68             |
| I-CoNa                    | 0.837        | 0.829        | 99.04         | 0.719        | 0.708        | 98.47             |
| I-Controversial           | 0.596        | 0.621        | 104.19        | 0.425        | 0.45         | 105.88            |
| I-MaliciousInstructions   | 0.824        | 0.83         | 100.73        | 0.7          | 0.71         | 101.43            |
| I-Physical-Safety         | 0.493        | 0.459        | 93.1          | 0.34         | 0.31         | 91.18             |
| JBB Behaviors             | 0.86         | 0.864        | 100.47        | 0.86         | 0.86         | 100               |
| MaliciousInstruct         | 0.953        | 0.958        | 100.52        | 0.91         | 0.92         | 101.1             |
| MITRE                     | 0.663        | 0.668        | 100.75        | 0.495        | 0.502        | 101.41            |
| NicheHazardQA             | 0.46         | 0.454        | 98.7          | 0.299        | 0.294        | 98.33             |
| OpenAI Moderation Dataset | 0.739        | 0.735        | 99.46         | 0.787        | 0.782        | 99.36             |
| ProsocialDialog           | 0.427        | 0.42         | 98.36         | 0.276        | 0.27         | 97.83             |
| SafeText                  | 0.372        | 0.361        | 97.04         | 0.254        | 0.243        | 95.67             |
| SimpleSafetyTests         | 0.985        | 0.985        | 100           | 0.97         | 0.97         | 100               |
| StrongREJECT Instructions | 0.91         | 0.905        | 99.45         | 0.836        | 0.826        | 98.8              |
| TDCRedTeaming             | 0.947        | 0.947        | 100           | 0.9          | 0.9          | 100               |
| TechHazardQA              | 0.758        | 0.755        | 99.6          | 0.61         | 0.606        | 99.34             |
| Toxic Chat                | 0.433        | 0.436        | 100.69        | 0.519        | 0.525        | 101.16            |
| ToxiGen                   | 0.46         | 0.465        | 101.09        | 0.315        | 0.32         | 101.59            |
| XSTest                    | 0.834        | 0.836        | 100.24        | 0.78         | 0.78         | 100               |
| Average Score             | 0.6711282051 | 0.6687692308 | 99.42051282   | 0.5706410256 | 0.5682051282 | 99.30282051       |


## Model creation

This model is created with `compressed-tensors==0.13.0` and `llmcompressor==0.9.0.1`, and the following LLM-Compressor quantization script:

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py --model_path meta-llama/Llama-Guard-4-12B --quant_path RedHatAI/Llama-Guard-4-12B-quantized.w8a8 --calib_size 512 --dampening_frac 0.03 --pipeline independent
```

```python
from datasets import load_dataset
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
import argparse
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--quant_path', type=str, required=True)
parser.add_argument('--calib_size', type=int, required=True)
parser.add_argument('--dampening_frac', type=float, required=True)
parser.add_argument('--pipeline', type=str, default="basic") #['basic', 'datafree', 'sequential', independent]

args = parser.parse_args()
print(f"[DEBUGGING ARGS] {args}")

model = Llama4ForConditionalGeneration.from_pretrained(
    args.model_path,
    torch_dtype="auto",
    trust_remote_code=True,
)
processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

def preprocess_fn(example):
    # prepare for multimodal processor
    for msg in example["messages"]:
        msg["content"] = [{'type': 'text', 'text': msg['content']}]

    return {"text": processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False)}

ds = load_dataset("neuralmagic/LLM_compression_calibration", split="train")
ds = ds.map(preprocess_fn)

print(f"================================================================================")
print(f"[For debugging] Calibration data sample is:\n{repr(ds[0]['text'])}")
print(f"================================================================================")

quant_scheme = QuantizationScheme(
    targets=["Linear"],
    weights=QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        symmetric=True,
        strategy=QuantizationStrategy.CHANNEL,
    ),
    input_activations=QuantizationArgs(
        dynamic=True,
        num_bits=8,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        type=QuantizationType.INT,
    ),
    output_activations=None,
)

recipe = [
    GPTQModifier(
        targets=["Linear"],
        ignore=[
            "re:.*lm_head",
            "re:.*multi_modal_projector",
            "re:.*vision_model",
        ],
        dampening_frac=args.dampening_frac,
        config_groups={"group_0": quant_scheme},
    )
]
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    num_calibration_samples=args.calib_size,
    max_seq_length=2048,
    pipeline=args.pipeline,
)

SAVE_DIR = args.quant_path
model.save_pretrained(SAVE_DIR)
```