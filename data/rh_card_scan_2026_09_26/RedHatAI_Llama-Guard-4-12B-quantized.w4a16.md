Evaluations are produced with [https://github.com/neuralmagic/GuardBench](https://github.com/neuralmagic/GuardBench) and vLLM as an inference engine.

Evaluations are obtained with `vllm==0.15.0` and bug fixes from this [PR](https://github.com/vllm-project/vllm/pull/34471).


| Dataset                   |  meta-llama/Llama-Guard-4-12B<br>F1     |  RedHatAI/Llama-Guard-4-12B-quantized.w4a16<br>(this model)<br>F1     | F1 Recovery % | meta-llama/Llama-Guard-4-12B<br>Recall  | RedHatAI/Llama-Guard-4-12B-quantized.w4a16<br>Recall  | Recall<br>Recovery % |
| ------------------------- | :----------: | :----------: | :-----------: | :----------: | :----------: | :---------------: |
| AART                      | 0.874        | 0.865                                                                                                           | 98.97         | 0.776        | 0.761        | 98.07             |
| AdvBench Behaviors        | 0.964        | 0.968                                                                                                           | 100.41        | 0.931        | 0.938        | 100.75            |
| AdvBench Strings          | 0.83         | 0.823                                                                                                           | 99.16         | 0.709        | 0.699        | 98.59             |
| BeaverTails 330k          | 0.732        | 0.727                                                                                                           | 99.32         | 0.591        | 0.584        | 98.82             |
| Bot-Adversarial Dialogue  | 0.513        | 0.499                                                                                                           | 97.27         | 0.376        | 0.361        | 96.01             |
| CatQA                     | 0.932        | 0.927                                                                                                           | 99.46         | 0.873        | 0.864        | 98.97             |
| ConvAbuse                 | 0.241        | 0.248                                                                                                           | 102.9         | 0.148        | 0.156        | 105.41            |
| DecodingTrust Stereotypes | 0.591        | 0.54                                                                                                            | 91.37         | 0.419        | 0.37         | 88.31             |
| DICES 350                 | 0.118        | 0.118                                                                                                           | 100           | 0.063        | 0.063        | 100               |
| DICES 990                 | 0.219        | 0.226                                                                                                           | 103.2         | 0.135        | 0.135        | 100               |
| Do Anything Now Questions | 0.746        | 0.74                                                                                                            | 99.2          | 0.595        | 0.587        | 98.66             |
| DoNotAnswer               | 0.546        | 0.539                                                                                                           | 98.72         | 0.376        | 0.368        | 97.87             |
| DynaHate                  | 0.603        | 0.587                                                                                                           | 97.35         | 0.481        | 0.459        | 95.43             |
| HarmEval                  | 0.56         | 0.571                                                                                                           | 101.96        | 0.389        | 0.4          | 102.83            |
| HarmBench Behaviors       | 0.959        | 0.954                                                                                                           | 99.48         | 0.922        | 0.912        | 98.92             |
| HarmfulQ                  | 0.86         | 0.857                                                                                                           | 99.65         | 0.755        | 0.75         | 99.34             |
| HarmfulQA Questions       | 0.588        | 0.583                                                                                                           | 99.15         | 0.416        | 0.411        | 98.8              |
| HarmfulQA                 | 0.374        | 0.347                                                                                                           | 92.78         | 0.231        | 0.21         | 90.91             |
| HateCheck                 | 0.782        | 0.77                                                                                                            | 98.47         | 0.667        | 0.649        | 97.3              |
| Hatemoji Check            | 0.625        | 0.609                                                                                                           | 97.44         | 0.474        | 0.457        | 96.41             |
| HEx-PHI                   | 0.966        | 0.955                                                                                                           | 98.86         | 0.933        | 0.913        | 97.86             |
| I-CoNa                    | 0.837        | 0.813                                                                                                           | 97.13         | 0.719        | 0.685        | 95.27             |
| I-Controversial           | 0.596        | 0.621                                                                                                           | 104.19        | 0.425        | 0.45         | 105.88            |
| I-MaliciousInstructions   | 0.824        | 0.817                                                                                                           | 99.15         | 0.7          | 0.69         | 98.57             |
| I-Physical-Safety         | 0.493        | 0.482                                                                                                           | 97.77         | 0.34         | 0.33         | 97.06             |
| JBB Behaviors             | 0.86         | 0.86                                                                                                            | 100           | 0.86         | 0.86         | 100               |
| MaliciousInstruct         | 0.953        | 0.958                                                                                                           | 100.52        | 0.91         | 0.92         | 101.1             |
| MITRE                     | 0.663        | 0.649                                                                                                           | 97.89         | 0.495        | 0.48         | 96.97             |
| NicheHazardQA             | 0.46         | 0.469                                                                                                           | 101.96        | 0.299        | 0.307        | 102.68            |
| OpenAI Moderation Dataset | 0.739        | 0.741                                                                                                           | 100.27        | 0.787        | 0.78         | 99.11             |
| ProsocialDialog           | 0.427        | 0.414                                                                                                           | 96.96         | 0.276        | 0.265        | 96.01             |
| SafeText                  | 0.372        | 0.368                                                                                                           | 98.92         | 0.254        | 0.246        | 96.85             |
| SimpleSafetyTests         | 0.985        | 0.99                                                                                                            | 100.51        | 0.97         | 0.98         | 101.03            |
| StrongREJECT Instructions | 0.91         | 0.902                                                                                                           | 99.12         | 0.836        | 0.822        | 98.33             |
| TDCRedTeaming             | 0.947        | 0.958                                                                                                           | 101.16        | 0.9          | 0.92         | 102.22            |
| TechHazardQA              | 0.758        | 0.75                                                                                                            | 98.94         | 0.61         | 0.6          | 98.36             |
| Toxic Chat                | 0.433        | 0.433                                                                                                           | 100           | 0.519        | 0.508        | 97.88             |
| ToxiGen                   | 0.46         | 0.444                                                                                                           | 96.52         | 0.315        | 0.3          | 95.24             |
| XSTest                    | 0.834        | 0.832                                                                                                           | 99.76         | 0.78         | 0.765        | 98.08             |
| Average Score             | 0.6711282051 | 0.6654871795                                                                                                    | 99.12538462   | 0.5706410256 | 0.5629487179 | 98.45897436       |


## Model creation

This model is created with `compressed-tensors==0.13.0` and `llmcompressor==0.9.0.1`, and the following LLM-Compressor quantization script:

```bash
CUDA_VISIBLE_DEVICES=0 python quantize.py --model_path meta-llama/Llama-Guard-4-12B --quant_path RedHatAI/Llama-Guard-4-12B-quantized.w4a16 --group_size 128 --calib_size 1024 --dampening_frac 0.01 --observer minmax --sym True --actorder False --pipeline independent
```

```python
from datasets import load_dataset
from transformers import AutoProcessor, Llama4ForConditionalGeneration
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
import argparse
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs, QuantizationType, QuantizationStrategy

def parse_actorder(value):
    # Interpret the input value for --actorder
    if value.lower() == "false":
        return False
    elif value.lower() == "group":
        return "group"
    elif value.lower() == "weight":
        return "weight"
    else:
        raise argparse.ArgumentTypeError("Invalid value for --actorder. Use 'group', 'weight', or 'False'.")

def parse_sym(value):
    if value.lower() == "false":
        return False
    elif value.lower() == "true":
        return True
    else:
        raise argparse.ArgumentTypeError(f"Invalid value for --sym. Use false or true, but got {value}")

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--quant_path', type=str, required=True)
parser.add_argument('--group_size', type=int, required=True)
parser.add_argument('--calib_size', type=int, required=True)
parser.add_argument('--dampening_frac', type=float, required=True)
parser.add_argument('--observer', type=str, required=True) # mse or minmax
parser.add_argument('--sym', type=parse_sym, required=True) # true or false
parser.add_argument('--actorder', type=parse_actorder, required=True) # group or weight or false
parser.add_argument('--pipeline', type=str, default="basic") # ['basic', 'datafree', 'sequential', independent]

args = parser.parse_args()

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
        num_bits=4,
        type=QuantizationType.INT,
        symmetric=args.sym,
        group_size=args.group_size,
        strategy=QuantizationStrategy.GROUP,
        observer=args.observer,
        actorder=args.actorder
    ),
    input_activations=None,
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
    max_seq_length=4096,
    pipeline=args.pipeline,
)

SAVE_DIR = args.quant_path
model.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}. Please manually copy other files like tokenizer, proprocessors, etc.")
```

