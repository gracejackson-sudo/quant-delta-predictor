"""
Minimal public-facing tool, working prototype.

    ./.venv/bin/python src/cli.py w4a16
    ./.venv/bin/python src/cli.py --list

Input: a quantization scheme name. Output: the calibrated 90% interval, the
historical coverage rate, and the limitations that apply to that scheme.

It deliberately does NOT accept a model name, because the artifact has no
per-model signal and pretending otherwise would be the dishonest version.
"""
from __future__ import annotations

import json
import os
import sys

ART = os.path.join(os.path.dirname(__file__), "..", "out",
                   "scheme_envelope.json")
ALIASES = {
    "w4a16": "w4a16", "int4": "w4a16", "gptq": "w4a16", "4bit": "w4a16",
    "w8a8": "w8a8_int", "w8a8_int": "w8a8_int", "int8": "w8a8_int",
    "smoothquant": "w8a8_int",
    "w8a16": "w8a16",
    "fp8": "fp8", "fp8_static": "fp8",
    "fp8_dynamic": "fp8_dynamic", "fp8-dynamic": "fp8_dynamic",
    "nvfp4": "nvfp4", "fp4": "nvfp4",
}


def main(argv):
    if not os.path.exists(ART):
        print("artifact missing -- run: python src/build_envelope.py")
        return 1
    art = json.load(open(ART))
    schemes = art["schemes"]

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        print("\nschemes: " + ", ".join(sorted(schemes)))
        return 0

    if argv[0] == "--list":
        print(f"{'scheme':<14}{'mean':>8}{'90% interval':>18}"
              f"{'loss-side':>10}{'n':>6}{'families':>10}")
        for s in sorted(schemes, key=lambda k: -schemes[k]["mean_delta_pp"]):
            v = schemes[s]
            c = v["coverage"]["one_sided"]
            print(f"{s:<14}{v['mean_delta_pp']:>+8.2f}"
                  f"{str(v['interval_90_pp']):>18}"
                  f"{f'{100*c:.0f}%':>10}"
                  f"{v['n_observations']:>6}{v['n_families']:>10}")
        return 0

    key = ALIASES.get(argv[0].strip().lower().replace("-", "_"))
    if key is None or key not in schemes:
        print(f"unknown scheme '{argv[0]}'. known: "
              f"{', '.join(sorted(set(ALIASES)))}")
        return 2

    v = schemes[key]
    lo, hi = v["interval_90_pp"]
    cov = v["coverage"]
    print(f"\n  {v['label']}")
    print(f"  {'-' * len(v['label'])}")
    print(f"  expected accuracy change   {v['mean_delta_pp']:+.2f} pp")
    print(f"  90% interval               [{lo:+.2f}, {hi:+.2f}] pp")
    print(f"  worst ever observed        {v['worst_observed_delta_pp']:+.2f} "
          f"pp")
    print(f"\n  based on                   {v['n_observations']} published "
          f"evaluations")
    print(f"                             {v['n_checkpoints']} checkpoints, "
          f"{v['n_families']} model families")
    lo, hi = cov["bootstrap_90_pct"]
    print(f"  measured coverage          {100*cov['one_sided']:.0f}% of held-out results at or above the "
          f"lower bound (bootstrap 90% [{lo:.0f}, {hi:.0f}]%),")
    print(f"                             {100*cov['two_sided']:.0f}% inside the full interval "
          f"(target 90%); {cov['rows']} rows, {cov['checkpoints']} checkpoints")
    print(f"  coverage verdict           {cov['state'].replace('_', ' ')}")
    for b_, bv in cov["by_size_band"].items():
        print(f"    size {b_:<6}             {bv['state'].replace('_', ' ')}"
              + (f" ({bv['checkpoints']} checkpoint{'s' if bv['checkpoints'] != 1 else ''})"
                 if bv.get('checkpoints') else ""))
    print(f"\n  can this rule out 'no change'?  "
          f"{'yes' if v['excludes_zero'] else 'NO -- zero is inside the interval'}")
    if v.get("warning"):
        print(f"\n  !! {v['warning']}")
    print("\n  This is a historical baseline across published checkpoints, not")
    print("  a prediction about your model. It does not use model size, family")
    print("  or benchmark. Read the limitations:")
    for L in art["known_limitations"][:3]:
        print(f"    - {L}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
