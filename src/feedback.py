"""
Feedback loop: turn real usage into data.

Why this exists, stated plainly: this tool's calibration is static and drawn
only from published model cards. Today's research established that published
data alone cannot answer the questions users most need answered -- the
published corpus contains only recipes that worked, and two (scheme, size)
cells have too little data to answer at all. Real user-reported results are
the fastest way to close that gap.

WHAT THIS CURRENTLY IS: a one-line prompt pointing at a collection form, plus
a local append-only log of what people asked about. Nothing is transmitted
anywhere. A human reads the form and re-runs the pipeline.

WHAT IT IS NOT (yet): an automated ingestion pipeline. Submitted results do
not flow into the envelope on their own; someone validates them with
src/adversarial_schema.py and re-runs the build.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CONFIG = os.path.join(ROOT, "feedback_config.json")
REQUEST_LOG = os.path.join(ROOT, "out", "requests.log")

# Until a real form exists this stays an obvious placeholder. It must never
# render as a working link that goes nowhere.
DEFAULT = {
    "contact": None,
    "form_url": None,
    "note": "Set contact or form_url in feedback_config.json to enable the "
            "report-back prompt.",
}


def config():
    try:
        with open(CONFIG) as f:
            c = json.load(f)
        return {**DEFAULT, **c}
    except (OSError, ValueError):
        return dict(DEFAULT)


_PLACEHOLDER_MARKERS = ("replace_me", "example.com", "your_", "changeme",
                        "<", "todo")


def _is_placeholder(value):
    """Guard against the example config being copied without editing.

    A truthy-but-fake URL would print a dead link to every user, which is
    worse than printing nothing at all.
    """
    return any(m in value.lower() for m in _PLACEHOLDER_MARKERS)


def destination():
    """-> (text, configured) where configured is False for the placeholder."""
    c = config()
    for key in ("form_url", "contact"):
        v = (c.get(key) or "").strip()
        if v and not _is_placeholder(v):
            return v, True
    return "[reporting address not yet configured]", False


def log_request(schemes, band=None, moe=False, refused=None):
    """
    Append-only local record of what was asked about. This is the
    prioritisation list for future GPU time: real demand, not guesswork.
    Stored locally only; nothing leaves the machine.
    """
    try:
        os.makedirs(os.path.dirname(REQUEST_LOG), exist_ok=True)
        new = not os.path.exists(REQUEST_LOG)
        with open(REQUEST_LOG, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp_utc", "schemes", "size_band", "moe",
                            "refused_cells"])
            w.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "|".join(sorted(schemes)),
                band or "",
                int(bool(moe)),
                "|".join(sorted(refused or [])),
            ])
    except OSError:
        pass          # logging must never break the tool


def prompt_line(width=78):
    """One low-friction line, printed after a normal result."""
    dest, ok = destination()
    if not ok:
        return []
    return _wrap(
        f"Ran your own eval on this config? Sending the real number back "
        f"improves this for everyone: {dest}", width)


def refused_prompt(cells, width=78):
    """A stronger, specific ask where we genuinely could not answer."""
    dest, ok = destination()
    lines = _wrap(
        "We do not have enough data to answer this confidently. If you test "
        "this yourself, it is exactly the result that would help most - "
        "these cells are the reason the tool refuses rather than guesses.",
        width)
    if ok:
        lines += _wrap(f"Report it here: {dest}", width)
    return lines


FORM_FIELDS = [
    ("quantization_scheme", "required",
     "w4a16 / w8a8 / w8a16 / fp8 / fp8-dynamic / nvfp4 / other"),
    ("model_size_params_b", "required", "e.g. 1.5 - a number in billions"),
    ("model_family_or_name", "optional",
     "e.g. Qwen2.5, Llama-3.1. Leave blank if you cannot share it"),
    ("benchmark", "required",
     "mmlu / arc_challenge / hellaswag / gsm8k / winogrande / truthfulqa / "
     "other"),
    ("accuracy_before", "required", "unquantized score, 0-100 scale"),
    ("accuracy_after", "required", "quantized score, 0-100 scale"),
    ("eval_harness_and_shots", "optional",
     "e.g. lm-eval-harness, MMLU 5-shot. Needed to compare like with like"),
    ("quantization_tool_and_config", "optional",
     "e.g. llm-compressor GPTQ, 512x2048 calibration, group 128"),
    ("anything_unusual", "optional",
     "especially: did anything go wrong, or was this a config you would not "
     "recommend?"),
    ("contact_optional", "optional", "only if you want a reply"),
]


def form_spec():
    """The exact fields a collection form needs, for manual setup."""
    return FORM_FIELDS


def _wrap(t, width):
    words, line, out = t.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > width:
            out.append(line)
            line = w
        else:
            line = (line + " " + w).strip()
    if line:
        out.append(line)
    return out


if __name__ == "__main__":
    print("Collection form fields\n" + "=" * 60)
    for name, req, help_ in FORM_FIELDS:
        print(f"  {name}  [{req}]\n      {help_}")
    dest, ok = destination()
    print(f"\ndestination: {dest}  (configured: {ok})")
    print(f"request log: {REQUEST_LOG}")
