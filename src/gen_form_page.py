"""Generate feedback_form.html from the CLI's own field spec.

The page is a plain HTML form using Formspree's documented pattern (action +
method="POST", named inputs), so submitting navigates to Formspree's thanks
page. No JavaScript, no CORS dependency, no styling framework.

Generated rather than hand-written for the same reason the docs are: the
fields must not drift from what `rank.py --form-fields` prints.
"""
from __future__ import annotations

import html
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)

import feedback  # noqa: E402

OUT = os.path.join(ROOT, "feedback_form.html")
# Long free-text answers get a textarea rather than a single-line input.
TEXTAREA = {"anything_unusual", "quantization_tool_and_config",
            "open_feedback"}
# Field names are machine-readable; these read better to a human.
LABELS = {
    "model_size_params_b": "model size (billions of parameters)",
    "contact_optional": "contact (only if you want a reply)",
    "eval_harness_and_shots": "eval harness and shot count",
    "quantization_tool_and_config": "quantization tool and config",
    "accuracy_before": "accuracy before quantization",
    "accuracy_after": "accuracy after quantization",
    "model_family_or_name": "model family or name",
    "anything_unusual": "anything unusual",
    "name_and_role": "name and role",
    "open_feedback": "your feedback",
}


def field_html(name, req, hint):
    label = LABELS.get(name, name.replace("_", " "))
    required = ""      # see module docstring: never hard-block submission
    star = (' <span class="req">needed if reporting a result</span>'
            if req == "required" else "")
    out = [f'<div class="f"><label for="{name}">{html.escape(label)}{star}'
           f'</label>', f'<p class="h">{html.escape(hint)}</p>']
    # A hint written as "a / b / c" enumerates the allowed values.
    opts = [o.strip() for o in hint.split("/")] if hint.count("/") >= 2 else []
    if opts:
        out.append(f'<select id="{name}" name="{name}"{required}>')
        out.append('<option value="">-- choose --</option>')
        for o in opts:
            out.append(f'<option value="{html.escape(o)}">'
                       f'{html.escape(o)}</option>')
        out.append("</select>")
    elif name in TEXTAREA:
        out.append(f'<textarea id="{name}" name="{name}" rows="3"'
                   f'{required}></textarea>')
    else:
        out.append(f'<input id="{name}" name="{name}" type="text"{required}>')
    out.append("</div>")
    return "\n".join(out)


def main():
    # The form ACTION must be the raw POST endpoint, never the human-facing
    # page URL that the CLI prints -- pointing the form at itself would make
    # every submission a no-op.
    endpoint = (feedback.config().get("post_endpoint") or "").strip()
    if not endpoint:
        raise SystemExit("set post_endpoint in feedback_config.json")
    spec = feedback.form_spec()
    fields = "\n".join(field_html(n, r, h) for n, r, h in spec)
    who = field_html("name_and_role",
                     "optional",
                     'Optional. Anything that fits - "student ML '
                     'researcher at Berkeley", "ML engineer", "just '
                     'tinkering". Applies to either section.')
    openfb = field_html("open_feedback", "optional",
                        "What was confusing, what you expected it to do, "
                        "whether it was worth your time. Criticism is more "
                        "useful than praise here.")
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quantization risk tool - feedback</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font-family: -apple-system, system-ui, sans-serif; line-height: 1.5;
  max-width: 620px; margin: 0 auto; padding: 24px 16px 64px; }}
h1 {{ font-size: 1.35rem; margin-bottom: .25rem; }}
h2 {{ font-size: 1.05rem; margin: 26px 0 2px; }}
hr {{ border: 0; border-top: 1px solid #ddd; margin: 26px 0 0; }}
.lede {{ color: #555; margin-top: 0; }}
.f {{ margin: 18px 0; }}
label {{ font-weight: 600; display: block; }}
label::first-letter {{ text-transform: uppercase; }}
.h {{ color: #666; font-size: .85rem; margin: 2px 0 6px; }}
.req {{ color: #b00; font-weight: 400; font-size: .78rem; }}
input, select, textarea {{ width: 100%; padding: 7px 8px; font-size: 1rem;
  border: 1px solid #999; border-radius: 4px; box-sizing: border-box;
  background: transparent; color: inherit; font-family: inherit; }}
button {{ font-size: 1rem; padding: 10px 18px; border-radius: 4px;
  border: 1px solid #333; cursor: pointer; }}
.note {{ font-size: .85rem; color: #666; border-top: 1px solid #ddd;
  margin-top: 28px; padding-top: 12px; }}
@media (prefers-color-scheme: dark) {{
  .lede, .h, .note {{ color: #aaa; }}
  .note {{ border-top-color: #444; }}
  input, select, textarea {{ border-color: #666; }}
  button {{ border-color: #aaa; }}
}}
</style>
</head>
<body>
<h1>Quantization risk tool &mdash; feedback</h1>
<p class="lede">This is optional and there is no wrong amount to fill in. A
single number is useful; so is a sentence saying the tool was not much use.</p>

<form action="{html.escape(endpoint)}" method="POST">
{who}
<hr>
<h2>1. A measured result, if you have one</h2>
<p class="lede">Only if you happened to run an eval. Skip the whole section
otherwise.</p>
{fields}
<hr>
<h2>2. Open feedback</h2>
<p class="lede">Send this on its own if you like &mdash; nothing above is
required.</p>
{openfb}
<button type="submit">Send</button>
</form>

<p class="note">Submissions are reviewed by a human before anything enters the
dataset; nothing is ingested automatically. Only what you type here is sent
&mdash; there is no tracking on this page or in the tool.</p>
</body>
</html>
"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"wrote {OUT} ({len(doc.splitlines())} lines) -> {endpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
