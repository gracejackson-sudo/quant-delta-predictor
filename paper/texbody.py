"""Reduce a LaTeX source to the prose a checker should scan, and SAY WHAT IT
REMOVED.

WHY THIS IS A MODULE. paper/audit_paper.py used to build its scan body with
four inline re.sub calls, one of which read every `%` as a comment mark. An
escaped `\\%` is a printed percent sign, so every line was cut at its first
printed percentage and 3,162 characters -- 3.05% of the body -- were invisible
to four correct checks from the commit that introduced the checker (db34a29,
2026-09-23) until it was fixed (2aa90e4, 2026-10-11).

A test was then written to assert what each stripper removed, and it held its
own private copies of the patterns. Planting the bug in audit_paper.py left
the test green AND left audit_paper.py reporting zero problems, because the
defect re-hides the evidence that would have fired. A test that validates a
copy of the code under test is not a test of that code.

So the reduction lives here, the shipped gate imports it, the test imports the
same function, and the test's predicates are written independently -- character
walkers with backslash parity, not a second copy of these regexes.

strip_tex(tex) -> (body, spans)
    body  : the text a checker should scan, same length as `tex` with removed
            runs replaced by spaces, so every offset still lines up with the
            source. Offsets mattering is why this does not join fragments.
    spans : [(rule, text, start)] -- every run removed and which rule took it.
"""
from __future__ import annotations

import re

# Each rule is (name, compiled pattern). Order is the order they are applied.
RULES = [
    ("input", re.compile(r"\\input\{[^}]*\}")),
    ("cross-reference",
     re.compile(r"\\(?:label|ref|cite\w*|eprint|url|href)\{[^}]*\}")),
    ("tabular block",
     re.compile(r"\\begin\{tabular\}.*?\\end\{tabular\}", re.S)),
    # An ESCAPED \% is a percent sign, not a comment. The lookbehind is the
    # whole fix; see the module docstring for what it cost.
    ("comment", re.compile(r"(?<!\\)%.*")),
]


def strip_tex(tex: str):
    """-> (body, spans). See the module docstring."""
    body = tex
    spans = []
    for name, pat in RULES:
        out = []
        last = 0
        for m in pat.finditer(body):
            spans.append((name, m.group(0), m.start()))
            out.append(body[last:m.start()])
            out.append(" " * len(m.group(0)))
            last = m.end()
        out.append(body[last:])
        body = "".join(out)
    assert len(body) == len(tex), (
        "strip_tex changed the length of the document; offsets into `body` "
        "would no longer index the source")
    return body, spans
