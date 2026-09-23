"""
Render the project's markdown docs to PDF.

Pure Python (xhtml2pdf + reportlab) so it needs no Homebrew/system libraries.
Fonts are registered from macOS system TTFs so that arrows, +/-, <=, Greek and
box characters in the docs render instead of coming out as black squares.

Output: ~/Desktop/quant-delta-docs/*.pdf  (one per doc, plus a combined file)
"""
from __future__ import annotations

import os
import re
import sys

import markdown
from xhtml2pdf import pisa
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont


def _register_fonts():
    """Register faces with reportlab directly.

    xhtml2pdf silently ignores an @font-face it cannot resolve and falls back
    to base-14 Courier, which has no glyph for the arrows, Greek and math
    signs these docs use. Registering by name makes the failure loud instead.
    """
    faces = [("Body", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
             ("Mono", "/System/Library/Fonts/Supplemental/Andale Mono.ttf")]
    for name, path in faces:
        if not os.path.exists(path):
            raise SystemExit(f"missing font {path}")
        pdfmetrics.registerFont(RLTTFont(name, path))
        pdfmetrics.registerFontFamily(name, normal=name, bold=name,
                                      italic=name, boldItalic=name)

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.expanduser("~/Desktop/quant-delta-docs")

DOCS = [
    ("RESEARCH.md", "2 - Research (written before building)"),
    ("FINDINGS.md", "3 - Full Findings"),
    ("ADVERSARIAL_AUDIT.md", "4 - Adversarial Audit"),
    ("SCOPE.md", "5 - Product Scope"),
    ("ACCOUNTING.md", "6 - Data Accounting"),
    ("PROVENANCE.md", "7 - Change Provenance"),
    ("TOOL_SUMMARY.md", "8 - The Tool, Honestly"),
    ("NEGATIVE_RESULT.md", "9 - Measured Left Tail"),
    ("BIAS_CORRECTION.md", "10 - Selection Bias"),
    ("RANKING.md", "11 - Scheme Ranking"),
]

F = "/System/Library/Fonts/Supplemental"
FONTS = f"""
@font-face {{ font-family: "Body"; src: url("{F}/Arial Unicode.ttf");
              font-weight: normal; font-style: normal; }}
@font-face {{ font-family: "Body"; src: url("{F}/Arial Bold.ttf");
              font-weight: bold; font-style: normal; }}
@font-face {{ font-family: "Mono"; src: url("{F}/Andale Mono.ttf");
              font-weight: normal; font-style: normal; }}
@font-face {{ font-family: "Mono"; src: url("{F}/Andale Mono.ttf");
              font-weight: bold; font-style: normal; }}
"""

CSS = FONTS + """
@page { size: a4 portrait; margin: 15mm 14mm 17mm 14mm;
        @frame footer { -pdf-frame-content: footer;
                        bottom: 8mm; left: 14mm; right: 14mm; height: 9mm; } }
body { font-family: "Body"; font-size: 9.6pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 16pt; color: #111; margin: 0 0 4pt 0;
     border-bottom: 1.4pt solid #222; padding-bottom: 3pt; }
h2 { font-size: 12pt; color: #111; margin: 14pt 0 4pt 0; }
h3 { font-size: 10.4pt; color: #222; margin: 10pt 0 3pt 0; }
p  { margin: 0 0 6pt 0; }
li { margin: 0 0 3pt 0; }
code { font-family: "Mono"; font-size: 8.4pt; background-color: #f2f2f5; }
pre  { font-family: "Mono"; font-size: 7.8pt; background-color: #f6f6f9;
       border: 0.6pt solid #e0e0e6; padding: 6pt; margin: 6pt 0;
       line-height: 1.3; }
table { border-collapse: collapse; width: 100%;
        margin: 7pt 0; font-size: 7.9pt; }
th { background-color: #eeeef2; border: 0.5pt solid #cfcfd6;
     padding: 2.5pt 3.5pt; font-weight: bold; text-align: left;
     -pdf-word-wrap: CJK; }
td { border: 0.5pt solid #cfcfd6; padding: 2.5pt 3.5pt; text-align: left;
     -pdf-word-wrap: CJK; }
blockquote { border-left: 2pt solid #c0c0cc; background-color: #fafafc;
             margin: 6pt 0; padding: 4pt 8pt; color: #333; }
a { color: #0b5fc7; text-decoration: none; }
hr { border: 0.5pt solid #dddde3; margin: 10pt 0; }
.hdr { font-size: 7.8pt; color: #666; margin-bottom: 10pt;
       border-bottom: 0.5pt solid #e8e8ee; padding-bottom: 4pt; }
.footer { font-size: 7.5pt; color: #999; text-align: center; }
.cover-t { font-size: 22pt; font-weight: bold; color: #111;
           margin: 150pt 0 0 0; text-align: center; }
.cover-s { font-size: 11pt; color: #555; text-align: center; margin-top: 8pt; }
.cover-m { font-size: 9pt; color: #777; text-align: center;
           margin-top: 34pt; line-height: 1.8; }
"""

HDR = ("Grace Jackson &middot; Quantization accuracy-delta predictor "
       "&middot; 22 September 2026")
FOOT = ('<div id="footer" class="footer">Grace Jackson &middot; '
        'Quantization accuracy-delta predictor &middot; '
        '<pdf:pagenumber> / <pdf:pagecount></div>')

MD_EXT = ["tables", "fenced_code", "sane_lists"]

# The mono face reportlab falls back to (base-14 Courier) has no glyph for the
# arrows, Greek and math signs these docs use, so they render as boxes. Prose
# uses Arial Unicode and is fine, so this rewrites ONLY inside <code>/<pre>.
CODE_SAFE = {
    "\u2265": "&gt;=", "\u2264": "&lt;=", "\u2192": "-&gt;", "\u2190": "&lt;-",
    "\u2212": "-", "\u00b1": "+/-", "\u2248": "~=", "\u2260": "!=",
    "\u03b1": "alpha", "\u03c0": "pi", "\u0177": "y-hat", "\u0302": "",
    "\u00d7": "x", "\u2011": "-", "\u2013": "-", "\u2014": "--",
}

_CODE_RE = re.compile(r"(<(code|pre)\b[^>]*>)(.*?)(</\2>)", re.S)


def _asciify_code(html):
    def fix(m):
        inner = m.group(3)
        for bad, good in CODE_SAFE.items():
            inner = inner.replace(bad, good)
        return m.group(1) + inner + m.group(4)
    return _CODE_RE.sub(fix, html)


def to_html(path):
    html = markdown.markdown(open(path, encoding="utf-8").read(),
                             extensions=MD_EXT)
    return _asciify_code(html)


def render(html_body, out_path):
    doc = (f'<html><head><meta charset="utf-8"><style>{CSS}</style></head>'
           f"<body>{FOOT}{html_body}</body></html>")
    with open(out_path, "wb") as f:
        res = pisa.CreatePDF(doc, dest=f, encoding="utf-8")
    return not res.err


def main():
    _register_fonts()
    os.makedirs(DST, exist_ok=True)
    made, failed = [], []

    for fn, title in DOCS:
        p = os.path.join(SRC, fn)
        if not os.path.exists(p):
            continue
        body = to_html(p)
        html = f'<div class="hdr">{HDR} &middot; <b>{title}</b></div>{body}'
        out = os.path.join(DST, f"{title}.pdf")
        if render(html, out):
            made.append(out)
            print(f"  ok   {os.path.basename(out)} "
                  f"({os.path.getsize(out)//1024}KB)")
        else:
            failed.append(title)
            print(f"  FAIL {title}")

    # combined
    parts = [
        '<div class="cover-t">Quantization Accuracy-Delta Predictor</div>'
        '<div class="cover-s">Feasibility build &mdash; research, '
        'product and audits</div>'
        '<div class="cover-m"><b>Grace Jackson</b><br/>'
        '21&ndash;22 September 2026<br/>'
        '850 evaluations &middot; 102 model cards &middot; 113 tests '
        '&middot; 4 audit passes</div>'
    ]
    for fn, title in DOCS:
        p = os.path.join(SRC, fn)
        if not os.path.exists(p):
            continue
        parts.append(f"<pdf:nextpage />{to_html(p)}")
    out = os.path.join(DST, "0 - ALL DOCS (combined).pdf")
    if render("".join(parts), out):
        made.append(out)
        print(f"  ok   {os.path.basename(out)} "
              f"({os.path.getsize(out)//1024}KB)")
    else:
        failed.append("combined")
        print("  FAIL combined")

    print(f"\n{len(made)} PDFs written to {DST}")
    if failed:
        print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
