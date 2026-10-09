"""Build the anonymized supplementary-material zip for TMLR review.

Two jobs, run in this order:

1. Copy the working tree into a build directory, EXCLUDING every file that
   carries an identifying string (a script that hard-codes a name, the
   named long-form paper, the feedback form, etc.) and REWRITING the small
   number of files where the identity is a specific line rather than the
   whole file (LICENSE holder, git clone URL in README, form URL in the
   generated docs, byline in RESEARCH.md).

2. Run a build check that scans the whole build tree for any identity
   string that survived. If any string is found, print each hit with a
   line reference and EXIT NONZERO. The build zip is not written when the
   check fails; the caller has to fix the leak or extend the redaction map
   and rerun.

Every source of identity in this repo has been enumerated once by hand
into IDENT_STRINGS below. A future name leak means either:
   (a) a file that carries a new identifier was added and needs to be
       covered by one of DROP_FILES / RENAMES / TEXT_REPLACEMENTS, or
   (b) a new identifier snuck into a file that was already covered but
       redaction did not remove it. Either way the build check catches it.

Usage: python src/build_supplement.py OUT_DIR OUT_ZIP
       (OUT_DIR is emptied and recreated; OUT_ZIP is overwritten.)

Test that the build check is actually running:
       python src/build_supplement.py OUT_DIR OUT_ZIP --plant
   plants a canary identity string in one file, so the check must fail.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# ---------------------------------------------------------------- 1. drops
# Files that carry identifying content in a form we cannot cleanly rewrite,
# and that the supplement does not need. Paths are relative to ROOT.
DROP_FILES: list[str] = [
    "paper/main.tex",       # long-form paper with named byline
    "paper/main.pdf",       # its built PDF
    "paper/main.aux",       # LaTeX aux; may contain author
    "paper/main.log",       # LaTeX log; may contain absolute paths
    "paper/main.out", "paper/main.bbl", "paper/main.blg",
    "src/make_pdfs.py",     # hard-codes name in cover / header / footer
    "src/gen_form_page.py", # generates the feedback form with the endpoint
    "src/build_supplement.py",  # THIS script: its regex patterns quote the
                                # author name and would tip a reviewer off
                                # to what was being scrubbed
    "PACK_STATUS.md",       # submission-process checklist, not research
                            # material: it names the public repository and
                            # the review venue, and a reviewer has no use
                            # for it. Committed to the repo on 2026-10-08 so
                            # a test could read it (it had lived outside git
                            # and nothing watched the pack items); the scrub
                            # caught the GitHub handle in it immediately,
                            # which is the gate working.
    "tests/test_supplement.py", # tests for THIS script; same reason as above
    "tests/test_pack.py",       # tests the TMLR-submission-pack path; the
                                # EF-Day-N path literal inside it would trip
                                # the affiliation check if it shipped.
    "feedback_form.html",   # feedback form, if present
    "feedback_config.json", # Formspree endpoint config, if present
    "index.html",           # GitHub Pages landing page; posts to Formspree
    "out/requests.log",     # per-machine request log (the actual path is
                            # under out/; keep the bare name too in case a
                            # future build moves it back)
    "requests.log",
]
# Whole directories to skip.
DROP_DIRS: list[str] = [
    ".git", ".venv", "__pycache__", ".pytest_cache", "node_modules",
]

# ---------------------------------------------------------------- 2. rewrites
# (relative path, list of (regex, replacement)) pairs.  Applied only inside
# the build tree, never to the working tree.
TEXT_REPLACEMENTS: list[tuple[str, list[tuple[str, str]]]] = [
    ("LICENSE", [
        (r"Copyright \(c\) 20\d\d Grace Jackson",
         "Copyright (c) 2026 Anonymous author (TMLR submission)"),
    ]),
    ("README.md", [
        (r"https://github\.com/gracejackson-sudo/quant-delta-predictor",
         "https://ANONYMISED/quant-delta-predictor"),
    ]),
    ("TOOL_SUMMARY.md", [
        (r"https://gracejackson-sudo\.github\.io/quant-delta-predictor/",
         "https://ANONYMISED/quant-delta-predictor/"),
    ]),
    ("RESEARCH.md", [
        (r"Grace Jackson.*?\d{4}-\d{2}-\d{2}",
         "Anonymous author"),
        (r"Grace Jackson",
         "Anonymous author"),
    ]),
    ("EXTERNAL_FEEDBACK.md", [
        # if anyone was named by mistake
        (r"Grace Jackson", "the first author"),
    ]),
]

# ---------------------------------------------------------------- 3. check
# Identity strings the build check refuses to ship. Each is (regex, label).
# Every entry uses re.IGNORECASE at check time (see check_no_identity_strings
# below), so lowercased, mixed-case, name-reversed and bare-handle forms all
# match a single pattern instead of needing separate ones. Adding a case-only
# variant here is a red flag: the search is already case-insensitive.
IDENT_STRINGS: list[tuple[str, str]] = [
    # Full name in either order and with either separator.
    (r"\bGrace[ ._-]+Jackson\b", "author name"),
    (r"\bJackson[ ._,]+Grace\b", "author name (reversed)"),
    # Bare GitHub handle, with or without the `-sudo` suffix, in URL and body.
    (r"\bgracejackson(?:-sudo)?\b", "GitHub handle"),
    # Email in any host, including bare @berkeley.
    (r"gracejackson@[A-Za-z0-9.-]+", "personal email"),
    (r"[A-Za-z0-9._-]+@berkeley\.edu", "berkeley email"),
    # Personal pages (author's own; not the paper's ANONYMISED URL).
    (r"gracejackson(?:-sudo)?\.github\.io", "personal github page"),
    # Formspree: full URL AND the 8-hex-char bare id our form uses.
    (r"formspree\.io/f/[A-Za-z0-9]+", "Formspree endpoint URL"),
    (r"\bmkjgbwyl\b", "Formspree endpoint id (bare)"),
    # Absolute paths that leak the home directory.
    (r"/Users/grace/", "author's local home directory (macOS)"),
    (r"/home/grace/", "author's local home directory (linux)"),
    (r"\bPhantomEFStartProduct\b", "author's project directory"),
    # Affiliation. "Berkeley" alone matches benign benchmark names
    # (Berkeley Function-Calling Leaderboard is on RedHat model cards), so
    # we anchor on phrases that name Berkeley as an author affiliation.
    (r"\b(?:UC\s+)?Berkeley\s+(?:student|sophomore|junior|senior|freshman"
     r"|undergrad(?:uate)?|graduate|master(?:s)?|MS|MIDS|Haas|EECS|CS|BA)\b",
     "Berkeley affiliation"),
    (r"\b(?:researcher|student|engineer|scientist|develop(?:er)?|analyst"
     r"|founder|CEO)\s+at\s+(?:UC\s+)?Berkeley\b",
     "'at Berkeley' affiliation"),
    (r"\bat\s+UC\s+Berkeley\b", "'at UC Berkeley' affiliation"),
    (r"\bEntrepreneur[\s\-]?First\b",
     "Entrepreneur First (program affiliation)"),
    (r"\bEF[- ]Day-?\d+\b", "EF-Day-N project folder name"),
    # Git commit trailers that carry a real identity: these are what a
    # co-authored-by trailer or DCO sign-off would look like in an anonymised
    # supplement, so they must not survive.
    (r"^\s*Co-Authored-By:\s*Grace[^<\n]*<[^>]+>", "git Co-Authored-By trailer"),
    (r"^\s*Signed-off-by:\s*Grace[^<\n]*<[^>]+>", "git Signed-off-by trailer"),
]

# Files that a build check MAY skip because they are historical (e.g. a
# provenance file listing the retraction), where an identifier is quoted
# in-context. Empty on purpose: if you need to add anything here, ask.
CHECK_EXEMPT: list[str] = []


def log(*a):
    print(*a, flush=True)


def copy_tree(dst: str) -> None:
    """Copy ROOT into dst, honouring DROP_FILES and DROP_DIRS."""
    drop_files_abs = {os.path.join(ROOT, p) for p in DROP_FILES}
    for base, dirs, files in os.walk(ROOT):
        # skip drop dirs
        dirs[:] = [d for d in dirs if d not in DROP_DIRS]
        # skip the build output itself if it lives under root
        rel = os.path.relpath(base, ROOT)
        if rel.startswith("_supp_build") or rel == ".":
            if rel.startswith("_supp_build"):
                continue
        for f in files:
            src = os.path.join(base, f)
            if src in drop_files_abs:
                continue
            r = os.path.relpath(src, ROOT)
            if r in DROP_FILES:
                continue
            out = os.path.join(dst, r)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            shutil.copy2(src, out)


def rewrite(dst: str) -> None:
    """Apply TEXT_REPLACEMENTS to the build tree."""
    for rel, subs in TEXT_REPLACEMENTS:
        p = os.path.join(dst, rel)
        if not os.path.isfile(p):
            continue
        s = open(p, encoding="utf-8", errors="replace").read()
        before = s
        for pat, repl in subs:
            s = re.sub(pat, repl, s, flags=re.S)
        if s != before:
            open(p, "w", encoding="utf-8").write(s)
            log(f"  rewrote {rel}")


def _pdf_text(path):
    """-> all text in a PDF, or None if it cannot be read.

    Returning None is deliberate and is reported as a finding: a PDF the
    scanner cannot read is a PDF whose contents are unknown, and "unknown"
    must not be treated as "clean". That is the distinction the old
    skip-by-extension rule collapsed.

    Text is joined across pages so a name split over a page boundary is still
    matched, and a hard-wrapped byline ("Grace\nJackson") is normalised,
    because the patterns are written for prose and PDF extraction inserts
    newlines at layout positions rather than sentence ones.
    """
    try:
        from pypdf import PdfReader
    except ImportError:          # pragma: no cover
        return None
    try:
        reader = PdfReader(path)
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        import re as _re
        return _re.sub(r"\s+", " ", "\n".join(parts))
    except Exception:
        return None


def _pdf_metadata_text(path):
    """-> (label, value) pairs from a PDF's metadata, or None if unreadable.

    Text extraction sees a visible byline. It does not see the document info
    dictionary or the XMP packet, and those are the most common way an
    anonymized PDF deanonymizes itself: a LaTeX toolchain with hyperref
    writes \\author{...} straight into /Author, and /Producer routinely
    carries a local username or an absolute path. Neither appears anywhere in
    page text.

    Every field is returned rather than a chosen subset, so a producer that
    invents its own keys (Overleaf, Distiller, Word) is covered without this
    function knowing the key names in advance. The caller scans the VALUES
    with the same identity patterns it applies to prose, so the check does
    not depend on which toolchain built the file.
    """
    try:
        from pypdf import PdfReader
    except ImportError:          # pragma: no cover
        return None
    try:
        reader = PdfReader(path)
    except Exception:
        return None
    out = []
    try:
        info = reader.metadata or {}
        for k, v in dict(info).items():
            if v is None:
                continue
            out.append((f"/Info {k}", str(v)))
    except Exception:
        out.append(("/Info", "<<UNREADABLE>>"))
    # XMP: a separate packet, and a separate leak path. dc:creator is the
    # canonical author field; the whole packet is scanned because producers
    # write arbitrary namespaces into it.
    try:
        xmp = reader.xmp_metadata
        if xmp is not None:
            raw = getattr(xmp, "rdf_root", None)
            if raw is not None:
                out.append(("XMP packet", raw.toxml()))
            for attr in ("dc_creator", "dc_title", "dc_description",
                         "pdf_producer", "xmp_creator_tool"):
                val = getattr(xmp, attr, None)
                if val:
                    out.append((f"XMP {attr}", str(val)))
    except Exception:
        out.append(("XMP", "<<UNREADABLE>>"))
    return out


def scan(dst: str) -> list[tuple[str, int, str, str]]:
    """Return every (path, line_number, matched_string, label) surviving the
    redaction."""
    hits: list[tuple[str, int, str, str]] = []
    # IGNORECASE catches lowercase / camelcase variants without needing a
    # second pattern. MULTILINE lets the ^ anchors on git trailers work.
    patterns = [(re.compile(p, re.IGNORECASE | re.MULTILINE), lbl)
                for p, lbl in IDENT_STRINGS]
    for base, dirs, files in os.walk(dst):
        dirs[:] = [d for d in dirs if d not in DROP_DIRS]
        # (see _pdf_text: PDFs are extracted rather than skipped)
        for f in files:
            src = os.path.join(base, f)
            rel = os.path.relpath(src, dst)
            if rel in CHECK_EXEMPT:
                continue
            ext = os.path.splitext(f)[1].lower()
            # PDFs are EXTRACTED, not skipped. A [final] build ships a PDF
            # with the byline in it, and until 2026-10-08 this scanner skipped
            # every .pdf by extension -- so the one file most likely to carry
            # an identity string was the one file never read. Five planted
            # text leaks were caught and a leak planted inside
            # paper/neurips_main.pdf was not.
            if ext == ".pdf":
                # metadata first: it is the leak page text cannot show
                meta = _pdf_metadata_text(src)
                if meta is None:
                    hits.append((rel, 0, "UNREADABLE PDF", "cannot scan"))
                    continue
                for label, value in meta:
                    if value == "<<UNREADABLE>>":
                        hits.append((rel, 0, label, "metadata unreadable"))
                        continue
                    for pat, lbl in patterns:
                        for m in pat.finditer(value):
                            hits.append((f"{rel} [{label}]", 0,
                                         m.group(0), lbl))
                text = _pdf_text(src)
                if text is None:
                    hits.append((rel, 0, "UNREADABLE PDF", "cannot scan"))
                    continue
            elif ext in (".png", ".jpg", ".jpeg", ".gif", ".zip",
                         ".tar", ".gz", ".xz", ".bin", ".pyc", ".npy", ".pt",
                         ".safetensors", ".woff", ".woff2", ".ttf", ".ico"):
                continue
            else:
                try:
                    text = open(src, encoding="utf-8",
                                errors="ignore").read()
                except (OSError, UnicodeDecodeError):
                    continue
            for pat, lbl in patterns:
                for m in pat.finditer(text):
                    ln = text.count("\n", 0, m.start()) + 1
                    hits.append((rel, ln, m.group(0), lbl))
    return hits


def make_zip(dst: str, out_zip: str) -> None:
    """Write dst/ as a zip, rooted at 'artifact/'."""
    if os.path.exists(out_zip):
        os.remove(out_zip)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for base, dirs, files in os.walk(dst):
            dirs[:] = [d for d in dirs if d not in DROP_DIRS]
            for f in files:
                src = os.path.join(base, f)
                arc = os.path.join("artifact", os.path.relpath(src, dst))
                z.write(src, arc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="build directory, will be emptied")
    ap.add_argument("out_zip", help="destination zip path")
    ap.add_argument("--plant", action="store_true",
                    help="plant a canary identity string so the build "
                         "check must fail. Use to verify the check is on.")
    args = ap.parse_args()

    dst = args.out_dir
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)

    log(f"[1/4] copying tree into {dst}")
    copy_tree(dst)

    log("[2/4] rewriting per TEXT_REPLACEMENTS")
    rewrite(dst)

    if args.plant:
        canary_path = os.path.join(dst, "README.md")
        with open(canary_path, "a", encoding="utf-8") as f:
            f.write("\n\n<!-- canary: Grace Jackson -->\n")
        log(f"[plant] added canary in {canary_path}, build check must fail")

    log("[3/4] scanning for identity leaks")
    hits = scan(dst)
    if hits:
        log(f"\nSUPPLEMENT SCRUB FAILED. {len(hits)} identity string(s) "
            f"remain in the build tree:\n")
        for rel, ln, mstr, lbl in hits[:100]:
            log(f"  {rel}:{ln}  [{lbl}]  {mstr!r}")
        log("\nzip NOT written. Add coverage in DROP_FILES / "
            "TEXT_REPLACEMENTS and rerun.")
        return 1

    log(f"[4/4] writing {args.out_zip}")
    make_zip(dst, args.out_zip)
    sz = os.path.getsize(args.out_zip)
    log(f"OK  {args.out_zip}  {sz/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
