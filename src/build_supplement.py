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
    # THE REPOSITORY NAME IS A DE-ANONYMISER ON ITS OWN. Rewriting only the
    # host left "https://ANONYMISED/quant-delta-predictor", and that path
    # resolves to exactly one repository on GitHub -- a reviewer who searches
    # it finds the public repo, the author's name and the handle. Found
    # 2026-10-13 by scanning the shipped zip for the repo name rather than
    # for the identity strings. The path and the bare `cd` form both go.
    ("README.md", [
        (r"https://github\.com/gracejackson-sudo/quant-delta-predictor",
         "https://ANONYMISED/REPO"),
        (r"\bquant-delta-predictor\b", "REPO"),
    ]),
    ("TOOL_SUMMARY.md", [
        (r"https://gracejackson-sudo\.github\.io/quant-delta-predictor/",
         "https://ANONYMISED/REPO/"),
        (r"\bquant-delta-predictor\b", "REPO"),
    ]),
    ("BIAS_CORRECTION.md", [(r"\bquant-delta-predictor\b", "REPO")]),
    ("RANKING.md", [(r"\bquant-delta-predictor\b", "REPO")]),
    ("NEGATIVE_RESULT.md", [(r"\bquant-delta-predictor\b", "REPO")]),
    ("ONE_SIDED_COVERAGE.md", [(r"\bquant-delta-predictor\b", "REPO")]),
    ("EXTERNAL_FEEDBACK.md", [(r"\bquant-delta-predictor\b", "REPO")]),
    ("ACCOUNTING.md", [(r"\bquant-delta-predictor\b", "REPO")]),
    ("SCOPE.md", [(r"\bquant-delta-predictor\b", "REPO")]),
    ("src/qwen35_published.py",
     [(r"\bquant-delta-predictor\b", "REPO")]),
    ("src/publisher_census.py",
     [(r"\bquant-delta-predictor\b", "REPO")]),
    # the generator that WRITES the cd line into TOOL_SUMMARY.md
    ("src/gen_bias_doc.py",
     [(r"\bquant-delta-predictor\b", "REPO")]),
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
    # The repository name resolves to one repo on GitHub, so it identifies
    # the author as surely as the handle does. Added 2026-10-13.
    (r"\bquant-delta-predictor\b", "public repository name"),
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



# ---------------------------------------------------------- commit relabelling
# A COMMIT HASH DE-ANONYMISES AS WELL AS A NAME. GitHub indexes commit SHAs:
# pasting one into its search box resolves to the repository that contains it,
# and from there to the author's name and handle. The paper's audit sections
# are full of them -- that is the point of the sections -- so dropping them is
# not an option and neither is leaving them.
#
# They are relabelled instead. Every hash-shaped token in the build tree is
# resolved against THIS repository: if `git cat-file` says it is a commit
# here, it becomes a stable label; if it is not, it is left exactly as it is.
# That distinction is the whole design. Most hash-shaped tokens in the tree
# are not ours at all -- data/redhatai_models.json carries a Hugging Face
# revision for every card, and those are public identifiers of someone
# else's artifacts, carry no information about this repository, and must
# survive the scrub intact for the data to stay usable.
#
# Labels are assigned in committer-date order so that `COMMIT-01` is the
# earliest commit mentioned anywhere in the supplement. A reviewer can still
# follow "the defect at COMMIT-11 was fixed at COMMIT-14" and can still see
# that two sites mean the same commit; what they cannot do is search for it.
# The mapping is written beside the zip, OUTSIDE the build tree and
# gitignored, because the mapping is exactly the thing the relabelling
# exists to withhold.
HASHLIKE = re.compile(r"(?<![0-9a-zA-Z])([0-9a-f]{7,40})(?![0-9a-zA-Z])")

# OURS, BUT NO LONGER REACHABLE. The relabeller asks git whether a token is a
# commit in this repository, which is the right question and has one blind
# spot: a hash that was superseded by a rebase or amend stops resolving once
# the old objects fall out of a clone. It is still OUR hash -- GitHub keeps
# orphaned commits fetchable by full hash -- so it still resolves to this
# repository for anyone who pastes it into a search box, and it is exactly
# the kind of hash a disclosure paragraph cites on purpose.
#
# Found 2026-10-13 immediately after rewriting two commits to drop a
# forbidden trailer: the superseded hashes were relabelled correctly in the
# working clone ONLY because local backup branches still held the objects.
# A fresh clone of origin does not resolve them -- measured -- and would have
# shipped both in plain text. Reachability in whatever clone happens to run
# the build is not a property the anonymisation should depend on.
# The repository reached 122 commits when the completeness guard was
# written. The guard only needs a floor that a shallow or truncated clone
# falls under; it is not a pin on history length.
MIN_HISTORY_COMMITS = 100

KNOWN_OURS_ORPHANED = {
    "1ba6159": "earlier version of 99f1ca5, replaced by rebase/amend; cited "
               "in EXTERNAL_FEEDBACK.md's M1 disclosure",
    "1330b28": "amended to eef93d4 on 2026-10-13 to drop a Co-Authored-By "
               "trailer; cited in AUDIT_DISCIPLINE.md's rewrite disclosure",
    "3921ec5": "rebased to ad1b903 in the same rewrite; same disclosure",
}

# REWRITE PROSE, NEVER DATA. The relabeller only edits the file types that
# discuss commits; data files are read-only to it. The reason is a false
# positive it would otherwise be free to make: `[0-9a-f]{7,40}` matches a
# seven-digit decimal, and a seven-digit decimal is a perfectly ordinary
# value in a CSV. If one ever collided with a valid short hash in this
# repository, the relabeller would silently rewrite a data value, and the
# build would report a success. No such token exists in the tree today --
# measured, zero all-digit candidates in the data files -- which is exactly
# why the guard goes in now rather than after one appears.
#
# Excluding them is not the same as knowing they are clean, and
# "clean because nothing happens to look at it" is the shape this
# repository keeps finding. So the excluded files are still SCANNED: any of
# our own commit hashes sitting in one is reported and fails the build,
# which forces a decision about that specific site instead of leaving the
# exclusion to cover it.
RELABEL_EXTS = {".md", ".tex", ".py", ".txt", ".bib", ".html", ".cfg", ".ini"}

# Named data files that DO get relabelled, by exception and one at a time.
# The blanket rule excludes data because `[0-9a-f]{7,40}` matches a
# seven-digit decimal and rewriting a CSV value would be reported as a
# successful build. These files are exceptions because their hash-shaped
# tokens have been looked at: out/percent_defect.json records which
# revisions a measurement of history was taken at, so its only such tokens
# are commit hashes of ours, and its numbers are counts and four-place
# percentages with no token long enough to collide. Adding a file here means
# having checked that, not having assumed it.
RELABEL_FILES = {
    "out/percent_defect.json",
}


def _resolve_hashes(cands):
    """-> {token: full sha} for the tokens that are commits in THIS repo.

    One `git cat-file --batch-check` pass rather than a process per token.
    The first version shelled out once per candidate and took minutes on a
    tree whose data file carries a Hugging Face revision per card; the
    batched form is the same question asked once.
    """
    if not cands:
        return {}
    toks = sorted(cands)
    r = subprocess.run(["git", "cat-file", "--batch-check"],
                       input="\n".join(toks) + "\n",
                       capture_output=True, text=True, cwd=ROOT)
    out = {}
    for tok, line in zip(toks, r.stdout.splitlines()):
        parts = line.split()
        # "<sha> commit <size>" for ours; "<tok> missing" otherwise. An
        # ambiguous short hash reports "ambiguous" and is NOT treated as
        # ours: it would be guesswork about which commit was meant.
        if len(parts) >= 2 and parts[1] == "commit":
            out[tok] = parts[0]
    return out


def _commit_dates(fulls):
    """-> {full sha: committer timestamp}, in one pass."""
    if not fulls:
        return {}
    r = subprocess.run(["git", "log", "--no-walk", "--format=%H %ct",
                        *sorted(fulls)], capture_output=True, text=True,
                       cwd=ROOT)
    d = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            d[parts[0]] = int(parts[1])
    return d



def assert_history_is_complete():
    """Refuse to relabel from a clone that cannot see its own history.

    -> the number of reachable commits.

    THE RELABELLER AND ITS VERIFIER SHARE A DEPENDENCY, which is the
    shared-assumption trap the paper's float-boundary entry is about. Both
    ask `git cat-file` whether a token is a commit here. In a shallow clone
    the answer is "no" for every commit outside the fetched depth -- so the
    relabeller leaves those hashes alone AND the post-build verifier agrees
    they are not ours, and the build reports success having shipped most of
    them in plain text.

    Measured 2026-10-13: a `--depth 50` clone of origin relabelled 10
    commits at 51 sites where the full clone relabelled 31 at 145, and both
    builds printed OK. A second check written the same way would not have
    caught it; only a precondition does.
    """
    r = subprocess.run(["git", "rev-parse", "--is-shallow-repository"],
                       capture_output=True, text=True, cwd=ROOT)
    if r.stdout.strip() == "true":
        raise SystemExit(
            "REFUSING TO BUILD: this is a shallow clone. Commit-hash "
            "relabelling resolves each hash against local history, and a "
            "shallow clone cannot resolve the commits outside its depth -- "
            "so they would ship in plain text and the post-build check, "
            "which asks the same question, would agree they are clean. "
            "Run `git fetch --unshallow` and rebuild.")
    r = subprocess.run(["git", "rev-list", "--count", "HEAD"],
                       capture_output=True, text=True, cwd=ROOT)
    try:
        n = int(r.stdout.strip())
    except ValueError:
        raise SystemExit(
            "REFUSING TO BUILD: cannot count this clone's history, so the "
            "completeness of commit relabelling cannot be established.")
    if n < MIN_HISTORY_COMMITS:
        raise SystemExit(
            f"REFUSING TO BUILD: this clone reaches {n} commits and the "
            f"repository had at least {MIN_HISTORY_COMMITS} when this guard "
            f"was written. A truncated history relabels only what it can "
            f"see. Fetch the full history and rebuild.")
    return n


def relabel_commits(dst, map_path=None):
    """Replace this repository's own commit hashes with stable labels.

    -> (mapping {full-ish hash: label}, number of sites rewritten).
    """
    texts, others = [], []
    for root, _dirs, files in os.walk(dst):
        for fn in files:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, dst)
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gz",
                       ".pyc"):
                continue
            try:
                body = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            if ext in RELABEL_EXTS or rel in RELABEL_FILES:
                texts.append((rel, p, body))
            else:
                others.append((rel, body))

    # Collect first, assign labels second: a label has to be stable across
    # files, and assigning as we walk would make it depend on walk order.
    cands = set()
    for _rel, _p, _s in texts:
        cands.update(m.group(1) for m in HASHLIKE.finditer(_s))
    # One commit can appear at 7, 8 and 40 hex; they must get ONE label.
    canon = _resolve_hashes(cands)
    # An orphaned hash of ours gets a label of its own rather than the label
    # of the commit that replaced it: the disclosure's whole point is that
    # the two are different objects.
    for _h in sorted(cands & set(KNOWN_OURS_ORPHANED)):
        canon.setdefault(_h, _h)
    ours = set(canon)
    skipped_nonours = len(cands) - len(ours)
    dates = _commit_dates(set(canon.values()))
    # Orphans have no committer date in this clone, so they sort after the
    # dated commits by hash rather than silently taking date 0 and landing
    # at COMMIT-01.
    order = sorted(set(canon.values()),
                   key=lambda f: (f in KNOWN_OURS_ORPHANED,
                                  dates.get(f, 0), f))
    label_of = {full: "COMMIT-%02d" % (i + 1) for i, full in enumerate(order)}
    mapping = {h: label_of[canon[h]] for h in ours}

    # The excluded file types are not rewritten, but they are not trusted
    # either: report any of our commits that live in one.
    stranded = []
    for rel, body in others:
        found = sorted({m.group(1) for m in HASHLIKE.finditer(body)}
                       & set(canon))
        if found:
            stranded.append((rel, found))

    sites = 0
    for rel, p, s in texts:
        def _sub(m):
            nonlocal sites
            if m.group(1) in mapping:
                sites += 1
                return mapping[m.group(1)]
            return m.group(0)
        out = HASHLIKE.sub(_sub, s)
        if out != s:
            open(p, "w", encoding="utf-8").write(out)
    log(f"  relabelled {len(order)} distinct commit(s) at {sites} site(s); "
        f"left {skipped_nonours} distinct hash-shaped token(s) that are "
        f"not commits "
        f"in this repository (Hugging Face revisions and the like) untouched")
    if stranded:
        log("  COMMIT HASHES IN FILES THE RELABELLER WILL NOT REWRITE:")
        for rel, found in stranded:
            log(f"    {rel}: {found}")
        log("  Each is a de-anonymiser the relabelling did not remove. "
            "Decide per site: drop the file, move the reference into prose, "
            "or add the extension to RELABEL_EXTS if rewriting it is safe.")
    if map_path:
        with open(map_path, "w", encoding="utf-8") as f:
            f.write("# Commit relabelling for the anonymous supplement.\n"
                    "# NOT in the repository and NOT in the zip: this file is\n"
                    "# the de-anonymiser the relabelling removes. Local only.\n")
            for full in order:
                f.write(f"{label_of[full]}\t{full}\n")
        log(f"  mapping written to {map_path} (local, not shipped)")
    return mapping, sites, stranded

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



def verify_no_our_commits(dst):
    """Scan the FINISHED build tree for this repository's commit hashes.

    -> (remaining, coverage) where coverage maps extension -> (files read,
    files unreadable).

    The relabeller rewrites prose file types and reports our hashes found in
    the types it will not rewrite. Neither of those is a check that the
    output is clean: one edits, the other looks at a subset. This reads the
    tree the zip is about to be made from, EVERY extension, with no
    allowlist -- .py and .csv and .json included, and PDFs through the text
    extractor -- and resolves every hash-shaped token it finds against this
    repository. A file it cannot read at all is reported as unreadable
    rather than counted as clean, which is the distinction the old
    skip-by-extension rule collapsed.
    """
    cands, coverage, unreadable = {}, {}, []
    for root, _dirs, files in os.walk(dst):
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, dst)
            ext = os.path.splitext(fn)[1].lower() or "(none)"
            read, bad = coverage.get(ext, (0, 0))
            body = None
            if ext == ".pdf":
                body = _pdf_text(fp)
                if body is not None:
                    # _pdf_metadata_text returns (label, text) pairs, not a
                    # string: the document info dictionary and the XMP
                    # packet are separate leak paths and it reports which.
                    _meta = _pdf_metadata_text(fp) or []
                    body = body + "\n" + "\n".join(
                        str(_t) for _, _t in _meta)
            else:
                try:
                    body = open(fp, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    body = None
            if body is None:
                coverage[ext] = (read, bad + 1)
                unreadable.append(rel)
                continue
            coverage[ext] = (read + 1, bad)
            for m in HASHLIKE.finditer(body):
                cands.setdefault(m.group(1), set()).add(rel)
    resolved = _resolve_hashes(set(cands))
    resolved.update({h: h for h in set(cands) & set(KNOWN_OURS_ORPHANED)})
    remaining = {h: sorted(cands[h]) for h in resolved}
    return remaining, coverage, unreadable

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

    log(f"[1/6] copying tree into {dst}")
    copy_tree(dst)

    log("[2/6] rewriting per TEXT_REPLACEMENTS")
    rewrite(dst)

    if args.plant:
        canary_path = os.path.join(dst, "README.md")
        with open(canary_path, "a", encoding="utf-8") as f:
            f.write("\n\n<!-- canary: Grace Jackson -->\n")
        log(f"[plant] added canary in {canary_path}, build check must fail")

    _n_hist = assert_history_is_complete()
    log(f"[3/6] relabelling this repository's own commit hashes "
        f"({_n_hist} commits reachable)")
    _, _, _stranded = relabel_commits(dst, map_path=os.path.join(
        os.path.dirname(os.path.abspath(args.out_zip)) or ".",
        "COMMIT_RELABEL_MAP.local.tsv"))

    if _stranded:
        log("\nSUPPLEMENT SCRUB FAILED. This repository's own commit "
            "hashes remain in files the relabeller does not rewrite; see "
            "the list above. zip NOT written.")
        return 1

    log("[4/6] verifying no commit hash of ours survives anywhere")
    _left, _cov, _unread = verify_no_our_commits(dst)
    log(f"  files read by extension: "
        + ", ".join(f"{_e} {_r}" + (f" (+{_b} unreadable)" if _b else "")
                    for _e, (_r, _b) in sorted(_cov.items())))
    if _unread:
        log(f"  UNREADABLE, so NOT verified clean: {_unread}")
    if _left or _unread:
        log("\nSUPPLEMENT SCRUB FAILED.")
        for _h, _where in sorted(_left.items()):
            log(f"  {_h} still present in: {_where}")
        if _unread:
            log("  Files that could not be read cannot be declared clean; "
                "drop them in DROP_FILES or make them readable.")
        log("  zip NOT written.")
        return 1

    log("[5/6] scanning for identity leaks")
    hits = scan(dst)
    if hits:
        log(f"\nSUPPLEMENT SCRUB FAILED. {len(hits)} identity string(s) "
            f"remain in the build tree:\n")
        for rel, ln, mstr, lbl in hits[:100]:
            log(f"  {rel}:{ln}  [{lbl}]  {mstr!r}")
        log("\nzip NOT written. Add coverage in DROP_FILES / "
            "TEXT_REPLACEMENTS and rerun.")
        return 1

    log(f"[6/6] writing {args.out_zip}")
    make_zip(dst, args.out_zip)
    sz = os.path.getsize(args.out_zip)
    log(f"OK  {args.out_zip}  {sz/1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
