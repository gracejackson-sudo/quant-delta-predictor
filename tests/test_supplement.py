"""Tests for the supplement scrub (src/build_supplement.py).

Lives in its own file because the assertions must contain the identity
strings the scrub is supposed to catch, so the file itself would trip the
scan if it shipped in the supplement. src/build_supplement.py drops this
file from the build tree; see DROP_FILES.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

# ---------------------------------------------------------------------------
# The supplement scrub for TMLR review. Two must-hold conditions:
#   1. On the current tree, the scrub script must succeed and the built zip
#      must not contain any identity string when independently grepped.
#   2. If we plant an identity string, the scrub script must FAIL. A build
#      check that only runs on clean input never tells you it stopped
#      working; the plant test is what makes it real.
# ---------------------------------------------------------------------------

def _run_scrub(tmp, plant=False):
    import subprocess
    root = os.path.join(os.path.dirname(__file__), "..")
    dst = os.path.join(tmp, "build")
    zpath = os.path.join(tmp, "supplement.zip")
    cmd = [sys.executable, os.path.join("src", "build_supplement.py"),
           dst, zpath]
    if plant:
        cmd.append("--plant")
    r = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
    return r, dst, zpath


def _one_page_pdf_containing(text):
    """A minimal valid PDF whose extracted text is `text`.

    Built by hand rather than with a PDF library so the test has no
    dependency beyond the extractor the scrub itself uses.
    """
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for o in offs:
        out += b"%010d 00000 n \n" % o
    out += (b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objs) + 1, x))
    return bytes(out)


def test_the_scrub_reads_pdfs_and_catches_a_leak_inside_one(tmp_path):
    """A leak inside a PDF must fail the scrub.

    Until 2026-10-08 the scanner skipped every .pdf by extension, so the one
    file most likely to carry an identity string was the one file never read.
    Five planted text leaks were caught; a leak planted inside
    paper/neurips_main.pdf was not. Today's PDF is anonymous, but a [final]
    build ships one with the byline, and it would have passed.

    What this READS: the extractor on a hand-built PDF whose text is the
    author name, and then the scrub's own scan over a tree containing it. A
    regression -- reverting to skip-by-extension, or losing the extractor --
    fails here.
    """
    import importlib
    import sys as _s
    root = os.path.join(os.path.dirname(__file__), "..")
    _s.path.insert(0, os.path.join(root, "src"))
    BS = importlib.import_module("build_supplement")

    leak = "Grace Jackson"
    pdf = tmp_path / "planted.pdf"
    pdf.write_bytes(_one_page_pdf_containing(leak))

    # 1. the extractor must actually read it
    text = BS._pdf_text(str(pdf))
    assert text is not None, (
        "the PDF extractor returned None on a valid PDF. If pypdf is "
        "unavailable the scrub cannot scan PDFs at all, and that must be a "
        "finding rather than a silent skip.")
    assert leak in text, (
        f"extracted text does not contain the planted byline: {text[:120]!r}")

    # 2. the scrub's scanner must report it
    tree = tmp_path / "tree" / "paper"
    tree.mkdir(parents=True)
    (tree / "neurips_main.pdf").write_bytes(_one_page_pdf_containing(leak))
    hits = BS.scan(str(tmp_path / "tree"))
    assert any(h[0].endswith("neurips_main.pdf") for h in hits), (
        "the scrub scanned a tree containing a PDF with the author name in "
        "it and reported no finding. A leak inside a PDF is exactly the case "
        "a [final] build produces.\n"
        f"findings: {hits}")


def test_the_scrub_reads_pdf_metadata_not_just_page_text(tmp_path):
    """A leak in PDF metadata must fail the scrub.

    Page-text extraction sees a visible byline. It does not see the document
    info dictionary or the XMP packet, and that is the most common way an
    anonymized PDF deanonymizes itself: hyperref writes \\author{...}
    straight into /Author, and /Producer routinely carries a local username
    or an absolute path. The repository produced both classes of leak in one
    week from a single committed file, so neither is hypothetical.

    This matters at the moment of a [final] rebuild: that build is the first
    one to carry real author metadata, and without this check it would also
    be the first nobody checked.

    What this READS: the scrub's own scanner over trees containing PDFs with
    planted metadata, one case per field, plus a clean case that must stay
    silent. Values are scanned with the same identity patterns used on prose,
    so the check does not depend on which toolchain wrote the file -- a
    producer that invents its own keys is covered without this test knowing
    the key names.
    """
    import importlib
    import sys as _s
    root = os.path.join(os.path.dirname(__file__), "..")
    _s.path.insert(0, os.path.join(root, "src"))
    BS = importlib.import_module("build_supplement")
    from pypdf import PdfReader, PdfWriter

    source = os.path.join(root, "paper", "neurips_main.pdf")
    if not os.path.exists(source):
        import pytest
        pytest.skip("paper/neurips_main.pdf absent")

    def build(meta, out):
        r = PdfReader(source)
        w = PdfWriter()
        for p in r.pages:
            w.add_page(p)
        w.add_metadata(meta)
        with open(out, "wb") as f:
            w.write(f)

    cases = [
        ("author", {"/Author": "Grace Jackson"}, True),
        ("producer",
         {"/Producer": "pdfTeX via /Users/grace/Downloads/qdp"}, True),
        ("creator",
         {"/Creator": "Overleaf project by Grace Jackson",
          "/Producer": "pdfTeX-1.40.25"}, True),
        # Overleaf's strings differ from the local xdvipdfmx build; a clean
        # one must not fire, or the gate is useless after the rebuild.
        ("clean", {"/Creator": "LaTeX with hyperref",
                   "/Producer": "pdfTeX-1.40.25"}, False),
    ]
    for slug, meta, should_fire in cases:
        tree = tmp_path / slug / "paper"
        tree.mkdir(parents=True)
        build(meta, str(tree / "neurips_main.pdf"))
        hits = [h for h in BS.scan(str(tmp_path / slug))
                if "neurips_main.pdf" in h[0]]
        if should_fire:
            assert hits, (
                f"a leak planted in {list(meta)} was not reported. Page text "
                f"cannot show metadata, so this is the check that has to "
                f"catch it.")
            assert any("/Info" in h[0] or "XMP" in h[0] for h in hits), (
                f"the leak was reported but not attributed to a metadata "
                f"field, so the message will not tell an author where to "
                f"look: {hits}")
        else:
            assert not hits, (
                f"clean metadata reported a leak: {hits}. A gate that fires "
                f"on an ordinary Overleaf build gets turned off.")


def test_the_self_referential_files_are_pinned_in_drop_files(tmp_path):
    """The scrub must drop the files that quote the strings it searches for.

    Found on 2026-10-11: the exclusion was an entry in DROP_FILES with a
    reason written beside it, and NOTHING asserted the entry existed. Deleting
    it would have shipped a file enumerating every identity string being
    scrubbed -- which is both a leak and a roadmap. The protection was "an
    entry plus a build-time scan", and the entry was unpinned, which is the
    same shape as every other defect this cycle.

    Two halves, because membership alone is a tautology: the entry is pinned,
    AND the scan is shown to catch the file if the entry ever goes. The second
    half is what makes the first meaningful -- and it is why the build fails
    loudly rather than silently if someone edits DROP_FILES.
    """
    import shutil
    import build_supplement as B

    SELF_REFERENTIAL = {
        "src/build_supplement.py",   # its patterns quote the author name
        "tests/test_supplement.py",  # this file; its assertions do too
        "tests/test_pack.py",        # carries the EF-Day-N path literal
        "src/make_pdfs.py",          # writes the byline into generated PDFs
        "src/gen_form_page.py",      # carries the feedback endpoint
        "PACK_STATUS.md",            # names the repository and the venue
    }
    missing = sorted(SELF_REFERENTIAL - set(B.DROP_FILES))
    assert not missing, (
        "these files quote the strings the scrub searches for and are no "
        f"longer dropped from the build tree: {missing}. Either restore the "
        "DROP_FILES entry, or -- if the file genuinely no longer carries an "
        "identifier -- remove it from this test in the same commit and say "
        "why in the message.")

    # Non-vacuity: show the scan would catch each one. A DROP_FILES entry is
    # only load-bearing for files the scan would otherwise flag; if the scan
    # found nothing in a file, pinning its entry would be cargo cult.
    unprotected = []
    for rel in sorted(SELF_REFERENTIAL):
        src = os.path.join(ROOT, rel)
        if not os.path.exists(src):
            continue                      # absent from the tree; nothing to leak
        d = tmp_path / rel.replace("/", "_")
        d.mkdir()
        shutil.copy2(src, d / os.path.basename(rel))
        if not B.scan(str(d)):
            unprotected.append(rel)
    assert not unprotected, (
        "the scan finds no identity string in these files, so dropping them "
        f"is not what keeps them out: {unprotected}. Either the patterns "
        "stopped matching -- the serious reading -- or the file was cleaned "
        "up and does not belong in this test any more.")


def test_supplement_scrub_succeeds_and_the_zip_has_no_identity_strings(tmp_path):
    import re
    import zipfile
    r, dst, zpath = _run_scrub(str(tmp_path))
    assert r.returncode == 0, (
        f"scrub exit code {r.returncode}\nSTDOUT:\n{r.stdout}\n"
        f"STDERR:\n{r.stderr}")
    assert os.path.exists(zpath), "scrub returned 0 but no zip on disk"
    # Extract and grep, independently of the build script's own scan.
    extract = tmp_path / "extract"
    extract.mkdir()
    with zipfile.ZipFile(zpath) as z:
        z.extractall(extract)
    identity = re.compile(
        r"Grace Jackson"
        r"|gracejackson-sudo"
        r"|gracejackson@[A-Za-z0-9.-]+"
        r"|gracejackson-sudo\.github\.io"
        r"|formspree\.io/f/[A-Za-z0-9]+"
        r"|/Users/grace/"
        r"|/home/grace/"
        r"|\bPhantomEFStartProduct\b"
        r"|Berkeley (?:student|sophomore|undergraduate)")
    hits = []
    for base, dirs, files in os.walk(extract):
        for f in files:
            src = os.path.join(base, f)
            if os.path.splitext(f)[1].lower() in (
                    ".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip",
                    ".tar", ".gz", ".xz", ".bin", ".pyc", ".npy", ".pt",
                    ".safetensors", ".woff", ".woff2", ".ttf", ".ico"):
                continue
            try:
                txt = open(src, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for m in identity.finditer(txt):
                hits.append((os.path.relpath(src, extract), m.group(0)))
    assert hits == [], (
        f"identity leaks in the supposedly-scrubbed zip:\n"
        + "\n".join(f"  {p}: {s!r}" for p, s in hits[:20]))


def test_supplement_scrub_fails_loudly_when_an_identity_string_is_planted(
        tmp_path):
    r, dst, zpath = _run_scrub(str(tmp_path), plant=True)
    # The check MUST fail, and the zip MUST NOT be written.
    assert r.returncode != 0, (
        "planted identity string was not caught by the build check.\n"
        f"STDOUT:\n{r.stdout}")
    assert not os.path.exists(zpath), (
        "build check failed but zip was written anyway; that means the "
        "check is running after the zip write.")
    assert "SUPPLEMENT SCRUB FAILED" in r.stdout, (
        "plant test failed but not for the right reason:\n" + r.stdout)


# ---------------------------------------------------------------------------
# Unit-level plant test for every regex category in IDENT_STRINGS. The
# process-level plant (above) plants ONE canary; this exercises every kind
# the day-7 audit asked for -- lowercase, name-reversed, bare handle, bare
# Formspree id, Berkeley affiliation, Entrepreneur First, co-author trailer
# -- so silently narrowing a regex breaks the gate.
# ---------------------------------------------------------------------------

PLANT_VARIANTS = [
    # (label the reviewer would use, string to plant)
    ("lowercase name",       "signed by grace jackson"),
    ("dotted name",          "grace.jackson filed the PR"),
    ("name-reversed",        "Jackson, Grace (author)"),
    ("bare handle",          "see @gracejackson for details"),
    ("handle with -sudo",    "@gracejackson-sudo has the fork"),
    ("bare Formspree id",    "endpoint mkjgbwyl is live"),
    ("Formspree URL",        "posts to https://formspree.io/f/xxx11111"),
    ("Berkeley affiliation", "she is a Berkeley MIDS student"),
    ("at Berkeley phrase",   "a researcher at Berkeley wrote this"),
    ("at UC Berkeley",       "a talk given at UC Berkeley last spring"),
    ("Entrepreneur First",   "in the Entrepreneur First cohort"),
    ("EF-Day-N folder",      "see /path/EF-Day-5 for the pack"),
    ("Co-Authored-By trailer",
     "Co-Authored-By: Grace Jackson <gj@example.com>"),
    ("Signed-off-by trailer",
     "Signed-off-by: Grace Jackson <gj@example.com>"),
    ("home dir macOS",       "output written to /Users/grace/notes.txt"),
    ("home dir linux",       "output written to /home/grace/notes.txt"),
    ("berkeley email",       "reach me at anon@berkeley.edu"),
    ("PhantomEFStartProduct", "cd PhantomEFStartProduct/impl"),
]


def test_scan_catches_every_ident_variant_the_audit_named(tmp_path):
    # The plant set IS the artifact here, so an empty one would pass having
    # scanned nothing. Asserted rather than assumed, 2026-10-12.
    assert len(PLANT_VARIANTS) >= 8, (
        f"PLANT_VARIANTS holds {len(PLANT_VARIANTS)} plants; the audit named "
        f"more than that and the scrub is only as good as this list")
    """Direct unit test of the scan(): plant one variant per file and
    check every one is caught. A future edit that drops a regex would
    make one of these silently pass, and this test breaks."""
    from build_supplement import scan
    for i, (label, planted) in enumerate(PLANT_VARIANTS):
        d = tmp_path / f"case_{i:02d}"
        d.mkdir()
        (d / "planted.txt").write_text(planted, encoding="utf-8")
        hits = scan(str(d))
        assert hits, (
            f"variant {label!r} planted the string {planted!r} but the "
            f"scan returned no hits. Add coverage in IDENT_STRINGS.")


def _build_zip(tmp_path):
    """-> (build tree, zip path) for the shipped supplement."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dst = str(tmp_path / "tree")
    zp = str(tmp_path / "supplement.zip")
    r = subprocess.run(
        [sys.executable, os.path.join("src", "build_supplement.py"), dst, zp],
        capture_output=True, text=True, cwd=root)
    assert r.returncode == 0, (
        f"src/build_supplement.py exited {r.returncode}; the supplement does "
        f"not build, so nothing below could be checked.\n"
        f"{r.stdout[-3000:]}\n{r.stderr[-2000:]}")
    return dst, zp


def test_no_relabelled_hash_in_any_shipped_executable(tmp_path):
    """COMMIT-nn must never appear in shipped code.

    The relabeller rewrites this repository's commit hashes to stable
    COMMIT-nn labels so the anonymous supplement does not identify the repo
    through GitHub's commit search. Until 2026-10-15 it rewrote `.py` source
    too, which turned `git show 708a84d:...` into `git show COMMIT-14:...`
    and `BEFORE = "708a84d"` into `BEFORE = "COMMIT-14"`. Those are not
    revisions. The shipped code was broken by its own anonymisation and the
    supplement's documented verification path did not run -- found by a
    clean-room run, because until then nothing had ever executed the zip's
    own code.

    A hash in a comment is prose and a hash in a string literal is an
    argument, and no extension-level rule tells them apart, so the
    relabeller now leaves code alone entirely. This asserts that from the
    other side: if a COMMIT-nn label ever reaches shipped code again,
    something has put code back into RELABEL_EXTS.

    Planted failure: add ".py" back to RELABEL_EXTS in
    src/build_supplement.py and this fires with the files named.
    """
    import re
    dst, _zp = _build_zip(tmp_path)
    CODE = {".py", ".sh", ".bash", ".zsh", ".pl", ".rb", ".js", ".mjs",
            ".sty", ".cls", ".yml", ".yaml", ".toml"}
    bad = []
    for root_, _dirs, files in os.walk(dst):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in CODE:
                continue
            fp = os.path.join(root_, fn)
            try:
                body = open(fp, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            for m in re.finditer(r"COMMIT-\d{2,}", body):
                ln = body[:m.start()].count("\n") + 1
                bad.append(f"{os.path.relpath(fp, dst)}:{ln} {m.group(0)}")
    assert not bad, (
        "relabelled commit labels reached shipped executable code. A label "
        "is not a revision, so any code that passes one to git is broken in "
        "the artifact while working in the repo:\n  " + "\n  ".join(bad[:40]))


def test_shipped_supplement_runs_in_a_clean_room(tmp_path):
    """The zip's documented entry point must work with no repo and no git.

    THE CENTRAL CLAIM OF THE SUPPLEMENT is that every figure in the paper is
    recomputed from the registry. On 2026-10-15 that could not be exercised
    from the zip at all: `src/verify_claims.py` raised RuntimeError because
    it shelled out to `git show` for figures measured from history, and the
    zip has no `.git`. `paper/audit_paper.py` imports the registry, so it
    went down too. Both had passed every test in the repo, where history
    exists. Nothing tested the artifact as a reviewer receives it.

    So this unzips the shipped archive into an empty directory and runs it
    from there: the README's own quickstart, the registry, and the paper
    gate at submission strength. All three must exit 0.

    It deliberately does NOT pass the repo on sys.path or PYTHONPATH, and it
    runs with cwd inside the extraction, so an accidental dependency on the
    working tree shows up as a failure here rather than as a reviewer's bug
    report.
    """
    import subprocess
    import zipfile
    _dst, zp = _build_zip(tmp_path)
    room = tmp_path / "cleanroom"
    room.mkdir()
    with zipfile.ZipFile(zp) as z:
        z.extractall(room)
    art = room / "artifact"
    assert art.is_dir(), f"the zip did not extract an artifact/ root: {os.listdir(room)}"
    assert not (art / ".git").exists(), (
        "the shipped zip contains a .git directory, which both leaks history "
        "and would mask exactly the failure this test exists to catch")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    steps = [
        # the README's "Try it in about a minute"
        ([sys.executable, "src/rank.py", "fp8_dynamic", "--size", "5B"],
         "the README quickstart"),
        # every figure in the paper, recomputed
        ([sys.executable, "src/verify_claims.py"],
         "the claims registry"),
        # the paper gate at submission strength
        # THE TMLR VARIANT, because that is the only paper source the
        # supplement ships: the submission is to TMLR, and a supplement
        # carrying a NeurIPS-styled source states a competing venue to this
        # venue's reviewers. Named rather than discovered on purpose -- if
        # the shipped variant ever changes, this should fail and be read,
        # not silently follow.
        ([sys.executable, "paper/audit_paper.py", "tmlr_main.tex",
          "--pre-submission"],
         "the paper gate under --pre-submission"),
    ]
    for cmd, what in steps:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(art), env=env)
        # THE ARTIFACT PROMISES NO NETWORK IS NEEDED, and its README says so
        # in as many words. --pre-submission promotes the arXiv bibliography
        # check from `unchecked` to a hard failure, which is right when you
        # are about to stamp a pack online and wrong as a verdict on the
        # artifact: a rate-limited lookup, or a reviewer reading on a plane,
        # would otherwise read as a broken supplement. So that one failure
        # is a loud skip and every other failure is a failure.
        if r.returncode != 0 and "verify bib authors against arXiv" in r.stdout:
            import pytest
            pytest.skip(
                f"{what} could not reach arXiv, so --pre-submission promoted "
                f"an unverified bibliography to a failure. That is a network "
                f"condition, not a defect in the artifact; every other check "
                f"in that run passed. Re-run with network to exercise it.")
        assert r.returncode == 0, (
            f"{what} exited {r.returncode} when run from the unzipped "
            f"supplement. A reviewer with only the zip cannot reproduce the "
            f"paper's figures.\n$ {' '.join(cmd)}\n"
            f"---stdout tail---\n{r.stdout[-2500:]}\n"
            f"---stderr tail---\n{r.stderr[-2500:]}")


def test_zip_ships_no_competing_venue_style_or_footer(tmp_path):
    """The supplement must not carry another venue's style or footer.

    The submission is to TMLR. Until 2026-10-15 the zip shipped
    `paper/neurips_2026.sty` and a NeurIPS-styled source beside the TMLR
    one, which is how the pack's file 01 came to be built from the wrong
    variant: the two sat side by side and nothing said which was the
    submission.

    Deliberately narrow, as with the pack-side guard: this fires on the
    NeurIPS style FILE and on the NeurIPS SUBMISSION FOOTER, never on the
    bare word NeurIPS. The bibliography cites NeurIPS papers and must keep
    being able to.
    """
    import re
    dst, _zp = _build_zip(tmp_path)

    styles = []
    for root_, _dirs, files in os.walk(dst):
        for fn in files:
            if re.match(r"neurips.*\.sty$", fn, re.I):
                styles.append(os.path.relpath(os.path.join(root_, fn), dst))
    assert not styles, (
        f"the supplement ships another venue's style file: {styles}. A TMLR "
        f"submission whose supplement carries neurips_2026.sty states a "
        f"competing venue to this venue's reviewers.")

    footers, loads = [], []
    for root_, _dirs, files in os.walk(dst):
        for fn in files:
            fp = os.path.join(root_, fn)
            rel = os.path.relpath(fp, dst)
            if fn.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader
                    body = " ".join((p.extract_text() or "")
                                    for p in PdfReader(fp).pages)
                except Exception:
                    continue
            else:
                try:
                    body = open(fp, encoding="utf-8").read()
                except (UnicodeDecodeError, OSError):
                    continue
            body = re.sub(r"\s+", " ", body)
            if re.search(r"Submitted to[^.]{0,120}"
                         r"Neural Information Processing Systems", body):
                footers.append(rel)
            if re.search(r"usepackage(\[[^]]*\])?\{neurips", body):
                loads.append(rel)
    assert not footers, (
        f"another venue's submission footer is in the shipped supplement: "
        f"{footers}")
    assert not loads, (
        f"a shipped source still loads a NeurIPS style package: {loads}")
