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
