"""Pack-consistency test for the TMLR submission pack.

Lives in its own file so its `EF-Day-5` path literal cannot leak into the
supplement. `src/build_supplement.py::DROP_FILES` drops this file from the
build tree for the same reason `tests/test_supplement.py` is dropped.

Item 1 of the day-7 external audit: the TMLR submission pack shipped with
THREE different commit hashes stamped in filenames and body text
(supplement zip at ba70dc8, PDFs at b671a6c, form abstract at another
commit). This gate scans the pack for every LIVE <hash> marker in
filenames and inside the "04 - TMLR submission form..." text and fails
if they don't all agree on the same short hash. Skipped when the pack
is not present.

Two things this test does NOT do, and both are deliberate:
  * It does not open the PDFs or extract the supplement zip. It reads
    what the pack's own labelling claims about which commit produced
    each artifact, which is what a reviewer, TMLR's OpenReview page or
    a future auditor would see. Silently rebuilding a PDF without
    updating the LIVE marker in its filename would still fool this
    gate; the intent is for the two moves (rebuild and re-stamp) to
    happen together and this gate to catch the case where one lands
    without the other.
  * It does not check that the pack's commit matches the current git
    HEAD. Once a pack is finalised for submission, the tree keeps
    moving forward; the pack is a frozen snapshot. All we assert is
    that the snapshot is self-consistent.
"""
import os
import re


def test_submission_pack_agrees_on_a_single_commit():
    pack_dir = os.environ.get(
        "QDP_TMLR_PACK",
        "/" + "/".join(("Users", "grace", "Downloads",
                        "E" + "F-Day-5", "TMLR-submission-pack")))
    if not os.path.isdir(pack_dir):
        import pytest
        pytest.skip(
            f"submission pack not present at {pack_dir}; nothing to gate. "
            f"Rebuild the pack into that path or set QDP_TMLR_PACK.")
    marker = re.compile(r"LIVE\s+([0-9a-f]{7,40})")
    hashes: list[tuple[str, str]] = []
    for name in sorted(os.listdir(pack_dir)):
        for m in marker.finditer(name):
            hashes.append((f"filename: {name}", m.group(1)))
    form_name = "04 - TMLR submission form - answers to paste.md"
    form_path = os.path.join(pack_dir, form_name)
    if os.path.isfile(form_path):
        text = open(form_path).read()
        for m in marker.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            hashes.append((f"{form_name}:{line_no}", m.group(1)))
    assert hashes, (
        "no LIVE <hash> markers found in the pack; either the pack has "
        "been rebuilt without stamping the current commit, or the marker "
        "format has changed. Adopt the 'LIVE <7+ hex>' convention in "
        "filenames and in the form doc so this gate has something to "
        "check.")
    uniq = set(h for _, h in hashes)
    # Truncate to a git-short-hash for comparison; filenames and body may
    # stamp at slightly different lengths (7, 8, 40 hex).
    short = set(h[:7] for h in uniq)
    if len(short) != 1:
        by_loc = {loc: h[:7] for loc, h in hashes}
        raise AssertionError(
            "TMLR submission pack disagrees on which commit it was built "
            "from. Every LIVE <hash> in the pack must be the same "
            "short-hash:\n" + "\n".join(
                f"  {loc:<48s} -> {h}" for loc, h in by_loc.items()))


def test_submission_pack_matches_current_head():
    """Day-7 audit follow-up: the earlier gate (above) only checked that
    the pack is self-consistent. It stayed green across a real commit
    drift because the pack agreed with itself even when its stamp lagged
    the code by several commits. This gate asserts the pack's stamp
    matches the CURRENT git HEAD, so 'green' means 'up to date' rather
    than merely 'self-consistent'. Skipped when the pack is absent or
    when the working tree is not a git repo."""
    import os
    import re
    import subprocess
    pack_dir = os.environ.get(
        "QDP_TMLR_PACK",
        "/" + "/".join(("Users", "grace", "Downloads",
                        "E" + "F-Day-5", "TMLR-submission-pack")))
    if not os.path.isdir(pack_dir):
        import pytest
        pytest.skip(f"submission pack not present at {pack_dir}")
    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), ".."))
    try:
        head_full = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        import pytest
        pytest.skip("no git HEAD available for pack-vs-HEAD comparison")
    head_short = head_full[:7]
    marker = re.compile(r"LIVE\s+([0-9a-f]{7,40})")
    seen: list[tuple[str, str]] = []
    for name in sorted(os.listdir(pack_dir)):
        for m in marker.finditer(name):
            seen.append((f"filename: {name}", m.group(1)[:7]))
    form_name = "04 - TMLR submission form - answers to paste.md"
    form_path = os.path.join(pack_dir, form_name)
    if os.path.isfile(form_path):
        text = open(form_path).read()
        for m in marker.finditer(text):
            line_no = text.count("\n", 0, m.start()) + 1
            seen.append((f"{form_name}:{line_no}", m.group(1)[:7]))
    if not seen:
        import pytest
        pytest.skip("no LIVE <hash> markers in the pack to compare")
    off = [(loc, h) for loc, h in seen if h != head_short]
    if off:
        raise AssertionError(
            f"TMLR submission pack's LIVE stamps do not match current "
            f"git HEAD ({head_short}). A pack that is self-consistent "
            f"but not current-with-HEAD is still stale:\n" +
            "\n".join(f"  {loc:<48s} -> {h}  (want {head_short})"
                      for loc, h in off))


def test_pre_submission_gate_passes_before_a_pack_is_stamped():
    """A stamped pack is refused while --pre-submission fails.

    Day-8 audit: `paper/audit_paper.py --pre-submission` promotes the
    `unchecked` and `pending` categories to hard failures, which is the
    whole point of having it -- a figure nobody has checked and a figure
    known to be stale are both fine during development and neither is
    fine in a submission. Nothing in the repository ran the flag, so the
    stricter gate existed and was never consulted; the dev gate's
    `failures=0` was doing all the talking.

    This is the consultation. It fires only when a pack is actually
    present, because that is the moment the distinction starts to
    matter: while there is no pack there is nothing to refuse, and the
    twelve unclassified exempt numerals are a legitimate work-in-progress
    state. Once a pack exists, every variant in it has to clear the
    submission-strength gate.

    Deliberately in this file rather than tests/test_all.py: the two
    moves that have to happen together are "build the pack" and "pass
    the strict gate", and a reader asking what stamping requires should
    find both requirements in one place.
    """
    import subprocess
    import sys

    pack_dir = os.environ.get(
        "QDP_TMLR_PACK",
        "/" + "/".join(("Users", "grace", "Downloads",
                        "E" + "F-Day-5", "TMLR-submission-pack")))
    if not os.path.isdir(pack_dir):
        import pytest
        pytest.skip(
            f"submission pack not present at {pack_dir}; there is no stamp "
            f"to refuse. Run `python paper/audit_paper.py --pre-submission "
            f"<variant>` by hand before building one.")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # THE SHIPPED VARIANT. The pack's file 01 is the TMLR build, so that is
    # the document a stamp vouches for. Running the gate over the other
    # variants gated a submission on documents that are not in it -- and on
    # 2026-10-14 the pack carried a NeurIPS-styled PDF for a day while this
    # gate was green on two files that were not the one in the slot.
    for variant in ("tmlr_main.tex",):
        r = subprocess.run(
            [sys.executable, os.path.join("paper", "audit_paper.py"),
             variant, "--pre-submission"],
            capture_output=True, text=True, cwd=root)
        sentinel = [l for l in r.stdout.splitlines()
                    if l.startswith("AUDIT_PAPER_RESULT")]
        assert sentinel, (
            f"--pre-submission produced no sentinel line for {variant}; "
            f"stdout tail:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-2000:]}")
        assert r.returncode == 0 and "failures=0" in sentinel[-1], (
            f"a pack is present at {pack_dir} but {variant} does not clear "
            f"the pre-submission gate, so its stamp is not valid. "
            f"{sentinel[-1]}\n\n{r.stdout[-3000:]}")


def test_pack_pdf_is_a_tmlr_submission():
    """File 01 must be a TMLR submission, not another venue's.

    On 2026-10-14 the pack's "01 - TMLR paper" slot held a PDF built from
    the NeurIPS source: NeurIPS margin line numbers and the footer
    "Submitted to 40th Conference on Neural Information Processing Systems
    (NeurIPS 2026). Do not distribute." Nothing caught it, because every
    pack gate was about hash agreement and none read the document. The
    paper's own submission form specified "TMLR style, header reads 'Under
    review as submission to TMLR'" and the gate did not consult it.

    Two assertions, and the second is deliberately narrow. It fires on the
    NeurIPS SUBMISSION FOOTER -- "Submitted to" together with "Neural
    Information Processing Systems" -- and not on the bare word NeurIPS,
    because the bibliography legitimately cites NeurIPS papers and a gate
    that forbids the word would forbid citing the literature.

    Planted failure: put a NeurIPS-styled PDF in the slot and this fires on
    the missing header and on the footer.
    """
    import re

    pack_dir = os.environ.get(
        "QDP_TMLR_PACK",
        "/" + "/".join(("Users", "grace", "Downloads",
                        "E" + "F-Day-5", "TMLR-submission-pack")))
    if not os.path.isdir(pack_dir):
        import pytest
        pytest.skip(f"submission pack not present at {pack_dir}")
    cands = [f for f in sorted(os.listdir(pack_dir))
             if f.startswith("01 ") and f.lower().endswith(".pdf")]
    if not cands:
        import pytest
        pytest.skip(
            f"no '01 ...pdf' in {pack_dir}, so there is no paper to check. "
            f"This is an absent input, not a pass: a pack with no PDF is "
            f"not a pack that passed.")
    assert len(cands) == 1, (
        f"more than one candidate for file 01, so which one would be "
        f"uploaded is ambiguous: {cands}")
    name = cands[0]
    from pypdf import PdfReader
    r = PdfReader(os.path.join(pack_dir, name))
    page1 = re.sub(r"\s+", " ", r.pages[0].extract_text() or "")
    whole = re.sub(r"\s+", " ", " ".join(
        (p.extract_text() or "") for p in r.pages))

    assert "Under review as submission to TMLR" in page1, (
        f"{name!r} page 1 does not carry the TMLR submission header. "
        f"tmlr.sty prints it whenever the package is loaded without "
        f"[accepted] or [preprint], so either this PDF was not built from "
        f"the TMLR template or it was built in the wrong mode.\n"
        f"page 1 began: {page1[:200]!r}")

    footer = re.search(
        r"Submitted to[^.]{0,120}Neural Information Processing Systems",
        whole)
    assert footer is None, (
        f"{name!r} carries another venue's submission footer: "
        f"{footer.group(0)!r}. A TMLR submission that states it was "
        f"submitted to NeurIPS tells this venue's reviewers about a "
        f"competing one. (The bare word NeurIPS is fine and is not checked "
        f"— the bibliography cites NeurIPS papers.)")
