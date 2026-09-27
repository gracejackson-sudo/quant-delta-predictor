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
