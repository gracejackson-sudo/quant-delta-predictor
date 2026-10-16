"""Commit hashes this repository's own history gates compare against.

SEPARATED SO THEY CAN BE DROPPED FROM THE SUPPLEMENT. A commit hash of ours
identifies this repository through GitHub's search, so none may ship in an
anonymous submission -- and these two cannot be relabelled either, because
they are arguments the code compares rather than prose a reader follows.
`src/build_supplement.py::DROP_FILES` drops this module, and the tests that
import it skip loudly when it is absent, which is the correct state: in a
checkout with no history there is nothing for them to gate.

Keeping them here rather than inline also keeps `tests/test_all.py` itself
shippable, so the supplement still carries the test suite.
"""

# Baseline for the Co-Authored-By trailer gate: the last commit before the
# standing instruction took effect. Commits after it must carry no trailer.
TRAILER_GATE_BASELINE = "f09b9f05f75a99880a99a5babe49376f943fbef4"

# Orphaned commits that documents cite ON PURPOSE, whose
# non-resolvability is the point of the citation. Each needs a one-line
# reason and a matching disclosure in the doc that cites it.
ORPHANS_ALLOWED = {
    # M1 disclosure: earlier version of the commit that superseded it, same
    # message, replaced by rebase/amend. Retained on the GitHub side by
    # hash; cited in EXTERNAL_FEEDBACK.md's "Where the value appears in
    # git" bullet.
    "1ba6159",
}
