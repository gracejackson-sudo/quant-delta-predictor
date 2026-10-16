"""Make the gate summary state passed / failed / skipped explicitly.

pytest prints its own tally, but the number that mattered for weeks was the
one nobody printed: how many tests did not run. Three tests returned silently
on a missing input and counted as passes, so "198 passed" described a suite in
which two tests had read nothing. This prints the three-way split as its own
line, and names the skips (pytest.ini adds -rs for the reasons).
"""
from __future__ import annotations


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    tr = terminalreporter
    n = {k: len(tr.stats.get(k, [])) for k in
         ("passed", "failed", "error", "skipped", "xfailed", "xpassed")}
    tr.write_sep("=", "GATE SUMMARY")
    tr.write_line(
        f"  passed {n['passed']}   failed {n['failed'] + n['error']}   "
        f"skipped {n['skipped']}"
        + (f"   xfailed {n['xfailed']}" if n["xfailed"] else "")
        + (f"   xpassed {n['xpassed']}" if n["xpassed"] else ""))
    if n["skipped"]:
        tr.write_line("  a skip means an input was absent, not that the "
                      "property holds; the reasons are listed above.")
    for rep in tr.stats.get("failed", []) + tr.stats.get("error", []):
        tr.write_line(f"  FAILED  {rep.nodeid}")


# ---------------------------------------------------------------------------
# REPOSITORY GATES DO NOT APPLY TO THE SHIPPED ARTIFACT.
#
# The supplement ships this test suite, and on 2026-10-15 a clean-room run of
# the zip reported 14 failed, 178 passed. Every one of the fourteen was an
# absent input rather than a defect: they read `paper/main.tex` and
# `paper/neurips_main.tex` (the submission is to TMLR, so only the TMLR
# source ships), or `tests/test_supplement.py` and `tests/_git_constants.py`
# (dropped, because they carry a path literal or a commit hash that must not
# ship), or `PACK_STATUS.md` (which lives in the submission pack, not the
# repository), or `tools/measure_percent_defect.py` (dropped: it passes
# commit hashes to git and needs history the zip has not got).
#
# A reviewer who runs the suite and sees fourteen red lines learns something
# false about the work. Making each test probe for its own inputs would
# scatter artifact-only conditionals through fourteen repository gates, so
# the decision is made once, here, with the list spelled out — the list is
# the documentation of which gates are repository-scoped.
#
# THE SKIPS ARE LOUD. pytest.ini sets -rs and the GATE SUMMARY above says a
# skip means an absent input, never a property that holds. In the repository
# `.git` is present and none of this fires, so these gates keep their full
# force where they belong.
_REPO_SCOPED = {
    # read both non-TMLR paper variants
    "test_both_paper_variants_pass_audit_paper_cleanly",
    "test_the_two_paper_variants_say_the_same_thing",
    "test_paper_section6_protocol_sentence_states_both_grounds",
    "test_max_benchmarks_per_run_paper_prose_matches_registry",
    "test_no_dead_macros_in_numbers_tex",
    "test_every_removed_span_matches_the_rule_that_removed_it",
    "test_the_real_audit_fails_on_a_numeral_hidden_behind_a_percent",
    "test_no_hand_typed_numbers_in_outward_facing_docs",
    "test_docs_are_cross_consistent",
    "test_the_error_record_is_complete_across_both_its_halves",
    # read test files or tools that are deliberately withheld
    "test_no_test_can_pass_without_checking_anything",
    "test_percent_defect_json_matches_a_fresh_measurement",
    "test_audit_scripts_exit_zero",
    # reads the submission pack's own status file
    "test_pack_status_names_every_pack_item",
}


def _in_shipped_artifact():
    """-> True when running from the extracted supplement, not the repo.

    `.git` is the signal. The repository always has it; the zip never does,
    and the clean-room test asserts that it never does, so this cannot be
    fooled by an artifact that happens to carry history.
    """
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return not os.path.isdir(os.path.join(root, ".git"))


def pytest_collection_modifyitems(config, items):
    if not _in_shipped_artifact():
        return
    import pytest
    mark = pytest.mark.skip(
        reason="repository-scoped gate: its inputs are deliberately not "
               "shipped in the supplement (the non-TMLR paper variants, the "
               "self-referential test files, the history tools, or the "
               "submission pack's status file). This is an absent input, not "
               "a property that holds — see tests/conftest.py::_REPO_SCOPED "
               "for why each one is withheld.")
    for item in items:
        if item.name.split("[")[0] in _REPO_SCOPED:
            item.add_marker(mark)
