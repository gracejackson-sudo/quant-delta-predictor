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
