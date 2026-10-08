#!/usr/bin/env python3
"""Apply exact-string edits to a file, verifying each one landed.

WHY THIS EXISTS. Three times in this project a scripted multi-part edit
raised partway through -- an assertion on a string that did not match because
of line wrapping -- and exited before its single write at the end. The earlier
edits in the same script were lost silently, and once the loss was invisible:
src/build_envelope.py kept its in-sample construction, the rebuild produced
identical intervals, and only a before/after table showed nothing had moved.
The rule "write immediately after each replace, then re-read and check" is one
a person follows or forgets. This makes the tooling enforce it.

Guarantees, per edit, in order:
  1. the `old` string occurs exactly `count` times (default 1) -- so a
     near-miss from re-wrapped text fails loudly instead of matching the
     wrong place or nothing;
  2. the replacement is written to disk immediately, not batched;
  3. the file is re-read from disk and `new` is asserted present, and `old`
     asserted absent unless it is a substring of `new`;
  4. a failure at edit N leaves edits 1..N-1 applied AND REPORTED, so the
     caller knows the true state rather than assuming nothing happened.

Usage:
    from tools.patch import edit, edit_all
    edit("src/foo.py", "old text", "new text")
    edit_all("paper/main.tex", [(o1, n1), (o2, n2)])   # same file, in order

`edit_all` across several files that must stay in lockstep:
    for p in ("paper/neurips_main.tex", "paper/main.tex"):
        edit_all(p, SUBS)
and if one variant is wrapped differently, the failure names the file and the
string, which is the signal that the two have diverged.
"""
from __future__ import annotations

import os


class PatchError(RuntimeError):
    """Raised with the applied/failed boundary stated explicitly."""


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def edit(path: str, old: str, new: str, count: int = 1,
         applied: list | None = None) -> None:
    """Replace `old` with `new` in `path`, write, re-read, verify."""
    if not os.path.exists(path):
        raise PatchError(f"{path}: no such file")
    s = _read(path)
    seen = s.count(old)
    if seen != count:
        raise PatchError(
            f"{path}: expected {count} occurrence(s) of the anchor, found "
            f"{seen}.\n"
            f"  anchor starts: {old[:90]!r}\n"
            f"  If this is a .tex variant, the usual cause is different line "
            f"wrapping -- the two variants have diverged and need separate "
            f"anchors.\n"
            f"  Edits already applied in this run: "
            f"{applied if applied is not None else 'unknown'}")
    _write(path, s.replace(old, new))

    # re-read from disk: the point is to verify what is on disk, not in memory
    back = _read(path)
    if new not in back:
        raise PatchError(f"{path}: wrote the file but {new[:60]!r} is not in "
                         f"it on re-read. Do not proceed.")
    if old not in new and old in back:
        raise PatchError(f"{path}: anchor still present after the write "
                         f"({back.count(old)} time(s)). Do not proceed.")
    if applied is not None:
        applied.append(f"{path}: {old[:40]!r}")


def edit_all(path: str, subs, count: int = 1) -> list:
    """Apply `subs` in order to one file. Returns what landed.

    On failure the exception names the edits that already landed, so a caller
    never has to guess whether the file is half-edited.
    """
    applied: list = []
    for old, new in subs:
        edit(path, old, new, count=count, applied=applied)
    return applied
