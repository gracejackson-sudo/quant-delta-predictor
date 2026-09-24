"""Refuse to start a download that could fill the disk.

Twice in one day a download ran the machine out of space: once installing a
CUDA-sized torch build, once pulling an eval dataset. Both were discovered
afterwards, by which point the disk was already full and even writing a log
line failed.

The fix is not to clean up better afterwards. It is to check before starting,
and to refuse rather than proceed, because a half-written cache on a full disk
is worse than no cache at all.
"""
from __future__ import annotations

import os
import shutil

# Leave this much headroom beyond whatever the caller asks for, so that a
# download which exactly fits still leaves the machine usable.
HEADROOM_GB = 1.0


def free_gb(path=None):
    """Free space in GB on the volume holding `path` (default: this repo)."""
    path = path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return shutil.disk_usage(path).free / (1024 ** 3)


def require_free_gb(need_gb, what="this download", path=None):
    """Raise SystemExit unless `need_gb` + headroom is available.

    Returns the free space so callers can log it.
    """
    have = free_gb(path)
    want = need_gb + HEADROOM_GB
    if have < want:
        raise SystemExit(
            f"REFUSING TO START: {what} needs about {need_gb:.1f}GB "
            f"(+{HEADROOM_GB:.0f}GB headroom = {want:.1f}GB) but only "
            f"{have:.1f}GB is free.\n"
            f"Free some space and re-run. Nothing has been downloaded."
        )
    return have


def report(what="download", need_gb=None, path=None):
    have = free_gb(path)
    msg = f"[disk] {have:.1f}GB free"
    if need_gb is not None:
        msg += f", {what} needs ~{need_gb:.1f}GB + {HEADROOM_GB:.0f}GB headroom"
    return msg
