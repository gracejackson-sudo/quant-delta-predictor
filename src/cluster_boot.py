"""Cluster-bootstrap bounds for a coverage rate, seed-free where it matters.

A checkpoint-level bootstrap over few clusters has a discrete distribution, and
its 5th percentile can sit on a boundary between atoms. A Monte-Carlo estimate
with linear interpolation then moves by several points from seed to seed (and
across numpy versions). With 8 or fewer clusters we therefore enumerate every
multiset of clusters with its exact multinomial probability and take the
'lower' quantile (smallest value whose cumulative probability reaches the
level), so the result does not depend on any random stream.
"""
from __future__ import annotations
import itertools, math
import numpy as np

EXACT_MAX = 8


def bounds(ok_sums, ns, lo=0.05, hi=0.95, rng=None, draws=20000):
    """ok_sums/ns: per-cluster covered count and row count. Returns (lo, hi) as
    percentages of covered rows."""
    ok = np.asarray(ok_sums, float)
    n = np.asarray(ns, float)
    k = len(ok)
    if k <= EXACT_MAX:
        vals, probs = [], []
        for comp in itertools.combinations_with_replacement(range(k), k):
            cnt = np.bincount(comp, minlength=k)
            probs.append(math.factorial(k) / np.prod([math.factorial(int(c)) for c in cnt]) / k ** k)
            vals.append((cnt * ok).sum() / (cnt * n).sum())
        order = np.argsort(vals)
        v, p = np.asarray(vals)[order], np.cumsum(np.asarray(probs)[order])
        q = lambda a: v[min(np.searchsorted(p, a - 1e-12), len(v) - 1)]
        return 100 * q(lo), 100 * q(hi)
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, k, size=(draws, k))
    b = ok[idx].sum(1) / n[idx].sum(1)
    return (100 * np.percentile(b, 100 * lo, method="lower"),
            100 * np.percentile(b, 100 * hi, method="higher"))
