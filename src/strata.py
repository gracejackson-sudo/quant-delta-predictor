"""
SCOPE.md item #4: stratify the envelope by model size band (and flag MoE).

Why this exists: the audit found the unstratified envelope fails worst on small
models. The data agrees -- W4A16 damage is strongly size dependent:

    W4A16   <2B    mean -1.22pp   5th pct -4.46
            2-10B  mean -0.87pp   5th pct -3.40
            >10B   mean -0.36pp   5th pct -1.73

while near-lossless schemes show no such gradient. A single w4a16 interval of
[-2.87, +1.40] is therefore too narrow for a 1B model and too wide for a 70B
one.

Cells with thin support fall back to the scheme-level envelope rather than
emitting a confident number from 12 rows, and every estimate carries the level
it was actually computed at.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

ALPHA = 0.10
BANDS = ["<2B", "2-10B", ">10B", "unknown"]

# A stratified cell must clear BOTH: enough rows for the conformal index to
# exist with margin, and enough distinct checkpoints that it is not one model's
# quirk. nvfp4/2-10B has 13 rows but only ONE checkpoint, which is why the
# checkpoint test is here.
MIN_ROWS = 20
MIN_CHECKPOINTS = 3

MOE_RE = re.compile(r"(\d+x\d+b|a\d+(\.\d+)?b|[-_]\d+e[-_]|mixtral|moe)", re.I)


def size_band(params_b):
    if params_b is None or (isinstance(params_b, float) and np.isnan(params_b)):
        return "unknown"
    if params_b < 2:
        return "<2B"
    if params_b <= 10:
        return "2-10B"
    return ">10B"


def is_moe(base_model: str) -> bool:
    return bool(MOE_RE.search(str(base_model)))


def annotate(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["band"] = d.params_b.map(size_band)
    d["moe"] = d.base_model.map(is_moe)
    d["stratum"] = d.scheme + " | " + d.band
    return d


def conformal_q(vals, alpha=ALPHA):
    v = np.sort(np.asarray(vals, float))
    n = len(v)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(v[k - 1]) if k <= n else float("inf")


class StratifiedBaseline:
    """
    Per-(scheme, size band) mean + conformal half-width, backing off to
    scheme level, then global, whenever a cell is too thin to stand on its own.

    Every prediction reports which level produced it, so a caller can tell a
    well-supported number from a fallback.
    """

    def __init__(self, alpha=ALPHA, min_rows=MIN_ROWS,
                 min_checkpoints=MIN_CHECKPOINTS):
        self.alpha = alpha
        self.min_rows = min_rows
        self.min_ckpt = min_checkpoints

    def _cell(self, g):
        mu = float(g.delta.mean())
        return {
            "mean": mu,
            "half_width": conformal_q(np.abs(g.delta - mu), self.alpha),
            "n": int(len(g)),
            "n_checkpoints": int(g.base_model.nunique()),
            "n_families": int(g.family.nunique()),
            "worst": float(g.delta.min()),
            "best": float(g.delta.max()),
            "p05": float(g.delta.quantile(0.05)),
            "p95": float(g.delta.quantile(0.95)),
        }

    def fit(self, dtr: pd.DataFrame):
        d = annotate(dtr)
        self.global_ = self._cell(d)
        self.by_scheme = {s: self._cell(g) for s, g in d.groupby("scheme")}
        self.by_stratum, self.rejected = {}, {}
        for (s, b), g in d.groupby(["scheme", "band"]):
            c = self._cell(g)
            if (c["n"] >= self.min_rows
                    and c["n_checkpoints"] >= self.min_ckpt
                    and np.isfinite(c["half_width"])):
                self.by_stratum[(s, b)] = c
            else:
                self.rejected[(s, b)] = c
        self.moe_stats = self._cell(d[d.moe]) if d.moe.any() else None
        self.n_moe_checkpoints = int(d[d.moe].base_model.nunique())
        return self

    def fit_calibrated(self, dtr: pd.DataFrame, dcal: pd.DataFrame):
        """
        Proper split conformal: centres from `dtr`, interval widths from the
        residuals on a disjoint `dcal`.

        Fitting both on the same rows makes the widths in-sample and the
        resulting coverage optimistic -- measured at 94% for NVFP4 that way
        versus 74% when calibration is genuinely held out. Anything quoted to
        a user must come from this path.
        """
        self.fit(dtr)                       # centres + support counts
        c = annotate(dcal)
        # Residuals MUST be taken around the same centre that
        # predict_interval() will use, or the calibrated width is measured
        # about the wrong point. Subclasses that re-centre override _centre().
        c = c.assign(resid=np.abs(
            c.delta - [self._centre(s, b) for s, b in zip(c.scheme, c.band)]))

        for s, g in c.groupby("scheme"):
            if s in self.by_scheme and len(g) >= 9:
                self.by_scheme[s] = dict(self.by_scheme[s],
                                         half_width=conformal_q(g.resid,
                                                                self.alpha))
        keep = {}
        for (s, b), cell in self.by_stratum.items():
            g = c[(c.scheme == s) & (c.band == b)]
            if len(g) >= 9:
                q = conformal_q(g.resid, self.alpha)
                if np.isfinite(q):
                    keep[(s, b)] = dict(cell, half_width=q)
        self.by_stratum = keep
        if len(c) >= 9:
            self.global_ = dict(self.global_,
                                half_width=conformal_q(c.resid, self.alpha))
        return self

    def _centre(self, scheme, band):
        """The point predict_interval() will centre on for this cell."""
        return self.lookup(scheme, band)[0]["mean"]

    def lookup(self, scheme, band):
        """-> (cell, level) where level is 'stratum' | 'scheme' | 'global'."""
        c = self.by_stratum.get((scheme, band))
        if c is not None:
            return c, "stratum"
        c = self.by_scheme.get(scheme)
        if c is not None:
            return c, "scheme"
        return self.global_, "global"

    def predict_interval(self, d: pd.DataFrame):
        d = annotate(d)
        yhat, lo, hi, level = [], [], [], []
        for s, b in zip(d.scheme, d.band):
            c, lv = self.lookup(s, b)
            yhat.append(c["mean"])
            lo.append(c["mean"] - c["half_width"])
            hi.append(c["mean"] + c["half_width"])
            level.append(lv)
        return (np.array(yhat), np.array(lo), np.array(hi),
                np.array(level, dtype=object))


class SchemeOnlyBaseline(StratifiedBaseline):
    """The current shipped behaviour, for A/B comparison."""

    def fit(self, dtr):
        super().fit(dtr)
        self.by_stratum = {}          # force scheme-level for everything
        return self


class ConservativeStratified(StratifiedBaseline):
    """
    One-sided stratification: a size cell may only WIDEN the scheme-level
    interval, never narrow it.

    Full stratification was tested and rejected -- it narrows the >10B interval
    from 3.76pp to 2.92pp, and that extra confidence does not survive contact
    with unseen checkpoints (prospective coverage fell 90.3% -> 84.4%). But the
    small-model problem is real, so we keep the widening half and throw away
    the narrowing half. This can only ever raise coverage.

    The centre stays at the scheme mean; only the width can move.
    """

    def _centre(self, scheme, band):
        base = self.by_scheme.get(scheme) or self.global_
        return base["mean"]

    def predict_interval(self, d: pd.DataFrame):
        d = annotate(d)
        yhat, lo, hi, level = [], [], [], []
        for s, b in zip(d.scheme, d.band):
            base, _ = (self.by_scheme.get(s), None)
            if base is None:
                base = self.global_
            cell = self.by_stratum.get((s, b))
            hw, lv = base["half_width"], "scheme"
            if cell is not None and cell["half_width"] > base["half_width"]:
                hw, lv = cell["half_width"], "stratum-widened"
            yhat.append(base["mean"])
            lo.append(base["mean"] - hw)
            hi.append(base["mean"] + hw)
            level.append(lv)
        return (np.array(yhat), np.array(lo), np.array(hi),
                np.array(level, dtype=object))
