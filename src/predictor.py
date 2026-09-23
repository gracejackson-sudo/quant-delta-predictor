"""
The shipped predictor, chosen BY the diagnostics rather than in advance.

src/diagnose.py (Q1) showed that under leave-one-family-out, the only estimator
that beats "just guess the global mean delta" is a per-quantization-scheme mean.
Ridge and gradient boosting both do WORSE than the global mean. So the honest
predictor is a 6-cell lookup table, plus conformal intervals.

Calibration is Mondrian (scheme-conditional), because src/diagnose.py (Q4)
showed marginal calibration badly undercovers aggressive schemes (nvfp4 68%,
w4a16 78%) while overcovering safe ones (fp8 96%).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model import conformal_quantile, noise_scale


class SchemeMean:
    """Per-scheme mean delta, backing off to the global mean for unseen schemes."""

    name = "scheme_mean"

    def fit(self, dtr: pd.DataFrame):
        self.global_mean = float(dtr.delta.mean())
        self.by_scheme = dtr.groupby("scheme").delta.mean().to_dict()
        self.n_by_scheme = dtr.groupby("scheme").delta.size().to_dict()
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return np.array([self.by_scheme.get(s, self.global_mean)
                         for s in d.scheme], dtype=float)


class GlobalMean:
    name = "global_mean"

    def fit(self, dtr):
        self.mu = float(dtr.delta.mean())
        return self

    def predict(self, d):
        return np.full(len(d), self.mu)


class Conformal:
    """
    Split conformal with optional Mondrian (group-conditional) calibration and
    optional noise normalization.

    mondrian_by=None    -> one global quantile (marginal coverage only)
    mondrian_by='scheme'-> a quantile per scheme (approx. scheme-conditional
                           coverage, at the cost of smaller calibration sets)

    A group with too few calibration points for the requested alpha gets an
    infinite interval, which is reported rather than silently narrowed. Groups
    can optionally back off to the marginal quantile instead; we expose that as
    `backoff` and report how often it fires.
    """

    def __init__(self, alpha=0.10, mondrian_by=None, normalized=False,
                 backoff=True):
        self.alpha = alpha
        self.mondrian_by = mondrian_by
        self.normalized = normalized
        self.backoff = backoff

    def _scores(self, model, d):
        s = np.abs(d.delta.to_numpy(float) - model.predict(d))
        return s / noise_scale(d) if self.normalized else s

    def fit(self, model, dcal: pd.DataFrame):
        self.model = model
        s = self._scores(model, dcal)
        self.q_marginal = conformal_quantile(s, self.alpha)
        self.q_group, self.n_group = {}, {}
        if self.mondrian_by:
            for g, idx in dcal.groupby(self.mondrian_by).groups.items():
                sg = self._scores(model, dcal.loc[idx])
                self.q_group[g] = conformal_quantile(sg, self.alpha)
                self.n_group[g] = len(sg)
        return self

    def q_for(self, d):
        if not self.mondrian_by:
            return np.full(len(d), self.q_marginal), np.zeros(len(d), bool)
        q, fell_back = [], []
        for g in d[self.mondrian_by]:
            v = self.q_group.get(g, float("inf"))
            if not np.isfinite(v) and self.backoff:
                q.append(self.q_marginal)
                fell_back.append(True)
            else:
                q.append(v)
                fell_back.append(False)
        return np.array(q, float), np.array(fell_back, bool)

    def predict_interval(self, d: pd.DataFrame):
        yhat = self.model.predict(d)
        q, fell_back = self.q_for(d)
        half = q * noise_scale(d) if self.normalized else q
        return yhat, yhat - half, yhat + half, fell_back
