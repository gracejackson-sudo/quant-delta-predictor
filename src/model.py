"""
Features, point predictor, and split-conformal calibration.

Kept deliberately simple (ridge) per RESEARCH.md section 3: with a
noise-dominated target and a few hundred clustered rows, regularized linear is
the right capacity, not a compromise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
# scikit-learn is imported lazily inside Predictor.fit(). The shipped CLI
# (src/rank.py) only needs load() from this module, so a user running the
# tool needs numpy and pandas and nothing else.

BENCH_LEVELS = [
    "mmlu", "mmlu_cot", "mmlu_pro", "arc_challenge", "gsm8k", "hellaswag",
    "winogrande", "truthfulqa", "ifeval", "bbh", "math_lvl5", "gpqa", "musr",
    "humaneval", "humaneval_plus", "arena_hard",
]
METHOD_LEVELS = ["gptq", "smoothquant", "smoothquant+gptq", "rtn", "awq", "unknown"]

MIN_ACC_BEFORE = 20.0  # drop near-random baselines (RESEARCH.md s.3)


def load(path, drop_near_random=True, verified_only=False):
    d = pd.read_csv(path)
    if verified_only:
        d = d[d.verified == 1]
    if drop_near_random:
        d = d[d.acc_before >= MIN_ACC_BEFORE]
    return d.reset_index(drop=True)


def noise_scale(d: pd.DataFrame) -> np.ndarray:
    """
    Analytic standard error of the DELTA, in percentage points, assuming the two
    evals were independent binomial samples over n_items:
        se = 100 * sqrt(2 * p(1-p)/n),  p = acc_before/100
    Independence is an upper bound (the paired evals share items), but only the
    relative scale across benchmarks matters for normalization.
    """
    p = np.clip(d.acc_before.to_numpy() / 100.0, 1e-3, 1 - 1e-3)
    n = d.n_items.to_numpy().astype(float)
    return 100.0 * np.sqrt(2.0 * p * (1 - p) / n)


def featurize(d: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols, names = [], []

    def add(vec, name):
        cols.append(np.asarray(vec, dtype=float).reshape(-1, 1))
        names.append(name)

    wb = d.weight_bits.to_numpy(float)
    ab = d.act_bits.to_numpy(float)

    # bit-width, nonlinear per Scaling Laws for Precision (RESEARCH.md thread B)
    add(np.log2(16.0 / wb), "log2_weight_compression")
    add(np.log2(16.0 / ab), "log2_act_compression")
    add(np.exp(-wb / 4.0), "exp_neg_weight_bits")
    add((ab < 16).astype(float), "acts_quantized")
    add((d.num_type == "int").astype(float), "num_type_int")
    add((d.scheme == "fp8_dynamic").astype(float), "act_scale_dynamic")

    # model scale
    add(np.log10(d.params_b.to_numpy(float)), "log10_params_b")
    add(d.is_instruct.to_numpy(float), "is_instruct")

    # task difficulty / headroom
    ab_ = d.acc_before.to_numpy(float)
    add(ab_ / 100.0, "acc_before")
    add(ab_ * (100.0 - ab_) / 10000.0, "headroom_var")
    add(noise_scale(d), "analytic_noise_pp")

    for b in BENCH_LEVELS:
        add((d.benchmark == b).astype(float), f"bench={b}")
    for m in METHOD_LEVELS:
        add((d.method == m).astype(float), f"method={m}")

    return np.hstack(cols), names


class Predictor:
    """Standardize -> ridge. Also carries the two dumb baselines."""

    def __init__(self, alpha=3.0):
        self.alpha = alpha

    def fit(self, dtr: pd.DataFrame):
        X, self.names = featurize(dtr)
        y = dtr.delta.to_numpy(float)
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler().fit(X)
        self.ridge = Ridge(alpha=self.alpha).fit(self.scaler.transform(X), y)
        self.global_mean = float(y.mean())
        self.bench_mean = dtr.groupby("benchmark").delta.mean().to_dict()
        return self

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        X, _ = featurize(d)
        return self.ridge.predict(self.scaler.transform(X))

    def predict_global_mean(self, d):
        return np.full(len(d), self.global_mean)

    def predict_bench_mean(self, d):
        return d.benchmark.map(
            lambda b: self.bench_mean.get(b, self.global_mean)
        ).to_numpy(float)

    def coefs(self):
        return sorted(
            zip(self.names, self.ridge.coef_), key=lambda t: -abs(t[1])
        )


# --------------------------------------------------------------- conformal
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    Split-conformal quantile with the finite-sample correction:
    the ceil((n+1)(1-alpha))-th smallest score. Returns +inf when n is too
    small to support the requested level (n < 1/alpha - 1), which is the honest
    answer rather than a too-narrow interval.
    """
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def calibrate(model: Predictor, dcal: pd.DataFrame, alpha: float,
              normalized: bool) -> float:
    resid = np.abs(dcal.delta.to_numpy(float) - model.predict(dcal))
    if normalized:
        resid = resid / noise_scale(dcal)
    return conformal_quantile(resid, alpha)


def interval(model: Predictor, dte: pd.DataFrame, qhat: float,
             normalized: bool):
    yhat = model.predict(dte)
    half = qhat * noise_scale(dte) if normalized else np.full(len(dte), qhat)
    return yhat, yhat - half, yhat + half


def evaluate(dte: pd.DataFrame, yhat, lo, hi):
    y = dte.delta.to_numpy(float)
    return {
        "n": int(len(y)),
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "mean_half_width": float(np.mean((hi - lo) / 2.0)),
        "median_half_width": float(np.median((hi - lo) / 2.0)),
        "mae": float(np.mean(np.abs(y - yhat))),
        "rmse": float(np.sqrt(np.mean((y - yhat) ** 2))),
    }


def r2(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
