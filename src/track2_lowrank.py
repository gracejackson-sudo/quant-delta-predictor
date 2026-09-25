"""
TRACK 2 -- does a low-rank (BenchPress-style) score matrix help on PAIRED
quantization deltas, or does it collapse to predicting ~0 for everything?

BenchPress (arXiv:2606.24020) predicts an unseen (model, benchmark) score by
exploiting the fact that the frontier-model score matrix is approximately
rank-2. Our hypothesis was that this structure cannot separate a quantized
checkpoint from its base, because the two sit at almost the same point in a
low-rank space -- so it would predict delta ~ 0 everywhere and miss exactly
the damaging cases.

This is a FAST APPROXIMATION, not a reimplementation of BenchPress: we build
the same object (a model x benchmark score matrix), factor it at low rank,
and ask what delta it implies. Their method has extra machinery (link
functions, regularisation search, bias terms) that would sharpen point
accuracy but cannot change the structural question being asked here.

Protocol, per held-out quantized checkpoint:
  * reveal the FULL base-model row (what you already know before quantizing)
  * reveal k of the quantized model's scores (benchmarks you did run)
  * predict the remaining quantized scores
  * implied delta = predicted quantized score - known base score
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from model import load  # noqa: E402

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out")


def build_matrix(d):
    """Rows: base checkpoints AND quantized checkpoints. Cols: benchmarks."""
    base = (d.groupby(["base_model", "benchmark"]).acc_before.mean()
            .reset_index().rename(columns={"acc_before": "score"}))
    base["row"] = "BASE::" + base.base_model
    quant = (d.groupby(["model", "benchmark"]).acc_after.mean()
             .reset_index().rename(columns={"acc_after": "score"}))
    quant["row"] = "QUANT::" + quant.model
    allr = pd.concat([base[["row", "benchmark", "score"]],
                      quant[["row", "benchmark", "score"]]])
    M = allr.pivot_table(index="row", columns="benchmark", values="score")
    return M


def soft_impute(M, rank, iters=200, tol=1e-5):
    """Plain low-rank completion by iterative SVD on the observed entries."""
    X = M.to_numpy(float)
    mask = ~np.isnan(X)
    mu = np.nanmean(X)
    F = np.where(mask, X, mu)
    prev = None
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        s2 = s.copy()
        s2[rank:] = 0.0
        low = (U * s2) @ Vt
        F = np.where(mask, X, low)
        if prev is not None and np.linalg.norm(low - prev) / (
                np.linalg.norm(prev) + 1e-9) < tol:
            break
        prev = low
    return low


def variance_explained(M, max_rank=6):
    X = M.to_numpy(float)
    mask = ~np.isnan(X)
    F = np.where(mask, X, np.nanmean(X))
    # centre like BenchPress does (standardised scores)
    Fc = (F - F.mean(0)) / (F.std(0) + 1e-9)
    s = np.linalg.svd(Fc, compute_uv=False)
    ev = (s ** 2) / (s ** 2).sum()
    return [float(ev[:r].sum()) for r in range(1, max_rank + 1)]


def fair_variance_explained(M, k, rank=2, base_only=False):
    """Variance explained by `rank` factors on the largest fully observed
    submatrix with k benchmarks, each column mean-centred (the method of the
    BenchPress paper; suggested by its author on review). Returns
    (rows, columns, fraction). Unlike variance_explained() this fills nothing.
    """
    import itertools
    sub = M.loc[[r for r in M.index if r.startswith("BASE::")]] if base_only else M
    X = sub.to_numpy(float)
    ob = ~np.isnan(X)
    best = None
    for combo in itertools.combinations(range(X.shape[1]), k):
        n = int(ob[:, combo].all(1).sum())
        if best is None or n > best[0]:
            best = (n, combo)
    n, combo = best
    full = X[ob[:, combo].all(1)][:, combo]
    s = np.linalg.svd(full - full.mean(0), compute_uv=False)
    return n, [sub.columns[i] for i in combo], float((s[:rank] ** 2).sum() / (s ** 2).sum())


def best_neighbour(M, bench, min_overlap=8):
    """(correlation, other benchmark, overlap) of the benchmark most correlated
    with `bench`, using pairwise-complete rows with at least min_overlap pairs."""
    c = []
    for o in M.columns:
        if o != bench:
            ok = M[[bench, o]].dropna()
            if len(ok) >= min_overlap:
                c.append((float(np.corrcoef(ok[bench], ok[o])[0, 1]), o, len(ok)))
    return max(c) if c else None


def run(d, rank=2, k_revealed=3, seed=0):
    M = build_matrix(d)
    rng = np.random.default_rng(seed)
    recs = []
    quant_rows = [r for r in M.index if r.startswith("QUANT::")]

    meta = (d.groupby("model")
            .agg(base_model=("base_model", "first"),
                 scheme=("scheme", "first")).to_dict("index"))

    for qrow in quant_rows:
        mid = qrow.split("::", 1)[1]
        info = meta.get(mid)
        if info is None:
            continue
        brow = "BASE::" + info["base_model"]
        if brow not in M.index:
            continue
        obs = [b for b in M.columns if not np.isnan(M.loc[qrow, b])
               and not np.isnan(M.loc[brow, b])]
        if len(obs) < k_revealed + 2:
            continue
        perm = list(rng.permutation(obs))
        revealed, hidden = perm[:k_revealed], perm[k_revealed:]

        Mh = M.copy()
        Mh.loc[qrow, hidden] = np.nan          # hide the targets
        low = soft_impute(Mh, rank)
        pred = pd.DataFrame(low, index=M.index, columns=M.columns)

        for b in hidden:
            base_score = M.loc[brow, b]
            true_q = M.loc[qrow, b]
            recs.append({
                "model": mid, "scheme": info["scheme"], "benchmark": b,
                "base": base_score, "true_quant": true_q,
                "pred_quant": pred.loc[qrow, b],
                "true_delta": true_q - base_score,
                "pred_delta": pred.loc[qrow, b] - base_score,
            })
    return pd.DataFrame(recs), M


def main():
    d = load(DATA)
    M = build_matrix(d)
    print("=" * 74)
    print("TRACK 2 -- LOW-RANK STRUCTURE ON PAIRED QUANTIZATION DELTAS")
    print("=" * 74)
    print(f"\nscore matrix: {M.shape[0]} rows x {M.shape[1]} benchmarks, "
          f"{100*M.notna().mean().mean():.1f}% filled")
    ev = variance_explained(M)
    print("cumulative variance explained by rank r (standardised):")
    for r, e in enumerate(ev, 1):
        print(f"   rank {r}: {e*100:5.1f}%")

    out = {"matrix_rows": int(M.shape[0]), "matrix_cols": int(M.shape[1]),
           "fill_pct": float(100 * M.notna().mean().mean()),
           "variance_explained": ev, "by_rank": {}}

    for rank in (2, 3, 5):
        r, _ = run(d, rank=rank)
        if r.empty:
            continue
        mae = np.mean(np.abs(r.true_delta - r.pred_delta))
        # the scheme-mean baseline, our shipped predictor
        sm = d.groupby("scheme").delta.mean()
        base_mae = np.mean(np.abs(
            r.true_delta - r.scheme.map(sm).to_numpy()))
        zero_mae = np.mean(np.abs(r.true_delta))
        by_scheme = r.groupby("scheme").pred_delta.mean()
        spread = float(by_scheme.max() - by_scheme.min())
        true_spread = float(r.groupby("scheme").true_delta.mean().max()
                            - r.groupby("scheme").true_delta.mean().min())
        print(f"\n--- rank {rank} ---")
        print(f"  rows scored                    : {len(r)}")
        print(f"  MAE, low-rank implied delta    : {mae:.3f}pp")
        print(f"  MAE, scheme-mean (our tool)    : {base_mae:.3f}pp")
        print(f"  MAE, always predict 0          : {zero_mae:.3f}pp")
        print(f"  mean |predicted delta|         : "
              f"{np.mean(np.abs(r.pred_delta)):.3f}pp")
        print(f"  spread of predicted scheme means: {spread:.3f}pp "
              f"(true spread {true_spread:.3f}pp)")
        print("  predicted vs true mean delta, per scheme:")
        for s in sorted(by_scheme.index):
            t = r[r.scheme == s]
            print(f"     {s:<13} predicted {by_scheme[s]:+.3f}pp   "
                  f"true {t.true_delta.mean():+.3f}pp   n={len(t)}")
        out["by_rank"][str(rank)] = {
            "n": int(len(r)), "mae_lowrank": float(mae),
            "mae_scheme_mean": float(base_mae), "mae_zero": float(zero_mae),
            "mean_abs_pred_delta": float(np.mean(np.abs(r.pred_delta))),
            "pred_scheme_spread": spread, "true_scheme_spread": true_spread,
        }
        # does it catch the damaging cases?
        bad = r[r.true_delta <= -3.0]
        if len(bad):
            caught = int((bad.pred_delta <= -1.5).sum())
            print(f"  severe cases (true delta <= -3pp): {len(bad)}; "
                  f"low-rank predicted <= -1.5pp for {caught} "
                  f"({100*caught/len(bad):.0f}%)")
            out["by_rank"][str(rank)]["n_severe"] = int(len(bad))
            out["by_rank"][str(rank)]["n_severe_caught"] = caught

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "track2_lowrank.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}/track2_lowrank.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
