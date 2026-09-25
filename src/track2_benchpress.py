"""Re-test of the low-rank transfer claim after external review (Track 2b).

The reviewer (a BenchPress author) pointed out that (a) our rank-2 variance was
measured after filling half the matrix with a global mean, and (b) our imputer
was plain SVD completion, not BenchPress's method. This script answers both:

  A. rank-2 variance on the largest FULLY OBSERVED submatrices, columns
     mean-centred, as in the BenchPress paper (no filling);
  B. the same held-out protocol as track2_lowrank.py, run with (i) our original
     imputer and (ii) BenchPress's released Logit + Bias ALS predictor,
     unmodified, on 5 seeds, with and without the target's sibling rows;
  C. each benchmark's strongest correlated neighbour;
  D. choosing the 3 known scores by predictiveness instead of at random.

Needs a checkout of https://github.com/microsoft/benchpress (MIT), with its
data restored by `python -m benchpress.download_data`. Point BENCHPRESS_DIR at
it. Takes about 20 minutes on a laptop. Writes out/track2_benchpress.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import load  # noqa: E402
import track2_lowrank as T  # noqa: E402

DATA = os.path.join(HERE, "..", "data", "dataset.csv")
OUT = os.path.join(HERE, "..", "out", "track2_benchpress.json")
BP_DIR = os.environ.get("BENCHPRESS_DIR", os.path.join(HERE, "..", "..", "benchpress"))
SEEDS = 5
MIN_OVERLAP = 8       # pairs needed before a benchmark correlation is reported


def main():
    sys.path.insert(0, BP_DIR)
    from benchpress.methods.predictors import predict_benchpress_scores
    d = load(DATA)
    M = T.build_matrix(d)
    cols = list(M.columns)
    METRIC = {c: {"type": "pct", "range": [0.0, 100.0]} for c in cols}
    meta = d.groupby("model").agg(base_model=("base_model", "first"),
                                  scheme=("scheme", "first")).to_dict("index")
    sm_all = d.groupby("scheme").delta.mean()
    t0 = time.time()

    # ---- A
    A = {"filled_share": float(M.isna().to_numpy().mean()), "original_rank2": T.variance_explained(M)[1]}
    for scope, base_only in (("all", False), ("base", True)):
        for k in range(3, 7):
            n, cs, ve = T.fair_variance_explained(M, k, base_only=base_only)
            A[f"{scope}_k{k}"] = {"rows": n, "cols": cs, "rank2": ve}

    # ---- C
    def best_neighbours(sub):
        out = {}
        for a in cols:
            c = []
            for b in cols:
                if a != b:
                    ok = sub[[a, b]].dropna()
                    if len(ok) >= MIN_OVERLAP:
                        c.append((float(np.corrcoef(ok[a], ok[b])[0, 1]), b, len(ok)))
            out[a] = max(c) if c else None
        return out
    C = {"all_rows": best_neighbours(M),
         "base_rows": best_neighbours(M.loc[[r for r in M.index if r.startswith("BASE::")]])}
    base = M.loc[[r for r in M.index if r.startswith("BASE::")]]
    corr_rank = {}
    for a in cols:
        cs = [abs(np.corrcoef(*(base[[a, b]].dropna().to_numpy().T))[0, 1])
              for b in cols if a != b and len(base[[a, b]].dropna()) >= MIN_OVERLAP]
        corr_rank[a] = float(np.mean(cs)) if cs else 0.0

    # ---- B, D
    def imp_original(Mh):
        return T.soft_impute(Mh, 2)

    def imp_benchpress(Mh):
        return np.asarray(predict_benchpress_scores(Mh.to_numpy(float), metric=METRIC, benchmark_ids=cols))

    def run(imputer, seed, mode="random", drop_siblings=False, k=3):
        rng = np.random.default_rng(seed)
        recs = []
        for qrow in [r for r in M.index if r.startswith("QUANT::")]:
            mid = qrow.split("::", 1)[1]
            info = meta.get(mid)
            if info is None:
                continue
            brow = "BASE::" + info["base_model"]
            if brow not in M.index:
                continue
            obs = [b for b in M.columns if not np.isnan(M.loc[qrow, b]) and not np.isnan(M.loc[brow, b])]
            if len(obs) < k + 2:
                continue
            perm = list(rng.permutation(obs))
            rev = perm[:k] if mode == "random" else sorted(obs, key=lambda b: -corr_rank[b])[:k]
            hidden = [b for b in obs if b not in rev]
            Mh = M.copy()
            if drop_siblings:
                sib = [r for r in M.index if r.startswith("QUANT::") and r != qrow
                       and meta.get(r.split("::", 1)[1], {}).get("base_model") == info["base_model"]]
                Mh = Mh.drop(index=sib)
            Mh.loc[qrow, hidden] = np.nan
            pr = pd.DataFrame(imputer(Mh), index=Mh.index, columns=Mh.columns)
            for b in hidden:
                recs.append({"scheme": info["scheme"], "true_delta": M.loc[qrow, b] - M.loc[brow, b],
                             "pred_delta": pr.loc[qrow, b] - M.loc[brow, b]})
        return pd.DataFrame(recs)

    def score(r):
        mae = float(np.mean(np.abs(r.true_delta - r.pred_delta)))
        mae_s = float(np.mean(np.abs(r.true_delta - r.scheme.map(sm_all))))
        bad = r[r.true_delta <= -3]
        return {"n": int(len(r)), "mae": mae, "mae_scheme_mean": mae_s,
                "mae_zero": float(np.mean(np.abs(r.true_delta))),
                "mean_abs_pred": float(np.mean(np.abs(r.pred_delta))),
                "n_severe": int(len(bad)), "severe_caught": int((bad.pred_delta <= -1.5).sum()),
                "false_alarms": int(((r.true_delta > -3) & (r.pred_delta <= -1.5)).sum()),
                "corr_pred_true": float(np.corrcoef(r.pred_delta, r.true_delta)[0, 1])}

    B = []
    for seed in range(SEEDS):
        for name, imp in (("original", imp_original), ("benchpress", imp_benchpress)):
            for sib in (False, True):
                s = score(run(imp, seed, drop_siblings=sib))
                s.update(imputer=name, drop_siblings=sib, mode="random", seed=seed)
                B.append(s)
        print("seed", seed, round(time.time() - t0), "s", flush=True)
    for name, imp in (("original", imp_original), ("benchpress", imp_benchpress)):
        s = score(run(imp, 0, mode="predictive"))
        s.update(imputer=name, drop_siblings=False, mode="predictive", seed=0)
        B.append(s)

    def agg(imputer, sib, mode):
        v = [x for x in B if x["imputer"] == imputer and x["drop_siblings"] == sib and x["mode"] == mode]
        f = lambda k: float(np.mean([x[k] for x in v]))
        return {"runs": len(v), "n": v[0]["n"], "mae": f("mae"),
                "mae_sd": float(np.std([x["mae"] for x in v])), "mae_scheme_mean": f("mae_scheme_mean"),
                "mae_zero": f("mae_zero"), "ratio_vs_scheme_mean": f("mae") / f("mae_scheme_mean"),
                "mean_abs_pred": f("mean_abs_pred"), "n_severe": v[0]["n_severe"],
                "severe_caught": f("severe_caught"), "false_alarms": f("false_alarms"),
                "corr_pred_true": f("corr_pred_true")}
    summary = {f"{i}|sib={int(s)}|{m}": agg(i, s, m)
               for i, s, m in (("original", False, "random"), ("original", True, "random"),
                               ("benchpress", False, "random"), ("benchpress", True, "random"),
                               ("original", False, "predictive"), ("benchpress", False, "predictive"))}
    json.dump({"A": A, "C": C, "summary": summary, "runs": B, "seeds": SEEDS,
               "min_overlap": MIN_OVERLAP,
               "benchpress_commit": "a035bde (microsoft/benchpress, 2026-08-26)",
               "benchpress_predictor": "predict_benchpress_scores: Logit + Bias ALS, rank 2, lambda 0.1"},
              open(OUT, "w"), indent=1, default=str)
    print("wrote", OUT, round(time.time() - t0), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
