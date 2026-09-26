#!/usr/bin/env python3
"""
FULLY INDEPENDENT re-implementation of the scheme ranking.

Shares nothing with src/rank.py or src/strata.py:
  * no project imports, no numpy/pandas/scipy -- standard library only
  * reads data/dataset.csv directly and recomputes every displayed number
  * re-derives the flag rules from RANKING.md's stated definitions rather than
    from the code

Then diffs its rank order and every number against `src/rank.py --json`.

Usage:  python3 verify/independent_rank.py
"""
import csv
import itertools
import json
import math
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "dataset.csv")
ALPHA = 0.10
MIN_ACC = 20.0
RISK_PP = 3.0
SEVERE_RATE = 0.02
THIN_ROWS = 80
THIN_FAMILIES = 4
POOR_COVERAGE = 0.85
MIN_CKPTS = 3            # checkpoints needed before coverage can be judged
BOOT_EXACT_MAX = 8       # up to this many clusters the bootstrap is enumerated exactly
BOOT_DRAWS = 200_000
AGGRESSIVE = {"w4a16", "nvfp4"}


# ---------------------------------------------------------------- stats, by hand
def quantile_conformal(vals, alpha):
    v = sorted(vals)
    n = len(v)
    k = math.ceil((n + 1) * (1 - alpha))
    return v[k - 1] if k <= n else float("inf")


def percentile(vals, p):
    """Linear-interpolated percentile, matching pandas' default."""
    v = sorted(vals)
    if not v:
        return float("nan")
    idx = (len(v) - 1) * p
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return v[int(idx)]
    return v[lo] * (hi - idx) + v[hi] * (idx - lo)


def boot_bounds(oks, ns, lo=0.05, hi=0.95, seed=20260925):
    """Checkpoint-cluster bootstrap 90% interval of a coverage rate, in %.

    Resamples WHOLE checkpoints (clusters), because rows from one checkpoint
    are correlated views of a single quantization run. With few clusters the
    distribution is discrete, so it is enumerated exactly (each multiset of
    clusters with its multinomial probability); otherwise Monte Carlo."""
    k = len(oks)
    if k <= BOOT_EXACT_MAX:
        vals = []
        for comp in itertools.combinations_with_replacement(range(k), k):
            cnt = [comp.count(i) for i in range(k)]
            prob = math.factorial(k) / math.prod(math.factorial(c) for c in cnt) / k ** k
            vals.append((sum(c * o for c, o in zip(cnt, oks)) / sum(c * n for c, n in zip(cnt, ns)), prob))
        vals.sort()
        cum, acc = [], 0.0
        for _, pr in vals:
            acc += pr
            cum.append(acc)

        def q(a):
            for (v, _), c in zip(vals, cum):
                if c >= a - 1e-12:
                    return v
            return vals[-1][0]
        return 100 * q(lo), 100 * q(hi)
    rng = random.Random(seed)
    idx = range(k)
    draws = []
    for _ in range(BOOT_DRAWS):
        pick = rng.choices(idx, k=k)
        draws.append(sum(oks[i] for i in pick) / sum(ns[i] for i in pick))
    draws.sort()
    n = len(draws)
    return (100 * draws[math.floor((n - 1) * lo)], 100 * draws[math.ceil((n - 1) * hi)])


def size_band(p):
    if p is None:
        return "unknown"
    if p < 2:
        return "<2B"
    if p <= 10:
        return "2-10B"
    return ">10B"


# ---------------------------------------------------------------- data
def read_rows():
    out = []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if float(r["acc_before"]) < MIN_ACC:
                continue
            try:
                pb = float(r["params_b"])
            except (ValueError, KeyError, TypeError):
                pb = None
            out.append({
                "rid": len(out),
                "scheme": r["scheme"], "family": r["family"],
                "base_model": r["base_model"], "delta": float(r["delta"]),
                "band": size_band(pb),
            })
    return out


def cell(rows):
    ds = [r["delta"] for r in rows]
    mu = sum(ds) / len(ds)
    return {
        "mean": mu,
        "half_width": quantile_conformal([abs(x - mu) for x in ds], ALPHA),
        "n": len(ds),
        "n_checkpoints": len({r["base_model"] for r in rows}),
        "n_families": len({r["family"] for r in rows}),
        "worst": min(ds),
        "p05": percentile(ds, 0.05),
        "severe_rate": sum(1 for x in ds if x <= -RISK_PP) / len(ds),
        "n_severe": sum(1 for x in ds if x <= -RISK_PP),
    }


def group(rows, key):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return g


# --------------------------------------------- the predictor, re-derived
def build_model(train):
    """Conservative stratification: a size cell may only WIDEN."""
    by_scheme = {s: cell(v) for s, v in group(train, lambda r: r["scheme"]).items()}
    by_stratum = {}
    for (s, b), v in group(train, lambda r: (r["scheme"], r["band"])).items():
        c = cell(v)
        if c["n"] >= 20 and c["n_checkpoints"] >= 3 and \
                math.isfinite(c["half_width"]):
            by_stratum[(s, b)] = c
    return by_scheme, by_stratum


def calibrate(by_scheme, by_stratum, cal):
    """Widths from held-out residuals; centres untouched."""
    bs = {k: dict(v) for k, v in by_scheme.items()}
    for s, v in group(cal, lambda r: r["scheme"]).items():
        if s in bs and len(v) >= 9:
            res = [abs(r["delta"] - bs[s]["mean"]) for r in v]
            bs[s]["half_width"] = quantile_conformal(res, ALPHA)
    st = {}
    for (s, b), c in by_stratum.items():
        v = [r for r in cal if r["scheme"] == s and r["band"] == b]
        if len(v) >= 9 and s in bs:
            q = quantile_conformal(
                [abs(r["delta"] - bs[s]["mean"]) for r in v], ALPHA)
            if math.isfinite(q):
                st[(s, b)] = dict(c, half_width=q)
    return bs, st


def interval(bs, st, scheme, band):
    base = bs.get(scheme)
    if base is None:
        return None
    hw = base["half_width"]
    c = st.get((scheme, band))
    if c is not None and c["half_width"] > hw:
        hw = c["half_width"]
    return base["mean"] - hw, base["mean"] + hw


def coverage_by_scheme(rows):
    """Strict: unseen test family AND a different unseen calibration family.
    Every test row is scored once per calibration family (7 with 8 families),
    so `scored_pairs` is 7x the real row count; `rows` is the real count."""
    fams = sorted({r["family"] for r in rows})
    recs = {}
    for tf in fams:
        for cf in fams:
            if cf == tf:
                continue
            te = [r for r in rows if r["family"] == tf]
            ca = [r for r in rows if r["family"] == cf]
            tr = [r for r in rows if r["family"] not in (tf, cf)]
            if not te or len(ca) < 19 or len(tr) < 50:
                continue
            bs, st = build_model(tr)
            bs, st = calibrate(bs, st, ca)
            for r in te:
                iv = interval(bs, st, r["scheme"], r["band"])
                if iv is None:
                    continue
                recs.setdefault(r["scheme"], []).append((
                    r["rid"], r["base_model"], tf,
                    int(iv[0] <= r["delta"] <= iv[1]),      # two-sided
                    int(r["delta"] >= iv[0])))               # one-sided: loss side
    out = {}
    for s, L in recs.items():
        pairs = len(L)
        by_ck = {}
        for _, ck, _, two, one in L:
            a, b, c = by_ck.get(ck, (0, 0, 0))
            by_ck[ck] = (a + one, b + 1, c + two)
        lo, hi = boot_bounds([v[0] for v in by_ck.values()], [v[1] for v in by_ck.values()])
        per = {}
        for _, _, fam, _, one in L:
            a, b = per.get(fam, (0, 0))
            per[fam] = (a + one, b + 1)
        per = {f: a / b for f, (a, b) in per.items()}
        out[s] = {
            "one_sided": sum(x[4] for x in L) / pairs,
            "two_sided": sum(x[3] for x in L) / pairs,
            "scored_pairs": pairs,
            "rows": len({x[0] for x in L}),
            "ckpts": len(by_ck),
            "boot_lo": lo, "boot_hi": hi,
            "worst_family": min(per.values()),
            "worst_family_name": min(per, key=per.get),
            "spread": max(per.values()) - min(per.values()),
        }
    return out


def coverage_state(cv):
    """The three-state rule, re-derived from RANKING.md's definition rather
    than from rank.py: too few checkpoints, or a bootstrap interval that
    straddles the line, is 'insufficient_evidence'; an interval wholly below
    it is 'refused'; otherwise 'trusted'. The checkpoint floor is checked first."""
    if cv["ckpts"] < MIN_CKPTS:
        return "insufficient_evidence"
    if cv["boot_lo"] < POOR_COVERAGE * 100 < cv["boot_hi"]:
        return "insufficient_evidence"
    if cv["one_sided"] < POOR_COVERAGE:
        return "refused"
    return "trusted"


# ---------------------------------------------------------------- flags
def flags_for(e, cov, band=None, moe=False):
    f = []
    if e["severe_rate"] > SEVERE_RATE:
        f.append("TAIL_RISK")
    if e["n"] < THIN_ROWS or e["n_families"] < THIN_FAMILIES:
        f.append("THIN_DATA")
    st = coverage_state(cov) if cov else "trusted"
    if st == "insufficient_evidence":
        f.append("INSUFFICIENT_EVIDENCE")
    elif st == "refused":
        f.append("UNDERCOVERED")
    if band == "<2B" and e["scheme"] in AGGRESSIVE:
        f.append("SIZE_RISK")
    if band == "<2B":
        f.append("SIZE_UNVALIDATED")
    if moe:
        f.append("MOE_UNVALIDATED")
    return f


def tier(f):
    if {"TAIL_RISK", "UNDERCOVERED", "INSUFFICIENT_EVIDENCE"} & set(f):
        return "C"
    return "B" if f else "A"


def main():
    rows = read_rows()
    print("=" * 74)
    print("INDEPENDENT RE-IMPLEMENTATION OF THE RANKING (stdlib only)")
    print("=" * 74)
    print(f"  rows: {len(rows)}   python {sys.version.split()[0]}")

    bs, st = build_model(rows)
    cov = coverage_by_scheme(rows)

    mine = {}
    for s, c in bs.items():
        e = dict(c, scheme=s)
        cv = cov.get(s)
        f = flags_for(e, cv)
        mine[s] = {
            "scheme": s, "mean": c["mean"], "half_width": c["half_width"],
            "lo": c["mean"] - c["half_width"], "hi": c["mean"] + c["half_width"],
            "worst_observed": c["worst"], "p05": c["p05"], "n": c["n"],
            "n_checkpoints": c["n_checkpoints"], "n_families": c["n_families"],
            "severe_rate": c["severe_rate"],
            "coverage_one_sided": cv["one_sided"] if cv else float("nan"),
            "coverage_two_sided": cv["two_sided"] if cv else float("nan"),
            "coverage_rows": cv["rows"] if cv else 0,
            "coverage_ckpts": cv["ckpts"] if cv else 0,
            "coverage_scored_pairs": cv["scored_pairs"] if cv else 0,
            "coverage_worst_family": cv["worst_family"] if cv else float("nan"),
            "coverage_boot90": [cv["boot_lo"], cv["boot_hi"]] if cv else [None, None],
            "coverage_state": coverage_state(cv) if cv else "trusted",
            "flags": f, "tier": tier(f),
        }

    order = sorted(mine.values(),
                   key=lambda e: (e["tier"], e["half_width"],
                                  -e["worst_observed"]))
    print(f"\n  {'#':<3}{'scheme':<13}{'tier':<6}{'half_w':>8}{'worst':>8}"
          f"{'sev%':>7}{'n':>6}{'fam':>5}{'loss-side':>10}{'boot 90%':>14}  flags")
    for i, e in enumerate(order, 1):
        print(f"  {i:<3}{e['scheme']:<13}{e['tier']:<6}{e['half_width']:>8.3f}"
              f"{e['worst_observed']:>8.2f}{e['severe_rate']*100:>6.1f}%"
              f"{e['n']:>6}{e['n_families']:>5}"
              f"{e['coverage_one_sided']*100:>9.1f}%"
              f"{'[%.1f, %.1f]' % tuple(e['coverage_boot90']):>14}  {','.join(e['flags'])}")

    # ---------------------------------------------------------------- compare
    print("\n" + "=" * 74)
    print("COMPARISON WITH src/rank.py")
    print("=" * 74)
    venv = os.path.join(ROOT, ".venv", "bin", "python")
    py = venv if os.path.exists(venv) else sys.executable
    res = subprocess.run([py, os.path.join(ROOT, "src", "rank.py"),
                          "--all", "--json"], capture_output=True, text=True)
    if res.returncode != 0:
        print("  could not run src/rank.py:", res.stderr[-400:])
        return 1
    theirs = {e["scheme"]: e for e in json.loads(res.stdout)["ranked"]}
    their_order = [e["scheme"] for e in json.loads(res.stdout)["ranked"]]
    my_order = [e["scheme"] for e in order]

    print(f"  my order    : {my_order}")
    print(f"  their order : {their_order}")
    print(f"  ORDER MATCHES: {my_order == their_order}")

    fields = [("mean", 1e-9), ("half_width", 1e-9), ("lo", 1e-9),
              ("hi", 1e-9), ("worst_observed", 1e-9), ("p05", 1e-6),
              ("n", 0), ("n_checkpoints", 0), ("n_families", 0),
              ("coverage_one_sided", 1e-9), ("coverage_two_sided", 1e-9),
              ("coverage_rows", 0), ("coverage_ckpts", 0),
              ("coverage_scored_pairs", 0), ("coverage_worst_family", 1e-9)]
    bad = []
    for s, e in mine.items():
        t = theirs.get(s)
        if t is None:
            bad.append((s, "missing", "", ""))
            continue
        for fname, tol in fields:
            a, b = e.get(fname), t.get(fname)
            if a is None or b is None:
                continue
            if abs(float(a) - float(b)) > tol:
                bad.append((s, fname, a, b))
        # the bootstrap bounds: exact where enumerated, Monte Carlo (0.4pp) otherwise
        for i, nm in enumerate(("lower", "upper")):
            if abs(e["coverage_boot90"][i] - t["coverage_boot90"][i]) > 0.4:
                bad.append((s, "boot90_" + nm, e["coverage_boot90"][i], t["coverage_boot90"][i]))
        if e["coverage_state"] != t["coverage_state"]:
            bad.append((s, "coverage_state", e["coverage_state"], t["coverage_state"]))
        if set(e["flags"]) != set(t["flags"]):
            bad.append((s, "flags", e["flags"], t["flags"]))
        if e["tier"] != t["tier"]:
            bad.append((s, "tier", e["tier"], t["tier"]))
        if abs(e["severe_rate"] - t["severe_rate_used"]) > 1e-9:
            bad.append((s, "severe_rate", e["severe_rate"],
                        t["severe_rate_used"]))

    print(f"\n  field-level disagreements: {len(bad)}")
    for s, f, a, b in bad[:30]:
        print(f"    {s:<13}{f:<22} mine={a}  theirs={b}")

    ok = (my_order == their_order) and not bad
    print("\n" + "=" * 74)
    print(f"INDEPENDENT VERDICT: full agreement = {ok}")
    print("=" * 74)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
