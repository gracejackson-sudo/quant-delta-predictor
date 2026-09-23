# Standing audit discipline

**Rule: no change to the ranking, predictor, or calibration logic is done until it has been
adversarially audited. The audit is part of finishing the work, not a later request.**

Adopted because the posture "assume it overstates something until proven otherwise" caught a real
defect three rounds running:

| round | what the audit caught |
|---|---|
| 1 | `diffusiongemma-26B` matched into `gemma-2` by substring; that card also reported accuracy on a 0–1 scale |
| 2 | Llama-4 cards order columns `[Recovery, base, quant]`, producing a fabricated −68pp delta; `MATH-500` silently matched as `Math-Lvl-5` |
| 3 | `fit_calibrated()` measured residuals around the stratum mean while intervals centre on the scheme mean, mis-sizing every calibrated width |

Each was found by *trying to break a result that looked clean*, not by testing that it worked.

## What an audit must cover

1. **Leakage and scope creep.** Did anything enter the predictor beyond what is claimed? Did any
   design choice get informed by held-out data? Check the edit history against train/test
   membership and disclose, as `PROVENANCE.md` does.
2. **Sample-size honesty.** Report N behind every number. Check threshold boundaries — a value
   that clears a cutoff by zero margin is a finding, not a pass.
3. **Failure modes in their new form.** The known blind spots (cannot exclude zero; large losses
   fall below the interval floor) re-appear disguised after each change. Look for them
   specifically.
4. **Independent recomputation.** Reimplement the changed logic from scratch, sharing no code, and
   diff every displayed number. `verify/independent_rank.py` and `verify/independent_check.py`
   are the pattern; this is what caught round 3.
5. **Does the change do anything?** A guard that demotes nothing is decorative. State the count.
6. **Are new thresholds derived or chosen?** Say which. A judgement call is fine; presenting one
   as derived is not.

## Enforcement

```bash
./.venv/bin/python src/audit_ranking.py        # R1-R5 adversarial audit
./.venv/bin/python src/adversarial_audit.py    # headline-result audit
./.venv/bin/python src/census.py               # silent-exclusion census
./.venv/bin/python src/verify_claims.py        # every doc number recomputed
python3 verify/independent_rank.py             # from-scratch rank rebuild
python3 verify/independent_check.py            # from-scratch coverage rebuild
./.venv/bin/python -m pytest tests -q
```

All seven must pass before a change counts as landed. `tests/test_all.py` runs the claims check
and the banned-prose check automatically.

## The claims rule

Every quantitative statement in `RANKING.md` traces to a computed value. `RANKING.md` is
**generated** by `src/gen_ranking_doc.py`; figures carry `<!-- claim: key = value -->` tags;
`src/verify_claims.py` recomputes all of them and fails on mismatch or on any tag missing from its
registry. Do not hand-edit numbers in that file.

## Known gap: FINDINGS.md is outside the claim gate

`verify_claims.py` and `audit_traceability.py` cover the five **generated** documents
(`RANKING.md`, `NEGATIVE_RESULT.md`, `BIAS_CORRECTION.md`, `TOOL_SUMMARY.md`).
`FINDINGS.md` is hand-written and covered by neither. It was verified by
a **manual line-by-line pass on 2026-09-22**, which caught three real defects: a pull-quote
asserting a claim the same document retracts, a stale test count (60 vs 113), and a missing
pointer to the measured left tail.

Scope of the gap, measured rather than estimated: as of 2026-09-22, **280 untraced numeric
literals** in `FINDINGS.md`, and the 293-key claims registry backs **none** of them. (Both
counts are a snapshot; re-measure with `src/audit_traceability.py` rather than trusting them.)

**This is not fixable by tagging.** The five gated documents are trustworthy because they are
generated -- the number and its tag come from one computation. Hand-adding tags to a hand-written
document produces a hand-typed number beside a hand-typed tag, which can be wrong together, and
the gate would then report "0 untraced" while verifying nothing. That is worse than the honest
current state.

The real fix is to generate `FINDINGS.md`'s data tables from `out/` the way `RANKING.md` is
generated. The inputs exist (`holdout_demo.json`, `diagnostics.json`, `final_*.csv`). Estimated
1-2 hours. Until then, treat every figure in `FINDINGS.md` as manual-checked-only.
