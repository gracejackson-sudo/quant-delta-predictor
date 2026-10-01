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

   *Status of the two verifiers in the shipped repository.* `verify/independent_rank.py` runs
   live against `data/dataset.csv` and is gated by
   `tests/test_all.py::test_verifiers_exit_zero`. `verify/independent_check.py` was run once,
   on 2026-09-27, against the raw card corpus; that run produced `out/independent_check.csv`
   (prospective rows only — training rows were used to build the envelope but no per-row
   training-parse verification was written). The card corpus is not committed (license and PII
   constraints; `PROVENANCE.md` 2026-09-27), so `verify/independent_check.py` skips cleanly in
   this repository and the training-row-count and per-label comparisons it now performs have
   not fired against the shipped data. What remains as the standing gate on the 2026-09-27
   output is `tests/test_all.py::test_independent_check_csv_matches_tolerance_recomputation`,
   which recomputes each row's `inside` flag from its stored `lo`, `hi` and `delta` under a
   $10^{-9}$ boundary tolerance. That is a tolerance-integrity check on a past run's frozen
   output, not a fresh independent parse; both facts belong in every disclosure that cites
   this verifier.
5. **Does the change do anything?** A guard that demotes nothing is decorative. State the count.
6. **Are new thresholds derived or chosen?** Say which. A judgement call is fine; presenting one
   as derived is not.

## Enforcement

```bash
./.venv/bin/python src/audit_ranking.py        # R1-R5 adversarial audit             [LIVE]
./.venv/bin/python src/adversarial_audit.py    # headline-result audit               [SKIPS in shipped repo]
./.venv/bin/python src/census.py               # silent-exclusion census             [SKIPS in shipped repo]
./.venv/bin/python src/verify_claims.py        # every doc number recomputed         [LIVE]
python3 verify/independent_rank.py             # from-scratch rank rebuild           [LIVE]
python3 verify/independent_check.py            # from-scratch coverage rebuild       [SKIPS in shipped repo]
./.venv/bin/python -m pytest tests -q
```

`adversarial_audit.py`, `census.py`, and `verify/independent_check.py` all need the raw
card corpus (`data/cards/` and/or `data/prospective_cards/`), which is not committed
(license and PII constraints; `PROVENANCE.md` 2026-09-27). Each exits 0 with a `SKIPPED`
banner in this repository so `test_verifiers_exit_zero` and `test_audit_scripts_exit_zero`
gate them uniformly; the actual audit output for those three lives in the frozen artifacts
those runs produced last time they had the corpus (`out/independent_check.csv`,
`ACCOUNTING.md`, `out/adversarial_report.json` etc.), and the shipped repo's tolerance /
recomputation gates check that those frozen outputs remain internally consistent. Running
any of the three against the raw card corpus is what actually re-verifies the parse; the
shipped repo cannot.

All seven must pass before a change counts as landed. `tests/test_all.py` runs the claims
check, the banned-prose check, and the two exit-zero gates automatically.

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

## Known gap: one gate can fail for reasons that are not defects

`paper/audit_paper.py` section 2 checks every `\cite` key against arXiv's
export API, comparing the bib entry's author surnames and title to arXiv's own
record. That check exists for a good reason: an arXiv id that resolves says
nothing about who wrote the paper, and a wrong author list in a bibliography is
the kind of error a reviewer notices and an automated check should not miss.

It is also the only gate step that depends on a third party being reachable and
willing. Running the audit repeatedly — which a submission build does, once per
paper variant — earns an `HTTP Error 429: Too Many Requests` from arXiv, and
the step then reports a warning. Because
`tests/test_all.py::test_both_paper_variants_pass_audit_paper_cleanly` requires
zero warnings, a rate-limited network turns a green gate red without any
change to the paper or the code.

**Observed 2026-09-30:** both variants reported
`could not compare bib authors against arXiv: HTTP Error 429`, reproducibly,
and the failure persisted across a 90-second pause. Verified pre-existing by
stashing the working changes and reproducing it at `HEAD`.

**Deliberately not fixed during the methodological round.** The two obvious
remedies are both decisions rather than repairs: caching the arXiv responses
means the check can pass against a stale snapshot, and downgrading the step so
a network failure is not a warning means the check can silently stop running.
Either is defensible and neither should be chosen under deadline pressure.

**What to do in the meantime.** Treat a 429 as a non-finding. The relevant
distinction is between "the author comparison disagreed", which is a real
failure and reports the mismatched surnames, and "the author comparison could
not run", which is this. A submission build should not be blocked by the
second, and the rest of the gate — claim verification, traceability, the doc
check, both verifiers, and the other three sections of the paper audit — is
unaffected and still runs.
