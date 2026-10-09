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

## Scoped, not built: two classes of defect no gate here can see

Every gate in this repository verifies a **value**: that a tagged figure
equals the computed one, that no literal is untraced, that an artifact is not
older than its input. Three times in October 2026 a change was carried into
the registry and the macros but not into the prose that interprets them, and
no gate fired, because nothing in the paper's *values* was wrong. The two
classes below are what those failures were. Neither is addressed, and the
costs below are the reason to decide deliberately rather than drift.

### Class A: prose asserts a relation the registry contradicts

The §5 clustering paragraph said the cluster-robust standard error was
"smaller" than the naive one, that clustering "does not inflate this
statistic", and that the change of unit "accounts for the remaining"
narrowing. The macros beside those words read 2.10 against 1.94, a ratio of
1.084, a design effect of 1.112, and a unit term of −0.64. Every clause was
backwards. Separately, §5 called W4A16 "below the nominal 90%" at a macro
value of 97.6%.

In both cases the numbers were correct, registered, and verified. The
sentence around them was false.

**Candidate: a directional-word gate.** For each tagged macro, find
comparative words within a window — smaller, larger, below, above, narrower,
wider, exceeds, falls short — and the other macro or constant they relate it
to, then check the relation against the registry.

What it would catch: both instances above, and the "narrower" claim in the
decomposition.

What it costs. Parsing "X is smaller than Y" out of LaTeX prose where X and Y
are macros, negations, and subordinate clauses is the hard part; a naive
version will mostly produce false positives on sentences like "smaller than
the floor would imply", and a false-positive-heavy gate gets disabled. The
honest estimate is a day to build, and the maintenance risk is that it becomes
a gate people route around. A cheaper 80% version: restrict it to sentences
containing exactly two macros and one comparative, and report rather than
fail, so it is a review aid and not a build break. That version would have
caught the clustering paragraph and the W4A16 sentence.

### Class B: a correct figure with the wrong object attached

We wrote that size stratification "costs 6.4 points of coverage and buys no
tightening". The figure was right. It measured *full* stratification, a
variant discarded in September; the rule the artifact ships is widen-only,
which on the same rows gains two covered rows and loses none.

No directional check sees this. Nothing in the sentence is false about the
number it names. The number is correctly computed, correctly registered and
correctly quoted — it simply is not the number the claim requires. The defect
is in the correspondence between a key and the object a sentence is about,
which lives in the registry's schema rather than in the prose.

**Candidate: object descriptions plus a quoting index.** Every registry key
already carries a one-line description of how it was computed. Extend it to
state *what object it measures* — "the widen-only rule, shipped" versus "full
stratification, discarded" — and emit, per key, the list of sites that quote
it. A re-read of that index is then one cheap pass over the whole paper: for
each key, does every quoting site mean the object the description names?

What it would catch: this instance, and the "at the scheme level" mislabel,
where the value was right and the label named a granularity when the
distinction was whether a size rule had been applied.

What it costs, and how to bound it. Hand-writing object descriptions for all
~360 keys in one sitting produces 360 descriptions of uneven quality, written
by someone who stopped caring around key 200. Instead: require a description
at registration for every *new* key, and backfill only the keys the paper
actually quotes --- the per-key index says which those are, and it is a much
smaller set. An unquoted key gets its description when something quotes it.
That turns a grind into a standing rule plus a bounded backfill. The
quoting index is cheap — macros are already resolved from keys, and the
dead-macro gate walks the same graph. The real cost is that the check is a
human pass over a generated index rather than an assertion, so it only works
if someone does it; its value is making the pass possible at all, which it
currently is not.

### The mechanism to build first: pin the prose to the value

Both candidates above try to *understand* the sentence. A third does not, and
covers more of what actually happened.

For every macro, record two things in a committed pin file: its current value,
and a hash of the sentence containing it. On regeneration, fail when a macro's
value changed and its enclosing sentence did not. The message is "this figure
moved and the sentence around it did not --- confirm the sentence still
holds", and the author either edits the sentence or re-pins.

No comparative parsing, no negation handling, no LaTeX semantics, no false
positives from subordinate clauses. The only parsing is sentence boundaries,
which `src/check_prose.py` already does.

Checked against the four defects of this cycle:

| defect | fires? |
| --- | --- |
| clustering paragraph: macros moved when the band was corrected, prose did not | yes |
| W4A16 below nominal: `ProspCovWfour` moved to 97.6%, "below the nominal 90%" stayed | yes |
| abstract mixing the two coverage figures: figures changed, labels did not | yes |
| stratification mis-attribution: the figure never moved, only the object the sentence was about | **no** |

Three of four, assertable, and the one it misses is Class B --- which is the
evidence that Class B is a schema problem rather than a prose problem, not a
gap in this mechanism. Its failure mode is a one-line re-pin rather than a
build break worth disabling, which is what makes it survivable.

Two things to get right if it is built. The pin file must be reviewable in a
diff, so a re-pin is visible to whoever reads the commit rather than buried in
a regenerated blob. And re-pinning must require a stated reason, or it becomes
a reflex --- the same failure as an allowlist that grows without justification.

### Why neither is built yet

Both were identified at the end of a cycle in which four blocking defects
were found by audit rather than by gate. Building a gate against the class
that just bit you, in the same week, with no interval to see whether the
diagnosis holds, is how a repository accumulates checks nobody trusts. The
decision to build should be taken against the next cycle's evidence, not
this one's.

What is recorded now is the distinction, because it determines the mechanism:
Class A is a property of prose and could be asserted; Class B is a property of
the schema and probably cannot be, and conflating them would produce a gate
that fires on the easy half and misses the half that actually shipped.

### A known limit of the supplement scrub, stated rather than assumed

The scrub checks extracted page text and (since 2026-10-08) the document
metadata. It **cannot** see a byline rendered as an image, inside a figure, or
otherwise rasterized, because there is no text to extract. Nothing in the
current build renders author information that way, and no check enforces that.
If a future build adds a logo, a scanned signature or a figure containing a
name, the scrub will pass it.
