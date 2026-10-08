# Submission pack status

> **Status: RED (intentionally). Do not upload as-is without a LaTeX rebuild.**

## Why the gate is red

The pack's four `LIVE 77873d9` stamps are behind the repo's public `main`.
That is a real staleness signal, not a bug:

| Pack artifact | Stamped hash | Public `main` HEAD |
|---|---|---|
| `01 - TMLR paper (anonymous, ready to upload, LIVE 77873d9).pdf` | `77873d9` (2026-09-27) | see `git rev-parse origin/main` |
| `02 - Supplementary material (anonymized zip, LIVE 77873d9).zip` | `77873d9` | same |
| `04 - TMLR submission form - answers to paste.md:12` | `77873d9` | same |
| `04 - TMLR submission form - answers to paste.md:19` | `77873d9` | same |
| `03 - Claims versus evidence.md` | **no stamp at all** | unwatched until 2026-10-08 |

## Item 03 carries no stamp and was watched by nothing

`03 - Claims versus evidence.md` has no `LIVE <hash>` marker, so the staleness
table above could not include it and `tests/test_pack.py` could not see it.
It is the most content-bearing item in the pack --- it maps every headline
claim to its evidence --- and as of 2026-10-08 it still asserts, against a
repository that has retracted all of them:

- the tuning claim that round item 4 retracted;
- prospective coverage of 119/131, from the in-sample band;
- three operative verdicts, where the paper now reports two;
- 9 of 17 cells, a count that predates the split-conformal change;
- the Gemma-3 comparison, withdrawn because it reversed with the band.

Its own "what would move" list predates the split-conformal change entirely,
so it does not even anticipate the correction that invalidated it.

**This file is now committed to the repository** (pack contents stay outside
git) so that a test can read it. The checklist living where no test could see
it is the same unwatched-check class as everything else in this cycle: the
information existed, it was written down honestly, and nothing mechanical
consumed it.

Required before upload: rebuild 01 and 02 from current `main`, rewrite 03
against the current claims, and stamp all four.

`tests/test_pack.py::test_submission_pack_matches_current_head` in
`gracejackson-sudo/quant-delta-predictor` reads the four stamps above, compares
them against `git rev-parse HEAD` in that repo, and fails when they diverge.
It is failing right now.

## Why we are not silently rebuilding

The four artifacts are:

* two PDFs (paper + supplementary) that need `pdflatex` / `latexmk` to rebuild
* one zip (the anonymized supplement) built by the pack-assembly pipeline
* one markdown answer-sheet that is trivially regeneratable

The `.tex` sources under `paper/` on `origin/main` are up-to-date. A rebuild
would need a LaTeX pass I have not run yet, and a green gate over stale binaries
is worse than an honest red one — the pack would upload cleanly and *say* it
matches HEAD while carrying whatever the last LaTeX run produced.

## What would make it green

1. Rebuild both PDFs from the current `paper/main.tex` and `paper/neurips_main.tex`
   at HEAD (my recommended path is Overleaf, since no `pdflatex` is installed
   on the working machine — bundle already staged at
   `/private/tmp/claude-501/-Users-grace-Downloads/6e46ff93-9307-4cee-b215-b7e521c7491f/scratchpad/overleaf_bundle/`).
2. Regenerate `02 - Supplementary material (anonymized zip, LIVE <sha>).zip`
   from the pack-assembly script at the same commit (`src/build_supplement.py`).
3. Rename the two archive filenames to carry the new `LIVE <sha>` short hash.
4. Regenerate the `04 - TMLR submission form - answers to paste.md` content
   (its `LIVE <sha>` occurrences at lines 12 and 19 come from the same
   template).
5. Re-run `pytest tests/test_pack.py::test_submission_pack_matches_current_head`
   with `QDP_TMLR_PACK=<path to the TMLR-submission-pack folder>` (the pack
   lives outside this repository; the absolute path was redacted when this
   file was committed, because the tracked tree must carry no personal paths)
   set. Expect green.

The other pack test (`test_submission_pack_agrees_on_a_single_commit`) checks
that all four stamps *agree with each other*, and passes as long as the four
match — even at a stale hash. That test alone is not a green light to upload.
Both tests need to pass, and the second one must match `origin/main`.

## Last verified

`77873d9` was the state on 2026-09-27 12:40 −0700. **The paper's live figures
have moved substantively since**, not only in bookkeeping. A rebuilt PDF at
current HEAD would show:

* `\AdvBadOver`: 16 → **15** (Tier 1.2 strict `<` for the severe-loss
  comparator moved one boundary-case adversarial row out of the ≥3pp bucket)
* `\SevNvfp`: 15.6 → **14.1%** (same Tier 1.2 fix, per-scheme)
* `\PooledOneSided`: 95.8 → **95.7%** (Definition B family merge)
* `\BandEmpCov`: 93.4 → **91.0%**
* `\BandConfCov`: 89.6 → **90.6%**
* `\InfShare`: 43 → **26** (`\InfSharePct` = **25.8%**)
* `\MaeRidge`: 0.7639 → **0.7789** (the +0.0125 ridge move decomposed under
  Step 3, driven mainly by the Definition-B family merge)
* Every other MAE key moved in the third or fourth decimal (C1+C2 regen)
* ~50 new macros added: llama-3 subgroup coverage cells, LOFO fallback rates,
  the prospective gated-cohort breakdown, per-fold pred_mae, `\GpqaOtherRows`,
  `\MaxBenchmarksPerRun`, `\LThreeFoldRows`, per-fold row counts, etc.

Also new since `77873d9`: the §5 independent-verification disclosure
paragraph, the M1 IP disclosure rewrite, the H2/H3/M3 corrections in
EXTERNAL_FEEDBACK.md, the BIAS_CORRECTION.md task-1 wording fix, the
BENCH_LEVELS-class sweep, and the two new independent verifiers (noise
floor and cluster bootstrap).

**A rebuild is a substantive re-verification, not bookkeeping.** Anyone
comparing the `77873d9` pack against the current HEAD is comparing two
different papers, not two builds of the same one.
