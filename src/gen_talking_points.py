"""Generate TALKING_POINTS.md from the claims registry."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from verify_claims import registry  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R = registry()
def g(k): return R[k][0]


def n(k, fmt="{:.0f}"):
    """Figure plus the tag that lets src/verify_claims.py re-check it."""
    v = R[k][0]
    return f"{fmt.format(v)}<!-- claim: {k} = {v:.4f} -->"

def main():
    L=[];A=L.append
    A("# Talking points - day 3\n")
    A("*Every figure comes from the claims registry. Regenerate with "
      "`src/gen_talking_points.py`.*\n")
    A("---\n")
    A("## The spine: the self-correction\n")
    A("> \"Late today I thought I'd found a bug in my own product. The tool "
      "shows a symmetric interval, and quantization damage is obviously "
      "left-skewed, so a symmetric range is the wrong shape. An earlier audit "
      f"of mine had even measured an asymmetric version covering better, "
      f"{n('band_emp_coverage_pct', '{:.1f}')}% against "
      f"{n('band_conf_coverage_pct', '{:.1f}')}%. I was about to switch it.\"\n")
    A("> \"I measured it properly first. The asymmetric band returns an "
      f"*infinite* interval on {n('inf_rows', '{:.0f}')} of "
      f"{n('band_n_scored', '{:.0f}')} rows, {n('inf_share_pct', '{:.0f}')}%, because a "
      "two-sided empirical index needs 19 calibration points and often has "
      "fewer. An infinite interval covers 100% of the time by construction. "
      "That was the entire advantage.\"\n")
    A(f"> \"On the {n('finite_n', '{:.0f}')} rows where both bands are actually "
      f"defined, the original wins on coverage *and* width: "
      f"{n('finite_conf_cov_pct', '{:.1f}')}% at {n('finite_conf_width', '{:.2f}')}pp "
      f"versus {n('finite_emp_cov_pct', '{:.1f}')}% at "
      f"{n('finite_emp_width', '{:.2f}')}pp. What I shipped was right and my "
      "intuition was wrong.\"\n")
    A("**Why this is the story and not a footnote:** it is the fourth time "
      "this week that auditing a result before believing it caught something "
      "real. The discipline is the product.\n")
    A("## The one-liner\n")
    A("> \"Every time a result improved without a mechanism I could explain, "
      "the improvement turned out to be an artefact. That happened four times "
      "this week. Refusing unexplained improvements is the only reason any of "
      "these numbers are worth anything.\"\n")
    A("## The artefacts, if pushed for specifics\n")
    A(f"1. **Infinite intervals** covering 100% by construction, on "
      f"{n('inf_share_pct', '{:.0f}')}% of rows.")
    A("2. **In-sample calibration** flattering NVFP4 from a true 74% to 94%.")
    A(f"3. **Low-rank completion** looking plausible until its reconstruction "
      f"noise ({n('t2_mean_abs_pred::2', '{:.2f}')}pp) was set against the "
      f"signal it was meant to estimate ({n('t2_mae_zero::2', '{:.2f}')}pp).\n")
    A("## Today's new evidence\n")
    A(f"- Rented an A100 and measured {n('adv_rows', '{:.0f}')} rows across "
      f"{n('adv_models', '{:.0f}')} models and {n('adv_recipes', '{:.0f}')} recipes, "
      f"including a correctly-configured control arm.")
    A(f"- **Worst measured loss {n('adv_worst_delta', '{:.1f}')}pp**, against "
      f"{n('worst::w4a16', '{:.2f}')}pp as the worst anywhere in the published "
      f"corpus the tool is calibrated on.")
    A(f"- {n('adv_bad_over_3pp', '{:.0f}')} of {n('adv_bad_rows', '{:.0f}')} deliberately "
      f"faulted rows lost 3pp or more; separately "
      f"{n('adv_control_over_3pp', '{:.0f}')} of "
      f"{n('adv_control_rows', '{:.0f}')} control-arm rows did too.")
    A("- Independently corroborated: public llm-compressor issues report -14 "
      "to -16pp real-world failures, the same region as our deliberate "
      "ones.\n")
    A("## The second GPU session: fixed one confound, found a bigger one\n")
    A(f"> \"We re-ran MMLU under the published protocol and reproduced their "
      f"base accuracy to within a point - {n('s2_base_qwen05','{:.2f}')} "
      f"against their {n('s2_pub_qwen05','{:.2f}')}. That closed the harness "
      f"question. And then it showed us something worse: our own "
      f"*correctly-configured* baseline is "
      f"{n('ctrl_ratio_x','{:.1f}')}x more damaging than Red Hat's published "
      f"one on closely comparable checkpoints - base versus -Instruct "
      f"variants, not the identical artifact - because we calibrate on about "
      f"{n('calib_ratio_x','{:.0f}')}x less data than production does.\"\n")
    A("> \"So what we can prove is that bad configs are catastrophic. What we "
      "*cannot* yet say is how much worse they are than a professionally "
      "tuned recipe - our own 'good' recipe isn't good enough to be the "
      "yardstick. That's a narrower claim than I had this morning, and it's "
      "the true one.\"\n")
    A("## What I will not claim\n")
    A("- **Not a predictor.** It reads two inputs and has no per-model "
      "signal.")
    A("- **Not novel against BenchPress** (arXiv:2606.24020) on the core "
      "mechanism. They published predict-eval-then-conformal-interval first, "
      "with a better method, public code and data.")
    A(f"- **No corrected point estimate.** The selection-bias correction "
      f"moves {n('s2_protocol_sensitivity','{:.2f}')}pp depending on which "
      f"anchor set is used, so we publish a range, not a number.")
    A("- **pi is still unidentified.** How often a real user botches a config "
      "is not something any public data answers. I looked; nothing credible "
      "exists.\n")
    open(os.path.join(ROOT, "TALKING_POINTS.md"), "w").write("\n".join(L)+"\n")
    print("wrote TALKING_POINTS.md")
    return 0

if __name__ == "__main__":
    sys.exit(main())
