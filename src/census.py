"""
COMPLETE SILENT-EXCLUSION CENSUS.

Walks every gate in the pipeline, in pipeline order, and reports exactly how
many models and rows each one drops and why -- including the gates that log
NOTHING today (non-whitelisted benchmarks, rows whose numeric-run length is not
2 or 3, duplicate-benchmark dedupe, and the section-narrowing window).

Emits:
  out/census_models.csv      one row per model card, with its fate
  out/census_rows.csv        one row per dropped table row, with the reason
  out/census.json            the aggregate counts
  ACCOUNTING.md              the human-readable full accounting
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import harvest as H  # noqa: E402

HERE = os.path.dirname(__file__)
CARDS = os.path.join(HERE, "..", "data", "cards")
PCARDS = os.path.join(HERE, "..", "data", "prospective_cards")
OUT = os.path.join(HERE, "..", "out")
MODELS_JSON = os.path.join(HERE, "..", "data", "redhatai_models.json")
MIN_ACC = 20.0

# The family list used to pick which cards to download in the first place.
CORPUS_FAMILY_FILTER = [
    "Meta-Llama-3.1-8B", "Meta-Llama-3.1-70B", "Meta-Llama-3.1-405B",
    "Llama-3.2-1B", "Llama-3.2-3B", "Llama-3.3-70B",
    "Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B",
    "Qwen2.5-14B", "Qwen2.5-32B", "Qwen2.5-72B",
    "Mistral-Small", "Mistral-7B", "Mixtral", "Phi-3", "gemma-2",
    "Qwen3-8B", "Qwen3-14B", "Qwen3-32B", "granite",
]
QUANT_RE = re.compile(
    r"(quantized\.w4a16|quantized\.w8a8|quantized\.w8a16|-FP8$|-FP8-dynamic$"
    r"|-INT8$|-NVFP4$|-quantized\.w4a16$)")


# ---------------------------------------------------------------- row census
def row_census(text):
    """
    Re-walk one card's tables counting EVERY row we look at and what happened
    to it, using the same primitives harvest.extract() uses.
    """
    c = collections.Counter()
    detail = []

    start = 0
    for marker in ("## Evaluation", "## Accuracy", "### Accuracy",
                   "## Model Evaluation", "### Evaluation"):
        i = text.find(marker)
        if i > 0:
            start = i
            break
    cut = len(text)
    for marker in ("## Inference Performance", "### Inference Performance",
                   "Inference Performance", "## Performance", "## Deployment"):
        i = text.find(marker, start + 1)
        if i > 0:
            cut = min(cut, i)
    body = text[start:cut]

    # how much of the card did the narrowing window throw away?
    c["chars_total"] = len(text)
    c["chars_in_window"] = len(body)
    out_tables = len(H.html_tables(text)) + len(H.md_tables(text))
    in_tables = len(H.html_tables(body)) + len(H.md_tables(body))
    c["tables_in_card"] = out_tables
    c["tables_in_window"] = in_tables
    c["tables_excluded_by_window"] = out_tables - in_tables

    seen = set()
    for table in H.html_tables(body) + H.md_tables(body):
        for cells in table:
            label, nums = H.parse_row(cells)
            c["rows_seen"] += 1
            if label is None or not nums:
                c["drop_header_or_nonnumeric"] += 1
                continue
            bench = H.canon_benchmark(label)
            if bench is None:
                c["drop_benchmark_not_whitelisted"] += 1
                lab = re.sub(r"\s+", " ", label.strip().strip("*"))[:44]
                detail.append(("benchmark_not_whitelisted", lab, len(nums)))
                continue
            if len(nums) not in (2, 3):
                c["drop_numeric_run_not_2_or_3"] += 1
                detail.append(("numeric_run_len_%d" % len(nums), bench,
                               len(nums)))
                continue
            if bench in seen:
                c["drop_duplicate_benchmark"] += 1
                detail.append(("duplicate_benchmark", bench, len(nums)))
                continue
            seen.add(bench)
            c["kept_candidate"] += 1
    return c, detail


def fate_of_card(path, model_id, allow_unknown_family):
    """Reproduce harvest_card's gate order, naming the gate that fired."""
    if not os.path.exists(path):
        return "card_not_downloaded", 0
    if os.path.getsize(path) < 200:
        return "gate1_empty_or_missing_card", 0
    text = open(path, encoding="utf-8", errors="replace").read()
    scheme, _, _, _ = H.parse_config(model_id)
    if scheme is None:
        return "gate2_no_recognized_scheme", 0
    if H.parse_params_b(model_id) is None:
        return "gate3_no_param_count_in_name", 0
    if H.parse_family(model_id) == "other" and not allow_unknown_family:
        return "gate4_unrecognized_family", 0
    parsed = list(H.extract(text))
    accs = [b for _, b, _, _ in parsed if b is not None]
    if accs and max(accs) <= 1.0:
        return "gate5_accuracy_on_0_1_scale", 0
    rows, rej = H.harvest_card(path, model_id,
                               allow_unknown_family=allow_unknown_family)
    if not rows:
        return "gate6_no_benchmark_rows_survived", 0
    return "KEPT", len(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}

    # ============================================== stage 0: corpus selection
    print("=" * 72)
    print("STAGE 0 -- CORPUS SELECTION (before any card was downloaded)")
    print("=" * 72)
    allm = [x["modelId"] for x in json.load(open(MODELS_JSON))]
    quant = [m for m in allm if QUANT_RE.search(m)]
    targeted = [m for m in quant
                if any(f.lower() in m.lower() for f in CORPUS_FAMILY_FILTER)]
    print(f"  RedHatAI models listed by the HF API      : {len(allm)}")
    print(f"  matching the quantization-suffix regex    : {len(quant)}")
    print(f"  also matching my hand-written family list : {len(targeted)}")
    print(f"  => SILENTLY EXCLUDED at selection time    : "
          f"{len(quant) - len(targeted)} quantized models "
          f"({100*(len(quant)-len(targeted))/len(quant):.0f}% of the "
          f"available corpus)")
    excl_fams = collections.Counter()
    for m in quant:
        if m not in set(targeted):
            n = m.split("/")[-1]
            n = re.split(r"-(?:quantized|FP8|INT8|NVFP4|W4A16|W8A8)", n)[0]
            excl_fams[re.sub(r"-?\d+(\.\d+)?[BbMm](-.*)?$", "", n)[:28]] += 1
    print("\n  largest excluded groups (never downloaded, never considered):")
    for k, v in excl_fams.most_common(15):
        print(f"    {k:<30} {v:>3}")
    report["stage0"] = {
        "models_listed": len(allm), "quantized": len(quant),
        "targeted": len(targeted),
        "silently_excluded": len(quant) - len(targeted),
        "top_excluded_groups": excl_fams.most_common(25),
    }

    # ============================================== stage 1: per-card gates
    print("\n" + "=" * 72)
    print("STAGE 1 -- PER-CARD GATES (training corpus, data/cards)")
    print("=" * 72)
    rows_models = []
    fates = collections.Counter()
    for fn in sorted(os.listdir(CARDS)):
        if not fn.endswith(".md"):
            continue
        mid = fn[:-3].replace("_", "/", 1)
        fate, n = fate_of_card(os.path.join(CARDS, fn), mid, False)
        fates[fate] += 1
        rows_models.append({"set": "training", "model": mid, "fate": fate,
                            "rows_kept": n})
    for fate, n in fates.most_common():
        print(f"  {fate:<36} {n:>4} models")
    print(f"\n  explicit lists of every dropped training model:")
    for fate in sorted(f for f in fates if f != "KEPT"):
        ms = [r["model"] for r in rows_models if r["fate"] == fate]
        print(f"\n  [{fate}] {len(ms)} models")
        for m in ms:
            print(f"      {m}")
    report["stage1_training_fates"] = dict(fates)

    # ============================================== stage 2: per-row gates
    print("\n" + "=" * 72)
    print("STAGE 2 -- PER-ROW GATES INSIDE SURVIVING CARDS")
    print("=" * 72)
    agg = collections.Counter()
    det = collections.Counter()
    dropped_rows = []
    for r in rows_models:
        if r["fate"] != "KEPT":
            continue
        p = os.path.join(CARDS, r["model"].replace("/", "_", 1) + ".md")
        text = open(p, encoding="utf-8", errors="replace").read()
        c, detail = row_census(text)
        agg.update(c)
        for why, what, k in detail:
            det[(why, what)] += 1
            dropped_rows.append({"set": "training", "model": r["model"],
                                 "reason": why, "label": what, "n_nums": k})
    print(f"  table rows examined                        : "
          f"{agg['rows_seen']}")
    print(f"    header / non-numeric rows (expected)     : "
          f"{agg['drop_header_or_nonnumeric']}")
    print(f"    benchmark not on the whitelist           : "
          f"{agg['drop_benchmark_not_whitelisted']}  <-- SILENT")
    print(f"    numeric-run length not 2 or 3            : "
          f"{agg['drop_numeric_run_not_2_or_3']}  <-- SILENT")
    print(f"    duplicate benchmark within a card        : "
          f"{agg['drop_duplicate_benchmark']}  <-- SILENT")
    print(f"    candidate rows passed to the gate        : "
          f"{agg['kept_candidate']}")
    print(f"\n  tables excluded by the section window      : "
          f"{agg['tables_excluded_by_window']}  <-- SILENT")
    print(f"  characters outside the section window      : "
          f"{agg['chars_total'] - agg['chars_in_window']} of "
          f"{agg['chars_total']}")

    print("\n  top silently-dropped benchmark labels "
          "(these are real published results we ignore):")
    nw = [(k[1], v) for k, v in det.items()
          if k[0] == "benchmark_not_whitelisted"]
    for lab, v in sorted(nw, key=lambda t: -t[1])[:22]:
        print(f"    {v:>4}x  {lab}")
    print("\n  other silent row drops:")
    for (why, what), v in sorted(det.items(), key=lambda t: -t[1]):
        if why != "benchmark_not_whitelisted":
            print(f"    {v:>4}x  {why:<26} {what}")
    report["stage2_row_gates"] = {k: int(v) for k, v in agg.items()}
    report["stage2_silent_labels"] = [
        [k[0], k[1], v] for k, v in sorted(det.items(), key=lambda t: -t[1])]

    # ============================================== stage 3: integrity gate
    print("\n" + "=" * 72)
    print("STAGE 3 -- RECOVERY INTEGRITY GATE (logged, but let's be explicit)")
    print("=" * 72)
    import csv as _csv
    rej = list(_csv.DictReader(open(os.path.join(
        HERE, "..", "data", "rejected_rows.csv"))))
    by = collections.Counter(x["reason"] for x in rej)
    for k, v in by.most_common():
        print(f"  {k:<28} {v:>4}")
    print("\n  every recovery_mismatch row in the training set:")
    for x in rej:
        if x["reason"] == "recovery_mismatch":
            print(f"      {x['model']:<62} {x['benchmark']}")
    report["stage3_gate"] = dict(by)

    # ============================================== stage 4: modelling filter
    print("\n" + "=" * 72)
    print("STAGE 4 -- MODELLING FILTER (model.load)")
    print("=" * 72)
    import csv as _csv2
    ds = list(_csv2.DictReader(open(os.path.join(
        HERE, "..", "data", "dataset.csv"))))
    low = [r for r in ds if float(r["acc_before"]) < MIN_ACC]
    print(f"  rows in dataset.csv                        : {len(ds)}")
    print(f"  dropped by acc_before < {MIN_ACC:g}                : "
          f"{len(low)}")
    bb = collections.Counter(r["benchmark"] for r in low)
    for k, v in bb.most_common():
        print(f"      {k:<16} {v:>3}")
    unver = [r for r in ds if r["verified"] == "0"]
    print(f"  rows whose sign is NOT machine-verified    : {len(unver)} "
          f"(kept, flagged)")
    print(f"  rows used for modelling                    : "
          f"{len(ds) - len(low)}")
    report["stage4"] = {"dataset_rows": len(ds), "dropped_low_acc": len(low),
                        "unverified_kept": len(unver),
                        "modelling_rows": len(ds) - len(low)}

    # ============================================== stage 5: prospective set
    print("\n" + "=" * 72)
    print("STAGE 5 -- PROSPECTIVE SET (data/prospective_cards)")
    print("=" * 72)
    from real_use_case import UNSEEN
    pfates = collections.Counter()
    for mid in UNSEEN:
        p = os.path.join(PCARDS, mid.replace("/", "_", 1) + ".md")
        fate, n = fate_of_card(p, mid, True)
        pfates[fate] += 1
        rows_models.append({"set": "prospective", "model": mid, "fate": fate,
                            "rows_kept": n})
    for fate, n in pfates.most_common():
        print(f"  {fate:<36} {n:>4} models")
    print("\n  every prospective model and its fate:")
    for r in rows_models:
        if r["set"] == "prospective":
            print(f"      {r['fate']:<36} {r['rows_kept']:>3} rows  "
                  f"{r['model']}")
    report["stage5_prospective_fates"] = dict(pfates)

    # ============================================== write outputs
    with open(os.path.join(OUT, "census_models.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["set", "model", "fate",
                                           "rows_kept"])
        w.writeheader()
        w.writerows(rows_models)
    with open(os.path.join(OUT, "census_rows.csv"), "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["set", "model", "reason", "label",
                                           "n_nums"])
        w.writeheader()
        w.writerows(dropped_rows)
    with open(os.path.join(OUT, "census.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote out/census_models.csv ({len(rows_models)} models), "
          f"out/census_rows.csv ({len(dropped_rows)} dropped rows), "
          f"out/census.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
