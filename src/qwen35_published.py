"""Published Qwen3.5 quantization deltas, kept SEPARATE from the main corpus.

Reads Red Hat's quantized Qwen3.5 model cards (the same publisher the corpus is
built from) through the unchanged `harvest.harvest_card` parser and writes
data/qwen35/published_rows.csv. It never touches data/dataset.csv: Qwen3.5 is a
different architecture (multimodal wrapper, hybrid linear/full attention) and a
different distillation recipe, so its rows are compared with Qwen2.5's, never
pooled with them. Raw cards are cached under out/qwen35_cards/ (gitignored) and
not redistributed; only the extracted numbers are written.
"""
from __future__ import annotations
import csv, json, os, re, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harvest  # noqa: E402
import diskguard  # noqa: E402

ROOT = os.path.join(HERE, "..")
LISTING = os.path.join(ROOT, "data", "redhatai_models.json")
CACHE = os.path.join(ROOT, "out", "qwen35_cards")
OUT_ROWS = os.path.join(ROOT, "data", "qwen35", "published_rows.csv")
OUT_REJ = os.path.join(ROOT, "data", "qwen35", "published_rejects.csv")
OUT_SUM = os.path.join(ROOT, "out", "qwen35_published_summary.json")
UA = {"User-Agent": "quant-delta-predictor/1.0 (research)"}
QWEN35 = re.compile(r"(?:^|/)Qwen3\.5-", re.I)
MOE = re.compile(r"-A\d+B\b", re.I)


def card(mid):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, mid.replace("/", "_") + ".md")
    if not os.path.exists(p) or os.path.getsize(p) < 200:
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/{mid}/raw/main/README.md", headers=UA)
            open(p, "wb").write(urllib.request.urlopen(req, timeout=25).read())
        except Exception:
            open(p, "w").close()
    return p


def repos():
    ids = [m["id"] for m in json.load(open(LISTING))
           if QWEN35.search(m["id"]) and harvest.parse_config(m["id"])[0]]
    return sorted(ids)


def main():
    diskguard.require_free_gb(0.05, "the Qwen3.5 card cache")
    rows, rejects, fetched = [], [], 0
    ids = repos()
    for mid in ids:
        p = card(mid)
        fetched += int(os.path.getsize(p) > 200)
        r, rej = harvest.harvest_card(p, mid, allow_unknown_family=True)
        for x in r:
            x["family"] = "qwen3.5"
            x["moe"] = int(bool(MOE.search(mid)))
        rows += r
        rejects += rej
    os.makedirs(os.path.dirname(OUT_ROWS), exist_ok=True)
    cols = ["model", "base_model", "family", "params_b", "moe", "is_instruct",
            "scheme", "weight_bits", "act_bits", "num_type", "method", "benchmark",
            "n_items", "acc_before", "acc_after", "delta", "orient_src", "verified"]
    with open(OUT_ROWS, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(OUT_REJ, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "benchmark", "reason"])
        w.writerows(rejects)
    summary = {"repos_listed": len(ids), "cards_fetched": fetched,
               "cards_with_rows": len({r["model"] for r in rows}),
               "rows": len(rows), "verified_rows": sum(r["verified"] for r in rows),
               "rejects": len(rejects), "by_scheme": {}}
    for s in sorted({r["scheme"] for r in rows}):
        d = [r["delta"] for r in rows if r["scheme"] == s]
        summary["by_scheme"][s] = {"n": len(d), "mean": round(sum(d) / len(d), 3),
                                   "worst": min(d)}
    json.dump(summary, open(OUT_SUM, "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT_ROWS} ({len(rows)} rows), {OUT_REJ} ({len(rejects)} rejects)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
