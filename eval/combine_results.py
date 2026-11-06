"""
combine_results.py — stitches together the separate eval/results/*.json
files (produced by running eval/evaluate.py once per experiment) into
one eval/experiment_report.json, and prints the same Markdown tables
run_all_experiments.py would have — without needing one long blocking
run. Use this after running each experiment separately, e.g.:

    python -m eval.evaluate --mode hybrid --tag main
    python -m eval.evaluate --mode bm25 --tag ablation_retrieval_bm25
    python -m eval.evaluate --mode dense --tag ablation_retrieval_dense
    python -m eval.evaluate --no-verification --tag ablation_no_verification
    python -m eval.evaluate --no-kg --tag ablation_no_kg

    python -m eval.combine_results

Any experiment whose results file is missing is skipped (its table row
just won't print) — so you can run this after only 2 or 3 experiments
finish, to check progress, and again later once the rest complete.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
REPORT_PATH = Path(__file__).parent / "experiment_report.json"

# (report_key, expected --tag filename)
EXPECTED = [
    ("main", "main"),
    ("retrieval_bm25", "ablation_retrieval_bm25"),
    ("retrieval_dense", "ablation_retrieval_dense"),
    ("verify_off", "ablation_no_verification"),
    ("kg_off", "ablation_no_kg"),
]


def load(tag):
    path = RESULTS_DIR / f"{tag}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["summary"]


def md_row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main():
    report = {}
    missing = []
    for key, tag in EXPECTED:
        s = load(tag)
        if s is None:
            missing.append(tag)
        else:
            report[key] = s

    if missing:
        print("Not found yet (run these when ready, then re-run this script):")
        for tag in missing:
            print(f"  - eval/results/{tag}.json")
        print()

    if "main" in report:
        report["retrieval_hybrid"] = report["main"]
        report["verify_on"] = report["main"]
        report["kg_on"] = report["main"]

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved combined report to {REPORT_PATH} ({len(report)} entries)\n")

    if {"retrieval_bm25", "retrieval_dense", "retrieval_hybrid"} <= report.keys():
        print("=" * 70)
        print("MARKDOWN — Table: Retrieval ablation (BM25 / Dense / Hybrid)")
        print("=" * 70)
        print(md_row("Retrieval mode", "Precision", "Hit Rate", "Citation Acc.", "Answer Score"))
        print(md_row("---", "---", "---", "---", "---"))
        for label, key in [("BM25-only", "retrieval_bm25"), ("Dense-only", "retrieval_dense"), ("Hybrid (RRF)", "retrieval_hybrid")]:
            s = report[key]
            print(md_row(label, s["retrieval_precision_avg"], s["retrieval_hit_rate"],
                          s["citation_accuracy_avg"], s["answer_keyword_score_avg"]))
        print()

    if {"verify_on", "verify_off"} <= report.keys():
        print("=" * 70)
        print("MARKDOWN — Table: Verification ablation")
        print("=" * 70)
        print(md_row("Verification", "Verif.-failure rate", "Out-of-scope correct-decline"))
        print(md_row("---", "---", "---"))
        for label, key in [("With verification", "verify_on"), ("Without verification", "verify_off")]:
            s = report[key]
            print(md_row(label, s["verification_failure_rate"], s["out_of_scope_correct_decline_rate"]))
        print()

    if {"kg_on", "kg_off"} <= report.keys():
        print("=" * 70)
        print("MARKDOWN — Table: Knowledge-graph ablation")
        print("=" * 70)
        print(md_row("Config", "Citation Acc.", "Answer Score", "Verif.-failure rate"))
        print(md_row("---", "---", "---", "---"))
        for label, key in [("With KG", "kg_on"), ("Without KG", "kg_off")]:
            s = report[key]
            print(md_row(label, s["citation_accuracy_avg"], s["answer_keyword_score_avg"], s["verification_failure_rate"]))
        print()

    if "main" in report:
        print("=" * 70)
        print("MARKDOWN — Table: Main results + latency")
        print("=" * 70)
        s = report["main"]
        print(md_row("Metric", "Value"))
        print(md_row("---", "---"))
        for k in ["retrieval_precision_avg", "retrieval_hit_rate", "citation_accuracy_avg",
                  "answer_keyword_score_avg", "verification_failure_rate",
                  "out_of_scope_correct_decline_rate", "latency_median_sec", "latency_stdev_sec",
                  "latency_n_samples"]:
            print(md_row(k, s.get(k)))


if __name__ == "__main__":
    main()
