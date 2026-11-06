"""
run_all_experiments.py — runs every experiment the reviewers asked for
in one command, and prints Markdown tables ready to paste into the
paper (main results, retrieval ablation, verification ablation, KG
ablation, latency).

This is a convenience wrapper around eval/evaluate.py — it does not
duplicate any pipeline logic. Each experiment is a separate, honest run
against the real pipeline; nothing here is simulated.

Usage:
    python -m eval.run_all_experiments
    python -m eval.run_all_experiments --repeats 3   # for real latency dispersion
"""

import argparse
import json
from pathlib import Path

from eval.evaluate import run_evaluation

REPORT_PATH = Path(__file__).parent / "experiment_report.json"


def md_row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main(repeats: int):
    report = {}

    print("\n#### 1/5 Main configuration (hybrid + KG + verification) ####\n")
    report["main"] = run_evaluation(mode="hybrid", use_kg=True, use_verification=True,
                                     repeats=repeats, tag="main")[0]

    print("\n#### 2/5 Retrieval ablation ####\n")
    report["retrieval_bm25"] = run_evaluation(mode="bm25", use_kg=True, use_verification=True,
                                               repeats=1, tag="ablation_retrieval_bm25")[0]
    report["retrieval_dense"] = run_evaluation(mode="dense", use_kg=True, use_verification=True,
                                                repeats=1, tag="ablation_retrieval_dense")[0]
    report["retrieval_hybrid"] = report["main"]  # hybrid == main config

    print("\n#### 3/5 Verification ablation ####\n")
    report["verify_off"] = run_evaluation(mode="hybrid", use_kg=True, use_verification=False,
                                           repeats=1, tag="ablation_no_verification")[0]
    report["verify_on"] = report["main"]

    print("\n#### 4/5 Knowledge-graph ablation ####\n")
    report["kg_off"] = run_evaluation(mode="hybrid", use_kg=False, use_verification=True,
                                       repeats=1, tag="ablation_no_kg")[0]
    report["kg_on"] = report["main"]

    print("\n#### 5/5 Done — building report ####\n")

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to {REPORT_PATH}\n")

    # ---- Markdown tables for the paper ----
    print("=" * 70)
    print("MARKDOWN — Table: Retrieval ablation (BM25 / Dense / Hybrid)")
    print("=" * 70)
    print(md_row("Retrieval mode", "Precision", "Hit Rate", "Citation Acc.", "Answer Score"))
    print(md_row("---", "---", "---", "---", "---"))
    for label, key in [("BM25-only", "retrieval_bm25"), ("Dense-only", "retrieval_dense"), ("Hybrid (RRF)", "retrieval_hybrid")]:
        s = report[key]
        print(md_row(label, s["retrieval_precision_avg"], s["retrieval_hit_rate"],
                      s["citation_accuracy_avg"], s["answer_keyword_score_avg"]))

    print("\n" + "=" * 70)
    print("MARKDOWN — Table: Verification ablation")
    print("=" * 70)
    print(md_row("Verification", "Verif.-failure rate", "Out-of-scope correct-decline"))
    print(md_row("---", "---", "---"))
    for label, key in [("With verification", "verify_on"), ("Without verification", "verify_off")]:
        s = report[key]
        print(md_row(label, s["verification_failure_rate"], s["out_of_scope_correct_decline_rate"]))

    print("\n" + "=" * 70)
    print("MARKDOWN — Table: Knowledge-graph ablation")
    print("=" * 70)
    print(md_row("Config", "Citation Acc.", "Answer Score", "Verif.-failure rate"))
    print(md_row("---", "---", "---", "---"))
    for label, key in [("With KG", "kg_on"), ("Without KG", "kg_off")]:
        s = report[key]
        print(md_row(label, s["citation_accuracy_avg"], s["answer_keyword_score_avg"], s["verification_failure_rate"]))

    print("\n" + "=" * 70)
    print("MARKDOWN — Table: Main results + latency")
    print("=" * 70)
    s = report["main"]
    print(md_row("Metric", "Value"))
    print(md_row("---", "---"))
    for k in ["retrieval_precision_avg", "retrieval_hit_rate", "citation_accuracy_avg",
              "answer_keyword_score_avg", "verification_failure_rate",
              "out_of_scope_correct_decline_rate", "latency_median_sec", "latency_stdev_sec",
              "latency_n_samples"]:
        print(md_row(k, s[k]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3,
                         help="Repeats for the MAIN config's latency median/stdev (default 3)")
    args = parser.parse_args()
    main(args.repeats)
