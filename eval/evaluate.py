"""
evaluate.py — runs the ground-truth Q&A set through the real pipeline
and computes the metrics needed for the paper. Supports the ablation
switches requested in review (retrieval mode, KG on/off, verification
on/off) and repeated runs for latency median/dispersion.

  - Retrieval Precision   : of the chunks retrieved, how many are from
                             an expected source document?
  - Retrieval Hit Rate    : did retrieval find at least one chunk from
                             an expected source at all?
  - Citation Accuracy     : of the citations in the FINAL answer, how
                             many point to an expected source?
  - Answer Keyword Score  : rough keyword-overlap score between the
                             generated answer and expected keywords
                             (an approximation, not semantic scoring —
                             noted clearly in the paper as a limitation)
  - Verification-Failure Rate : fraction of in-scope questions where the
                             verification agent did NOT mark the answer
                             fully SUPPORTED. (Renamed from "Hallucination
                             Rate" per reviewer #5 — this measures verifier
                             disagreement, not directly-annotated hallucination.)
  - Out-of-scope Correct-Decline Rate : for questions with no relevant
                             document, did the system correctly decline
                             instead of answering?
  - Response Latency      : seconds per query, end-to-end, with
                             reasoning/verification sub-stage breakdown,
                             reported as median + stdev over --repeats runs.

Usage:
    python -m eval.evaluate                                   # default: hybrid + KG + verification
    python -m eval.evaluate --mode bm25                        # retrieval ablation, BM25 only
    python -m eval.evaluate --mode dense                       # retrieval ablation, dense only
    python -m eval.evaluate --no-kg                            # KG ablation
    python -m eval.evaluate --no-verification                  # verification ablation
    python -m eval.evaluate --repeats 3                        # latency median/stdev over 3 runs
    python -m eval.evaluate --tag my_run_name                  # custom output filename
    python -m eval.evaluate --limit 5 --tag quick_test          # only run the first 5 questions (fast sanity check)
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from src.agents import retrieval_agent
from src import pipeline

GROUND_TRUTH_PATH = Path(__file__).parent / "qa_ground_truth.json"
RESULTS_DIR = Path(__file__).parent / "results"


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH, "r") as f:
        return json.load(f)


def retrieval_precision(retrieved_chunks: list[dict], expected_sources: list[str]) -> float:
    if not retrieved_chunks:
        return 0.0
    hits = sum(1 for c in retrieved_chunks if c["metadata"].get("source") in expected_sources)
    return hits / len(retrieved_chunks)


def retrieval_hit(retrieved_chunks: list[dict], expected_sources: list[str]) -> bool:
    return any(c["metadata"].get("source") in expected_sources for c in retrieved_chunks)


def citation_accuracy(citations: list[dict], expected_sources: list[str]) -> float:
    if not citations:
        return 0.0
    hits = sum(1 for c in citations if c["source"] in expected_sources)
    return hits / len(citations)


def keyword_overlap_score(answer: str, expected_keywords: list[str]):
    if not expected_keywords:
        return None
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def run_single_pass(ground_truth, mode, use_kg, use_verification, top_k=5):
    """One full pass over the ground-truth set; returns per-question rows."""
    results = []
    for item in ground_truth:
        query = item["query"]
        expected_sources = item["expected_sources"]
        expected_keywords = item["expected_keywords"]
        category = item["category"]

        t_total = time.time()
        retrieval_t0 = time.time()
        retrieval_result = retrieval_agent.retrieve(query, mode=mode, use_kg=use_kg)
        retrieval_time = time.time() - retrieval_t0

        final_result = pipeline.ask(
            query, session_id=f"eval_{item['id']}",
            mode=mode, use_kg=use_kg, use_verification=use_verification,
        )
        total_latency = time.time() - t_total

        r_precision = retrieval_precision(retrieval_result["evidence_chunks"], expected_sources)
        r_hit = retrieval_hit(retrieval_result["evidence_chunks"], expected_sources)
        c_accuracy = citation_accuracy(final_result.get("citations", []), expected_sources)
        a_score = keyword_overlap_score(final_result["answer"], expected_keywords)
        verdict = final_result.get("verification", {}).get("verdict", "UNKNOWN")
        declined = final_result.get("declined", False)
        stage_timing = final_result.get("timing", {})

        results.append({
            "id": item["id"],
            "query": query,
            "category": category,
            "expected_sources": expected_sources,
            "retrieval_precision": round(r_precision, 3),
            "retrieval_hit": r_hit,
            "citation_accuracy": round(c_accuracy, 3),
            "answer_keyword_score": round(a_score, 3) if a_score is not None else None,
            "verification_verdict": verdict,
            "declined_to_answer": declined,
            "latency_sec": round(total_latency, 3),
            "reasoning_sec": stage_timing.get("reasoning_sec"),
            "verification_sec": stage_timing.get("verification_sec"),
            "retrieval_sec": round(retrieval_time, 3),
            "answer_preview": final_result["answer"][:150],
        })
    return results


def summarize(results: list[dict]) -> dict:
    in_scope = [r for r in results if r["category"] != "out_of_scope"]
    out_scope = [r for r in results if r["category"] == "out_of_scope"]

    avg = lambda vals: round(sum(vals) / len(vals), 3) if vals else 0.0

    retrieval_precisions = [r["retrieval_precision"] for r in in_scope]
    hit_rate = avg([1 if r["retrieval_hit"] else 0 for r in in_scope])
    citation_accuracies = [r["citation_accuracy"] for r in in_scope]
    answer_scores = [r["answer_keyword_score"] for r in in_scope if r["answer_keyword_score"] is not None]

    verification_failed = [r for r in in_scope if r["verification_verdict"] not in ("SUPPORTED", "NOT_CHECKED")]
    verification_failure_rate = avg([1 if r["verification_verdict"] not in ("SUPPORTED", "NOT_CHECKED") else 0 for r in in_scope])

    out_scope_correct = avg([1 if r["declined_to_answer"] else 0 for r in out_scope]) if out_scope else None

    latencies = [r["latency_sec"] for r in results]

    summary = {
        "n_in_scope": len(in_scope),
        "n_out_of_scope": len(out_scope),
        "retrieval_precision_avg": avg(retrieval_precisions),
        "retrieval_hit_rate": hit_rate,
        "citation_accuracy_avg": avg(citation_accuracies),
        "answer_keyword_score_avg": avg(answer_scores),
        "verification_failure_rate": verification_failure_rate,
        "verification_failure_count": f"{len(verification_failed)}/{len(in_scope)}",
        "out_of_scope_correct_decline_rate": out_scope_correct,
        "latency_mean_sec": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_median_sec": round(statistics.median(latencies), 3) if latencies else None,
        "latency_stdev_sec": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0.0,
    }
    return summary


def run_evaluation(mode="hybrid", use_kg=True, use_verification=True, repeats=1, tag=None, limit=None):
    ground_truth = load_ground_truth()
    if limit:
        ground_truth = ground_truth[:limit]
        print(f"--limit {limit}: using only the first {limit} of {len(load_ground_truth())} questions")
    cfg_name = tag or f"mode-{mode}_kg-{use_kg}_verify-{use_verification}"
    print(f"Running evaluation on {len(ground_truth)} questions | "
          f"mode={mode} use_kg={use_kg} use_verification={use_verification} repeats={repeats}\n")

    all_passes = []
    for rep in range(repeats):
        print(f"--- Pass {rep + 1}/{repeats} ---")
        results = run_single_pass(ground_truth, mode, use_kg, use_verification)
        all_passes.append(results)
        for row in results:
            print(f"[{row['id']}] {row['query']}")
            print(f"    precision={row['retrieval_precision']}  hit={row['retrieval_hit']}  "
                  f"citation_acc={row['citation_accuracy']}  answer_score={row['answer_keyword_score']}  "
                  f"verdict={row['verification_verdict']}  latency={row['latency_sec']}s")
        print()

    # Use the LAST pass for per-question detail, but pool latency across all passes
    # for the median/stdev the reviewers asked for.
    final_results = all_passes[-1]
    all_latencies = [row["latency_sec"] for p in all_passes for row in p]

    summary = summarize(final_results)
    summary["latency_mean_sec"] = round(statistics.mean(all_latencies), 3)
    summary["latency_median_sec"] = round(statistics.median(all_latencies), 3)
    summary["latency_stdev_sec"] = round(statistics.stdev(all_latencies), 3) if len(all_latencies) > 1 else 0.0
    summary["latency_n_samples"] = len(all_latencies)
    summary["config"] = {"mode": mode, "use_kg": use_kg, "use_verification": use_verification, "repeats": repeats}

    print("=" * 60)
    print(f"SUMMARY — {cfg_name}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k:32s}: {v}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{cfg_name}.json"
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "per_question": final_results}, f, indent=2)
    print(f"\nSaved to {out_path}")
    return summary, final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid",
                         help="Retrieval mode ablation: hybrid (default), dense-only, or bm25-only")
    parser.add_argument("--no-kg", action="store_true", help="Disable knowledge-graph lookup (KG ablation)")
    parser.add_argument("--no-verification", action="store_true", help="Disable verification stage (verification ablation)")
    parser.add_argument("--repeats", type=int, default=1, help="Repeat the full run N times for latency median/stdev")
    parser.add_argument("--tag", default=None, help="Custom name for the output results file")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions (for quick testing)")
    args = parser.parse_args()

    run_evaluation(
        mode=args.mode,
        use_kg=not args.no_kg,
        use_verification=not args.no_verification,
        repeats=args.repeats,
        tag=args.tag,
        limit=args.limit,
    )
