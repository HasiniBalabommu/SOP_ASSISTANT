"""
retrieval_tokenizer_check.py — post-hoc retrieval-only check of the BM25
tokenizer (paper, Section V-E). No LLM is called, so Ollama is not needed.

For the 33 in-scope questions it measures Retrieval Hit Rate and Precision
for hybrid, dense and BM25 retrieval twice:
  - original  : config.BM25_STRIP_PUNCTUATION = False (the evaluated system)
  - corrected : config.BM25_STRIP_PUNCTUATION = True
and compares the original run with the stored results in eval/results/.

BM25 is deterministic. Dense results can differ between process starts,
because the ChromaDB HNSW index may return different neighbours each time
it is loaded, so run the check several times, each in a new process.

Usage:
    python -m eval.retrieval_tokenizer_check                  # one run
    python -m eval.retrieval_tokenizer_check --tag run1       # saves retrieval_tokenizer_check_run1.json
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from src import config, keyword_index
from src.agents import retrieval_agent
from eval.evaluate import load_ground_truth, retrieval_hit, retrieval_precision

RESULTS_DIR = Path(__file__).parent / "results"
STORED = {
    "hybrid": "main.json",
    "dense": "ablation_retrieval_dense.json",
    "bm25": "ablation_retrieval_bm25.json",
}


def run_mode(questions, mode):
    rows = []
    for item in questions:
        chunks = retrieval_agent.retrieve(item["query"], mode=mode, use_kg=False)["evidence_chunks"]
        rows.append({
            "id": item["id"],
            "category": item["category"],
            "retrieval_hit": retrieval_hit(chunks, item["expected_sources"]),
            "retrieval_precision": round(retrieval_precision(chunks, item["expected_sources"]), 3),
            "sources": [c["metadata"].get("source") for c in chunks],
        })
    return rows


def summarize(rows):
    n = len(rows)
    return {
        "hit_rate": f"{sum(r['retrieval_hit'] for r in rows)}/{n}",
        "precision_avg": round(sum(r["retrieval_precision"] for r in rows) / n, 3),
        "misses": [r["id"] for r in rows if not r["retrieval_hit"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None, help="suffix for the output filename")
    args = parser.parse_args()

    questions = [q for q in load_ground_truth() if q["category"] != "out_of_scope"]
    report = {
        "versions": {p: version(p) for p in ("torch", "transformers", "sentence-transformers",
                                             "chromadb", "rank-bm25")},
        "n_questions": len(questions),
    }
    for name, strip in (("original", False), ("corrected", True)):
        config.BM25_STRIP_PUNCTUATION = strip
        keyword_index.reset_cache()
        report[name] = {}
        for mode in ("hybrid", "dense", "bm25"):
            rows = run_mode(questions, mode)
            entry = summarize(rows)
            if name == "original":
                stored = json.loads((RESULTS_DIR / STORED[mode]).read_text(encoding="utf-8"))
                stored = {r["id"]: r for r in stored["per_question"]}
                entry["differs_from_stored"] = [
                    r["id"] for r in rows
                    if r["retrieval_hit"] != stored[r["id"]]["retrieval_hit"]
                    or abs(r["retrieval_precision"] - stored[r["id"]]["retrieval_precision"]) > 1e-3
                ]
            entry["per_question"] = rows
            report[name][mode] = entry
            print(f"{name:9s} {mode:6s} hit={entry['hit_rate']}  precision={entry['precision_avg']}  "
                  f"misses={entry['misses']}")
    config.BM25_STRIP_PUNCTUATION = False

    name = f"retrieval_tokenizer_check_{args.tag}.json" if args.tag else "retrieval_tokenizer_check.json"
    out = RESULTS_DIR / name
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
