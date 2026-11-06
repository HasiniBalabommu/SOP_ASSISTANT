"""
experiment.py — evaluation harness for the second round of experiments.

Differences from eval/evaluate.py (which is kept to reproduce the paper):
  - every question runs in a fresh conversation session, so no history
    leaks between questions, passes or configurations;
  - every repeat is stored (one JSON line per question), with the full
    answer, verdict, evidence, citations, timings and token counts;
  - metrics are computed on the evidence the model actually received;
  - generator and verifier models are chosen per run (cross-model
    verification), and each repeat uses its own seed;
  - extra configurations: no_retrieval and self_check baselines;
  - runs can be resumed: rows already in the output file are skipped.

Usage:
    python -m eval.experiment --configs full,no_verify,self_check,no_retrieval \
        --generator llama3.2:3b --repeats 3 --run-name g3b_core

    python -m eval.experiment --configs full --generator llama3.2:1b \
        --verifier llama3.1:8b --repeats 3 --run-name g1b_verify8b

Index variant, BM25 tokenizer and context length come from the SOP_* environment
variables described in src/config.py. Analyse the results with eval/analyze.py.
"""

import argparse
import json
import os
import platform
import re
import time
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # keep ChromaDB from sending telemetry

from src import baselines
from src import config
from src import keyword_index
from src import llm_client
from src import novelty
from src import pipeline
from eval.refusal import is_refusal

EVAL_DIR = Path(__file__).parent
DEFAULT_QUESTIONS = EVAL_DIR / "questions_v2.json"
DEFAULT_OUT_DIR = EVAL_DIR / "results_v2"

# uses_verifier: the configuration makes calls with the verifier model, so a
# --verifier choice applies to it.
CONFIGS = {
    "full":           {"kind": "pipeline", "mode": "hybrid", "use_kg": True,  "use_verification": True,  "uses_verifier": True},
    "bm25":           {"kind": "pipeline", "mode": "bm25",   "use_kg": True,  "use_verification": True,  "uses_verifier": True},
    "dense":          {"kind": "pipeline", "mode": "dense",  "use_kg": True,  "use_verification": True,  "uses_verifier": True},
    "no_kg":          {"kind": "pipeline", "mode": "hybrid", "use_kg": False, "use_verification": True,  "uses_verifier": True},
    "no_kg_no_verify": {"kind": "pipeline", "mode": "hybrid", "use_kg": False, "use_verification": False},
    "no_verify":      {"kind": "pipeline", "mode": "hybrid", "use_kg": True,  "use_verification": False},
    "self_check":     {"kind": "self_check", "mode": "hybrid", "use_kg": True},
    "no_retrieval":   {"kind": "no_retrieval"},
    "cove":           {"kind": "cove", "mode": "hybrid", "use_kg": True},  # Dhuliawala et al. 2023
    # new methods (src/novelty.py)
    "gate_no_verify": {"kind": "gate", "mode": "hybrid", "use_kg": True, "use_verification": False},
    "gate_full":      {"kind": "gate", "mode": "hybrid", "use_kg": True, "use_verification": True, "uses_verifier": True},
    "suff_full":      {"kind": "sufficiency", "mode": "hybrid", "use_kg": True, "uses_verifier": True},
    "claim_verify":   {"kind": "claim", "mode": "hybrid", "use_kg": True, "uses_verifier": True},
    "checker_verify": {"kind": "checker", "mode": "hybrid", "use_kg": True},  # MiniCheck, no LLM verifier
}

# A response counts as a refusal if it says the information is not available
# (patterns in eval/refusal.py).


def keyword_score(answer: str, keywords: list[str]):
    if not keywords:
        return None
    low = answer.lower()
    return sum(1 for k in keywords if k.lower() in low) / len(keywords)


def _answer_tokens(text: str) -> list[str]:
    """SQuAD-style normalisation: lower case, no punctuation, no articles."""
    text = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^\w\s]", " ", text.lower()))
    return text.split()


def answer_recall(answer: str, gold_answers: list[str]):
    """Best fraction of a gold answer's tokens that appear in the response
    (recall rather than F1, because generated answers are long sentences)."""
    if not gold_answers:
        return None
    response = set(_answer_tokens(answer))
    scores = [sum(t in response for t in toks) / len(toks)
              for toks in (_answer_tokens(g) for g in gold_answers) if toks]
    return max(scores) if scores else None


def run_question(item: dict, cfg: dict, session_id: str) -> dict:
    if cfg["kind"] == "no_retrieval":
        return baselines.no_retrieval(item["query"])
    if cfg["kind"] == "self_check":
        return baselines.self_check(item["query"], session_id=session_id, mode=cfg["mode"], use_kg=cfg["use_kg"])
    if cfg["kind"] == "cove":
        return baselines.cove(item["query"], session_id=session_id, mode=cfg["mode"], use_kg=cfg["use_kg"])
    if cfg["kind"] == "gate":
        return novelty.gated_ask(item["query"], session_id=session_id, mode=cfg["mode"],
                                 use_kg=cfg["use_kg"], use_verification=cfg["use_verification"])
    if cfg["kind"] == "sufficiency":
        return novelty.sufficiency_ask(item["query"], session_id=session_id, mode=cfg["mode"], use_kg=cfg["use_kg"])
    if cfg["kind"] == "claim":
        return novelty.claim_verified_ask(item["query"], session_id=session_id, mode=cfg["mode"], use_kg=cfg["use_kg"])
    if cfg["kind"] == "checker":
        return novelty.claim_verified_ask(item["query"], session_id=session_id, mode=cfg["mode"],
                                          use_kg=cfg["use_kg"], checker="minicheck")
    return pipeline.ask(item["query"], session_id=session_id, mode=cfg["mode"],
                        use_kg=cfg["use_kg"], use_verification=cfg["use_verification"])


def make_row(run_name, cfg_name, repeat, seed, item, result, latency) -> dict:
    expected = item.get("expected_sources", [])
    evidence = result.get("evidence_chunks", [])
    sources = [c["metadata"].get("source") for c in evidence]
    citations = result.get("citations", [])
    in_scope = item["category"] != "out_of_scope"
    poison_keywords = item.get("poison_keywords", [])
    low_answer = result["answer"].lower()
    return {
        "run_name": run_name,
        "config": cfg_name,
        "generator": llm_client.MODEL_NAME,
        "verifier": llm_client.VERIFIER_MODEL or llm_client.MODEL_NAME,
        "repeat": repeat,
        "seed": seed,
        "id": item["id"],
        "category": item["category"],
        "oos_type": item.get("oos_type"),
        "in_scope": in_scope,
        "query": item["query"],
        "expected_sources": expected,
        "expected_keywords": item.get("expected_keywords", []),
        "answer": result["answer"],
        "draft": result.get("draft"),
        "verdict": result.get("verification", {}).get("verdict", "UNKNOWN"),
        "verdict_reason": result.get("verification", {}).get("reason", ""),
        "declined": bool(result.get("declined", False)),
        "refusal_text": is_refusal(result["answer"]),
        "evidence": [{"chunk_id": c.get("chunk_id"), "source": c["metadata"].get("source"),
                      "page": c["metadata"].get("page"), "score": round(c.get("score", 0.0), 5)}
                     for c in evidence],
        "kg_facts": len(result.get("kg_facts", [])),
        "citations": [{"source": c["source"], "page": c["page"]} for c in citations],
        "rp": (sum(s in expected for s in sources) / len(sources)) if (in_scope and sources) else None,
        "hit": any(s in expected for s in sources) if in_scope else None,
        "ca": ((sum(c["source"] in expected for c in citations) / len(citations)) if citations else 0.0) if in_scope else None,
        "aks": keyword_score(result["answer"], item.get("expected_keywords", [])) if in_scope else None,
        "ans_recall": answer_recall(result["answer"], item.get("answers", [])) if in_scope else None,
        "query_identifiers": novelty.query_identifiers(item["query"]),
        "claims": result.get("claims"),
        "poison_keywords": poison_keywords,
        "poison_adopted": any(k.lower() in low_answer for k in poison_keywords) if poison_keywords else None,
        "latency_sec": round(latency, 3),
        "gate_sec": result.get("timing", {}).get("gate_sec"),
        "reasoning_sec": result.get("timing", {}).get("reasoning_sec"),
        "verification_sec": result.get("timing", {}).get("verification_sec"),
        "llm_stats": result.get("llm_stats", {}),
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def environment_info(args) -> dict:
    def ver(pkg):
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    try:
        with urllib.request.urlopen("http://localhost:11434/api/version", timeout=5) as r:
            ollama_server = json.loads(r.read()).get("version")
    except Exception:
        ollama_server = None
    return {
        "args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ollama_server": ollama_server,
        "packages": {p: ver(p) for p in ("ollama", "chromadb", "rank-bm25", "sentence-transformers",
                                          "transformers", "torch", "networkx", "PyMuPDF")},
        "data_dir": str(config.DATA_DIR),
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "bm25_strip_punctuation": config.BM25_STRIP_PUNCTUATION,
        "num_ctx": config.NUM_CTX,
        "generator": llm_client.MODEL_NAME,
        "verifier": llm_client.VERIFIER_MODEL or llm_client.MODEL_NAME,
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="full", help="comma-separated: " + ", ".join(CONFIGS))
    parser.add_argument("--generator", default=None, help="Ollama model for answers (default: SOP_GENERATOR_MODEL)")
    parser.add_argument("--verifier", default=None, help="Ollama model for verification (default: same as generator)")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--start-repeat", type=int, default=0, help="index of the first repeat (used by run_matrix)")
    parser.add_argument("--seed", type=int, default=1000, help="seed of repeat r is seed + r")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-name", required=True, help="results go to <out-dir>/<run-name>.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    parser.add_argument("--legacy-shared-session", action="store_true",
                        help="put every question in one session, as the original evaluation did")
    args = parser.parse_args()

    if args.generator:
        llm_client.MODEL_NAME = args.generator
    if args.verifier:
        llm_client.VERIFIER_MODEL = args.verifier
    keyword_index.reset_cache()

    cfg_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in cfg_names if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {unknown}. Choose from {list(CONFIGS)}")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    if args.limit:
        questions = questions[:args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run_name}.jsonl"
    meta_path = out_dir / f"{args.run_name}_meta.json"
    metas = json.loads(meta_path.read_text()) if meta_path.exists() else []
    metas.append(environment_info(args))
    meta_path.write_text(json.dumps(metas, indent=2))

    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["config"], r["generator"], r["verifier"], r["repeat"], r["id"]))

    print(f"Run {args.run_name}: generator={llm_client.MODEL_NAME} "
          f"verifier={llm_client.VERIFIER_MODEL or llm_client.MODEL_NAME} configs={cfg_names}")
    print(f"Data folder {config.DATA_DIR} | chunks {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP} | "
          f"BM25 strip punctuation {config.BM25_STRIP_PUNCTUATION} | num_ctx {config.NUM_CTX}")
    print(f"{len(questions)} questions, repeats {args.start_repeat}..{args.start_repeat + args.repeats - 1}, "
          f"{len(done)} rows already done\n")

    verifier = llm_client.VERIFIER_MODEL or llm_client.MODEL_NAME
    with open(out_path, "a", encoding="utf-8") as out:
        for repeat in range(args.start_repeat, args.start_repeat + args.repeats):
            seed = args.seed + repeat
            llm_client.SEED = seed
            for cfg_name in cfg_names:
                cfg = CONFIGS[cfg_name]
                if cfg["kind"] == "checker":
                    row_verifier = f"minicheck:{novelty.MINICHECK_MODEL}"
                else:
                    row_verifier = verifier if cfg.get("uses_verifier") else llm_client.MODEL_NAME
                for item in questions:
                    key = (cfg_name, llm_client.MODEL_NAME, row_verifier, repeat, item["id"])
                    if key in done:
                        continue
                    session = "default" if args.legacy_shared_session else \
                        f"{args.run_name}|{cfg_name}|r{repeat}|{item['id']}"
                    t0 = time.time()
                    result = run_question(item, cfg, session)
                    row = make_row(args.run_name, cfg_name, repeat, seed, item, result, time.time() - t0)
                    row["verifier"] = row_verifier
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"[r{repeat} {cfg_name:12s} {item['id']:>4}] verdict={row['verdict']:<20} "
                          f"declined={row['declined']!s:<5} refusal={row['refusal_text']!s:<5} "
                          f"hit={row['hit']!s:<5} {row['latency_sec']:.1f}s")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
