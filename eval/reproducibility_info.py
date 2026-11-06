"""
reproducibility_info.py — collects the exact model/parameter/hardware
info reviewers asked for (comment #15) into one compact table.

Usage:
    python -m eval.reproducibility_info
"""

import platform
import sys
import subprocess
import json
from pathlib import Path

from src import config
from src import llm_client
from src.hybrid_retriever import RRF_K
from src.agents.retrieval_agent import MIN_SCORE_THRESHOLD

OUT_PATH = Path(__file__).parent / "reproducibility.json"


def pip_freeze_versions(packages):
    try:
        out = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                              capture_output=True, text=True, check=True).stdout
    except Exception:
        return {p: "unknown" for p in packages}
    versions = {}
    lower_lines = {line.split("==")[0].lower(): line for line in out.splitlines() if "==" in line}
    for p in packages:
        versions[p] = lower_lines.get(p.lower(), "not found")
    return versions


def main():
    info = {
        "llm_model": llm_client.MODEL_NAME,
        "llm_backend": "Ollama (local)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "retrieval_mode_default": "hybrid (BM25 + dense, fused via RRF)",
        "rrf_k_constant": RRF_K,
        "min_retrieval_score_threshold": MIN_SCORE_THRESHOLD,
        "chunk_size_words": config.CHUNK_SIZE,
        "chunk_overlap_words": config.CHUNK_OVERLAP,
        "generation_temperature": 0.2,
        "verification_temperature": 0.0,
        "top_k_default": 5,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    info["package_versions"] = pip_freeze_versions(
        ["ollama", "chromadb", "rank-bm25", "sentence-transformers", "networkx", "fastapi"]
    )

    print("=" * 60)
    print("REPRODUCIBILITY TABLE")
    print("=" * 60)
    for k, v in info.items():
        if k == "package_versions":
            continue
        print(f"{k:32s}: {v}")
    print("\nPackage versions:")
    for k, v in info["package_versions"].items():
        print(f"  {k:24s}: {v}")

    with open(OUT_PATH, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    print("\nNOTE: fill in your actual hardware (CPU model, RAM, GPU/CPU-only) by hand —")
    print("      that isn't reliably auto-detectable across OSes.")


if __name__ == "__main__":
    main()
