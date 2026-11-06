"""
keyword_index.py — BM25-based keyword search, the exact-match half of
hybrid retrieval. Complements embed_store.py's semantic search.
"""

import json
import pickle
import string
from pathlib import Path

from rank_bm25 import BM25Okapi

from src import config

INDEX_PATH = config.DATA_DIR / "vector_store" / "bm25_index.pkl"

# Module-level cache so we don't rebuild the BM25 object (re-tokenizing
# every chunk) on every single search call — only when the index has
# actually grown (new documents added).
_bm25_cache = None
_bm25_cache_size = 0

def reset_cache() -> None:
    """Forces the next search() call to rebuild the BM25 index from scratch —
    needed after the underlying index file has been deleted."""
    global _bm25_cache, _bm25_cache_size
    _bm25_cache = None
    _bm25_cache_size = 0

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer — good enough for BM25.
    With config.BM25_STRIP_PUNCTUATION, punctuation is also stripped from
    token edges so "E205?" in a query matches "E205" in a chunk; tokens made
    only of punctuation (e.g. "+=") are kept as they are."""
    tokens = text.lower().split()
    if config.BM25_STRIP_PUNCTUATION:
        tokens = [t.strip(string.punctuation) or t for t in tokens]
    return tokens


def load_index() -> dict:
    """Returns {'chunk_ids': [...], 'texts': [...], 'metadatas': [...]}
    or an empty structure if no index exists yet."""
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    return {"chunk_ids": [], "texts": [], "metadatas": []}


def save_index(data: dict) -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(data, f)


def add_chunks_file(chunks_json_path: Path) -> int:
    """
    Reads a chunks JSON file and adds its chunks into the BM25 index.
    Since BM25Okapi doesn't support true incremental updates, we keep
    the raw chunk list persisted and rebuild the BM25 object in memory
    only when needed (see search() below), not on every add.
    """
    with open(chunks_json_path, "r") as f:
        chunks = json.load(f)
    if not chunks:
        return 0

    data = load_index()
    existing_ids = set(data["chunk_ids"])

    added = 0
    for c in chunks:
        if c["chunk_id"] in existing_ids:
            continue  # already indexed, skip (mirrors upsert behavior)
        data["chunk_ids"].append(c["chunk_id"])
        data["texts"].append(c["text"])
        data["metadatas"].append({"source": c["source"], "page": c["page"]})
        added += 1

    save_index(data)
    return added


def search(query: str, top_k: int = 5):
    """
    Keyword search: returns top_k chunks ranked by BM25 score.
    Caches the BM25 object in memory and only rebuilds it when the
    index has grown since the last search — avoids re-tokenizing the
    entire corpus on every single query, which was the main slowdown.
    """
    global _bm25_cache, _bm25_cache_size

    data = load_index()
    if not data["texts"]:
        return []

    if _bm25_cache is None or _bm25_cache_size != len(data["texts"]):
        tokenized_corpus = [_tokenize(t) for t in data["texts"]]
        _bm25_cache = BM25Okapi(tokenized_corpus)
        _bm25_cache_size = len(data["texts"])

    tokenized_query = _tokenize(query)
    scores = _bm25_cache.get_scores(tokenized_query)

    ranked = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    return [
        {
            "chunk_id": data["chunk_ids"][i],
            "text": data["texts"][i],
            "metadata": data["metadatas"][i],
            "score": float(scores[i]),
        }
        for i in ranked
        if scores[i] > 0  # drop zero-score (no keyword overlap at all) results
    ]


if __name__ == "__main__":
    chunk_files = list(config.CHUNKS_DIR.glob("*.json"))
    if not chunk_files:
        print(f"No chunk files in {config.CHUNKS_DIR} — run ingest.py first.")
    else:
        total = 0
        for cf in chunk_files:
            n = add_chunks_file(cf)
            print(f"Indexed {cf.name}: {n} new chunks")
            total += n
        print(f"\nTotal in BM25 index: {len(load_index()['chunk_ids'])}")

        print("\nTest query (exact fault code): 'E104'")
        results = search("E104", top_k=3)
        for r in results:
            print(f"  score={r['score']:.2f} [{r['metadata']['source']}] {r['text'][:100]}...")