"""
hybrid_retriever.py — combines semantic search (embed_store) and
keyword search (keyword_index) into a single ranked result list,
using Reciprocal Rank Fusion (RRF).

Supports a `mode` parameter for the retrieval ablation requested in
review: "hybrid" (default, dense+BM25 fused via RRF), "dense" (vector
search only), or "bm25" (keyword search only). This lets eval/evaluate.py
report Precision/Hit Rate separately for each retrieval configuration
without duplicating the search logic.
"""

from src import embed_store
from src import keyword_index

RRF_K = 60  # standard constant used in RRF — dampens the impact of rank 1 vs rank 2, etc.

VALID_MODES = {"hybrid", "dense", "bm25"}


def _rrf_scores(ranked_ids: list[str]) -> dict[str, float]:
    """
    Given a list of chunk_ids in rank order (best first), returns
    {chunk_id: rrf_score}. Rank 1 contributes the most, decaying
    smoothly for lower ranks.
    """
    scores = {}
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        scores[chunk_id] = 1.0 / (RRF_K + rank)
    return scores


def hybrid_search(query: str, top_k: int = 5, mode: str = "hybrid") -> list[dict]:
    """
    Runs semantic and/or keyword search depending on `mode`, fuses the
    rankings when both are used, and returns the top_k merged results.

      mode="hybrid" (default) : dense + BM25, fused via RRF
      mode="dense"            : semantic/embedding search only
      mode="bm25"             : keyword/BM25 search only
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown retrieval mode '{mode}', expected one of {VALID_MODES}")

    vector_ids, vector_texts, vector_meta = [], {}, {}
    if mode in ("hybrid", "dense"):
        vector_results = embed_store.search(query, top_k=top_k * 2)
        vector_ids = vector_results["ids"][0] if vector_results["ids"] else []
        vector_texts = {
            cid: doc for cid, doc in zip(vector_ids, vector_results["documents"][0])
        }
        vector_meta = {
            cid: meta for cid, meta in zip(vector_ids, vector_results["metadatas"][0])
        }

    keyword_ids = []
    if mode in ("hybrid", "bm25"):
        keyword_results = keyword_index.search(query, top_k=top_k * 2)
        keyword_ids = [r["chunk_id"] for r in keyword_results]
        for r in keyword_results:
            vector_texts.setdefault(r["chunk_id"], r["text"])
            vector_meta.setdefault(r["chunk_id"], r["metadata"])

    # --- Fuse rankings (no-op when only one source is active: the other
    # source's score dict is simply empty, so combined_score == that
    # source's own RRF score) ---
    vec_scores = _rrf_scores(vector_ids)
    kw_scores = _rrf_scores(keyword_ids)

    all_ids = set(vector_ids) | set(keyword_ids)
    fused = []
    for cid in all_ids:
        combined_score = vec_scores.get(cid, 0.0) + kw_scores.get(cid, 0.0)
        fused.append({
            "chunk_id": cid,
            "text": vector_texts.get(cid, ""),
            "metadata": vector_meta.get(cid, {}),
            "score": combined_score,
            "in_vector": cid in vec_scores,
            "in_keyword": cid in kw_scores,
        })

    # chunk_id breaks ties: all_ids is a set, whose order changes between processes
    fused.sort(key=lambda x: (-x["score"], x["chunk_id"]))
    return fused[:top_k]


if __name__ == "__main__":
    test_queries = [
        "low suction pressure",   # should favor semantic match
        "E104",                    # should favor keyword match
    ]
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = hybrid_search(q, top_k=3)
        for r in results:
            tags = []
            if r["in_vector"]:
                tags.append("semantic")
            if r["in_keyword"]:
                tags.append("keyword")
            print(f"  score={r['score']:.4f} [{'+'.join(tags)}] [{r['metadata'].get('source')}] {r['text'][:90]}...")