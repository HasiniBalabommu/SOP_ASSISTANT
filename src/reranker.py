"""
reranker.py — re-scores retrieved chunks against the query using a
cross-encoder model, which reads the query and chunk together rather
than comparing separately-computed embeddings. More accurate than
cosine similarity or BM25 alone, but too slow to run over an entire
corpus — so it's only applied to the small candidate set hybrid
retrieval has already narrowed things down to.
"""

from sentence_transformers import CrossEncoder

_model = None


def get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print("Loading re-ranking model (first time only)...")
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, candidates: list[dict], top_k: int = 4) -> list[dict]:
    """
    Re-scores a list of candidate chunks against the query and returns
    the top_k most relevant, sorted by the cross-encoder's score.
    Each candidate must have a 'text' field.
    """
    if not candidates:
        return []

    model = get_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


if __name__ == "__main__":
    # quick manual test
    test_candidates = [
        {"text": "The battery undervoltage error E501 requires checking the charger."},
        {"text": "Belt misalignment error E402 is caused by worn idler rollers."},
        {"text": "Weekly maintenance includes checking tire pressure and horn function."},
    ]
    results = rerank("What causes error E501?", test_candidates, top_k=2)
    for r in results:
        print(f"score={r['rerank_score']:.3f}  {r['text'][:60]}...")