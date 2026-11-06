"""
retrieval_agent.py — the first agent in the chain. Responsible for
deciding what evidence to retrieve for a query and packaging it in a
clean structure for the reasoning agent to consume.
"""

from src import hybrid_retriever
from src import kg_retriever
import re


MIN_SCORE_THRESHOLD = 0.005  # drop results with a very weak RRF score

def retrieve(query: str, top_k: int = 5, mode: str = "hybrid", use_kg: bool = True) -> dict:
    """
    mode:    "hybrid" (default), "dense", or "bm25" — passed straight
             through to hybrid_retriever for the retrieval ablation.
    use_kg:  set False to disable knowledge-graph fact lookup, for the
             KG ablation (answer/citation quality with vs. without the
             graph).
    """
    results = hybrid_retriever.hybrid_search(query, top_k=top_k, mode=mode)

    kg_facts = []
    if use_kg:
        mentioned_codes = re.findall(r"\b[A-Z]\d{2,4}\b", query)
        for code in mentioned_codes:
            kg_facts.extend(kg_retriever.query_related(code))

    filtered_chunks = [c for c in results if c["score"] >= MIN_SCORE_THRESHOLD]

    return {
        "query": query,
        "evidence_chunks": filtered_chunks,
        "kg_facts": kg_facts,
        "has_evidence": len(filtered_chunks) > 0,
    }


if __name__ == "__main__":
    result = retrieve("What should I do if I see error code E205?")
    print(f"Query: {result['query']}")
    print(f"Has evidence: {result['has_evidence']}")
    print(f"\nEvidence chunks ({len(result['evidence_chunks'])}):")
    for c in result["evidence_chunks"]:
        print(f"  [{c['metadata'].get('source')}] score={c['score']:.4f} {c['text'][:100]}...")
    if result["kg_facts"]:
        print(f"\nKG facts:")
        for s, rel, t in result["kg_facts"]:
            print(f"  {s} --{rel}--> {t}")