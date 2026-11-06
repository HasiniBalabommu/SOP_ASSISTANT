"""
pipeline.py — the single entry point for asking a question against
everything currently indexed. Delegates to the full agent chain:
retrieval -> reasoning -> verification -> citation. Greetings/small
talk are caught early and skip the heavy pipeline entirely.
"""

from src.agents import citation_agent
from src import intent_filter


def ask(query: str, session_id: str = "default", top_k: int = 5,
        mode: str = "hybrid", use_kg: bool = True, use_verification: bool = True) -> dict:
    """
    Runs the complete pipeline and returns a final, verified,
    citation-backed answer. This is what the app/UI layer should call.

    mode / use_kg / use_verification: ablation switches used by
    eval/evaluate.py — leave at their defaults for normal use.
    """
    if intent_filter.is_small_talk(query):
        return {
            "query": query,
            "answer": intent_filter.small_talk_response(query),
            "verification": {"verdict": "NO_EVIDENCE", "reason": "Small talk — no retrieval needed."},
            "declined": False,
            "citations": [],
            "timing": {},
        }

    # session_id is passed down so each session keeps its own history
    # (earlier versions dropped it here, so every call shared "default").
    return citation_agent.cite(query, session_id=session_id, top_k=top_k, mode=mode,
                               use_kg=use_kg, use_verification=use_verification)


if __name__ == "__main__":
    test_queries = [
        "Hi",
        "What should I do if I see error code E205?",
        "How do I maintain the hydraulic pump?",
    ]
    for q in test_queries:
        result = ask(q)
        citation_agent.print_full_result(result)
        print()