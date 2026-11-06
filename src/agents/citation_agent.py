"""
citation_agent.py — attaches document/page citations to a verified
answer, and enforces the verification verdict (reviewer comment #11:
"clarify the exact verification decision policy"). Policy implemented
here:

  SUPPORTED            -> answer returned as-is
  PARTIALLY_SUPPORTED  -> answer returned with an explicit caveat
                          prefix, so the technician sees the warning
                          rather than a silently-passed answer
  UNSUPPORTED          -> answer is REPLACED with an explicit decline
                          message; the unsupported draft is not shown
  NO_EVIDENCE           -> answer already is a decline message
  NOT_CHECKED           -> ablation mode (verification disabled);
                          answer returned as-is, unlabelled as verified

Citation formatting itself is pure data assembly, no LLM call needed —
the metadata already exists on each evidence chunk from ingestion, so
generating it with an LLM would only risk introducing incorrect citations.
"""

from src.agents import verification_agent

DECLINE_MESSAGE = (
    "I can't confirm an answer to this from the current knowledge base: "
    "the draft answer wasn't supported by the retrieved evidence, so I'm "
    "declining rather than giving you an unverified answer."
)
PARTIAL_CAVEAT = (
    "Note: parts of this answer could not be fully verified against the "
    "retrieved evidence. Treat the following with caution and confirm "
    "against the source document before acting on it.\n\n"
)


def _format_citations(evidence_chunks: list[dict]) -> list[dict]:
    """
    Deduplicates and formats citations — multiple chunks from the same
    source/page collapse into one citation entry.
    """
    seen = set()
    citations = []
    for c in evidence_chunks:
        source = c["metadata"].get("source", "unknown")
        page = c["metadata"].get("page", 0)
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "source": source,
            "page": page,
            "excerpt": c["text"][:150] + ("..." if len(c["text"]) > 150 else ""),
        })
    return citations


def _apply_verdict_policy(answer: str, verdict: str) -> tuple[str, bool]:
    """Returns (possibly-modified answer, declined: bool)."""
    if verdict == "UNSUPPORTED":
        return DECLINE_MESSAGE, True
    if verdict == "PARTIALLY_SUPPORTED":
        return PARTIAL_CAVEAT + answer, False
    return answer, False


def cite(query: str, session_id: str = "default", top_k: int = 5, mode: str = "hybrid",
         use_kg: bool = True, use_verification: bool = True) -> dict:
    """
    Runs the full chain — retrieval, reasoning, verification — applies
    the verdict policy above, then attaches formatted citations.
    mode / use_kg / use_verification are passed straight through for
    the retrieval / KG / verification ablations.
    """
    result = verification_agent.verify(
        query, session_id=session_id, top_k=top_k, mode=mode, use_kg=use_kg,
        use_verification=use_verification,
    )

    verdict = result["verification"]["verdict"]
    final_answer, declined = _apply_verdict_policy(result["answer"], verdict)

    citations = [] if declined else _format_citations(result["evidence"]["evidence_chunks"])

    return {
        "query": result["query"],
        "answer": final_answer,
        "draft": result["answer"],  # kept so evaluation can score drafts that were declined
        "verification": result["verification"],
        "declined": declined,
        "citations": citations,
        "timing": result.get("timing", {}),
        # the evidence actually shown to the model, so evaluation metrics
        # are computed on it rather than on a second retrieval call
        "evidence_chunks": result["evidence"]["evidence_chunks"],
        "kg_facts": result["evidence"].get("kg_facts", []),
        "llm_stats": result.get("llm_stats", {}),
    }


def print_full_result(result: dict) -> None:
    print(f"Query: {result['query']}\n")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Verification: {result['verification']['verdict']} — {result['verification']['reason']}\n")
    print("Citations:")
    for c in result["citations"]:
        print(f"  [{c['source']}, page {c['page']}] \"{c['excerpt']}\"")


if __name__ == "__main__":
    result = cite("What should I do if I see error code E205?")
    print_full_result(result)