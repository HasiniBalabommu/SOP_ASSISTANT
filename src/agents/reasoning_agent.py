"""
reasoning_agent.py — takes retrieved evidence and generates a draft
answer using the local LLM. Explicitly instructed to only use the
provided evidence, not its own general knowledge, to reduce hallucination.
"""
from src import memory
from src import llm_client
from src.agents import retrieval_agent

SYSTEM_PROMPT = """You are the Industrial Knowledge Assistant — a practical,
concise AI assistant for engineers and technicians. You must answer ONLY using
the evidence provided below. Do not use any outside knowledge. If the evidence
does not contain enough information to answer, say so clearly instead of
guessing. Speak like you're talking to a technician on the shop floor: direct,
practical, no fluff."""


def _build_prompt(query: str, evidence: dict, history_text: str = "") -> str:
    chunks_text = "\n\n".join(
        f"[Source: {c['metadata'].get('source')}]\n{c['text']}"
        for c in evidence["evidence_chunks"]
    )
    kg_text = "\n".join(
        f"{s} --{rel}--> {t}" for s, rel, t in evidence["kg_facts"]
    )

    prompt = ""
    if history_text:
        prompt += f"Conversation so far:\n{history_text}\n\n"

    prompt += f"Question: {query}\n\nEvidence:\n{chunks_text}"
    if kg_text:
        prompt += f"\n\nRelated facts:\n{kg_text}"
    prompt += "\n\nAnswer the question using only the evidence above. Use the conversation so far only to understand context — do not treat it as evidence."
    return prompt


def reason(query: str, session_id: str = "default", top_k: int = 5,
           mode: str = "hybrid", use_kg: bool = True) -> dict:
    """
    Retrieves evidence, generates a draft answer grounded in it.
    Now session-aware: pulls prior conversation turns for context,
    and saves this turn afterward so future questions can reference it.

    mode/use_kg: passed straight through to retrieval_agent.retrieve
    for the retrieval-mode and KG ablations.
    """
    evidence = retrieval_agent.retrieve(query, top_k=top_k, mode=mode, use_kg=use_kg)
    history_text = memory.format_history_for_prompt(session_id)

    if not evidence["has_evidence"]:
        answer = "I don't have enough information in the current knowledge base to answer this question."
        memory.add_turn(session_id, query, answer)
        return {
            "query": query,
            "answer": answer,
            "evidence": evidence,
        }

    prompt = _build_prompt(query, evidence, history_text)
    answer = llm_client.generate(prompt, system=SYSTEM_PROMPT)

    memory.add_turn(session_id, query, answer)

    return {
        "query": query,
        "answer": answer,
        "evidence": evidence,
    }


if __name__ == "__main__":
    sid = "test_session"
    r1 = reason("What should I do if I see error code E205?", session_id=sid)
    print(f"Q1: {r1['query']}\nA1: {r1['answer']}\n")

    r2 = reason("How often should I check that?", session_id=sid)  # deliberately vague - relies on memory
    print(f"Q2: {r2['query']}\nA2: {r2['answer']}")