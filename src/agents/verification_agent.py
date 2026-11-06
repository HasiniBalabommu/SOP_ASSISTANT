"""
verification_agent.py — checks a draft answer against the evidence it
was generated from, to catch hallucinated or unsupported claims.
Runs as a SEPARATE LLM call from the one that generated the answer,
so generation and checking are structurally split rather than the
same call grading its own output. (Reviewers correctly note this is
a *separate verification stage*, not statistical/model independence —
see VERIFICATION_ABLATION_NOTE below and the paper's terminology fix.)
"""

import time

from src import llm_client
from src.agents import reasoning_agent

VERIFICATION_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given
a QUESTION, some EVIDENCE, and a DRAFT ANSWER that claims to be based on that
evidence. Your job is to check whether every claim in the DRAFT ANSWER is
actually supported by the EVIDENCE.

Respond in exactly this format:
VERDICT: SUPPORTED or UNSUPPORTED or PARTIALLY_SUPPORTED
REASON: <one or two sentences explaining your verdict>

Be strict — if the answer includes any detail not present in the evidence
(even something that sounds plausible or generally true), mark it as
UNSUPPORTED or PARTIALLY_SUPPORTED and explain what wasn't backed up."""


def _build_verification_prompt(query: str, answer: str, evidence: dict) -> str:
    chunks_text = "\n\n".join(
        f"[Source: {c['metadata'].get('source')}]\n{c['text']}"
        for c in evidence["evidence_chunks"]
    )
    return (
        f"QUESTION: {query}\n\n"
        f"EVIDENCE:\n{chunks_text}\n\n"
        f"DRAFT ANSWER:\n{answer}\n\n"
        f"Check the DRAFT ANSWER against the EVIDENCE now."
    )


def _parse_verdict(raw_response: str) -> dict:
    """Pulls the VERDICT and REASON lines out of the LLM's response."""
    verdict = "UNKNOWN"
    reason = raw_response.strip()

    for line in raw_response.splitlines():
        if line.strip().upper().startswith("VERDICT:"):
            # llama3.2:3b sometimes writes "PARTIALLY SUPPORTED"; normalise to the label
            # the verdict policy expects, so the caveat is shown
            verdict = "_".join(line.split(":", 1)[1].strip(" *.").upper().replace("-", " ").split())
        elif line.strip().upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    return {"verdict": verdict, "reason": reason}


def verify(query: str, session_id: str = "default", top_k: int = 5, mode: str = "hybrid",
           use_kg: bool = True, use_verification: bool = True) -> dict:
    """
    Runs reasoning to get a draft answer, then (unless disabled) checks
    it against the same evidence in a separate LLM call. Returns the
    answer plus a verdict, and per-stage timings for the latency report.

    use_verification=False reproduces the "without verification" arm of
    the verification ablation: the draft answer is returned unchanged
    and unchecked, with verdict "NOT_CHECKED".
    """
    t0 = time.time()
    llm_client.LAST_STATS.clear()
    draft = reasoning_agent.reason(query, session_id=session_id, top_k=top_k, mode=mode, use_kg=use_kg)
    reasoning_time = time.time() - t0
    llm_stats = {"reasoning": dict(llm_client.LAST_STATS)}

    if not draft["evidence"]["has_evidence"]:
        # nothing was retrieved, so there's nothing to verify against
        return {
            **draft,
            "verification": {"verdict": "NO_EVIDENCE", "reason": "No evidence was retrieved to check against."},
            "timing": {"reasoning_sec": round(reasoning_time, 3), "verification_sec": 0.0},
            "llm_stats": llm_stats,
        }

    if not use_verification:
        return {
            **draft,
            "verification": {"verdict": "NOT_CHECKED", "reason": "Verification stage disabled for this run (ablation)."},
            "timing": {"reasoning_sec": round(reasoning_time, 3), "verification_sec": 0.0},
            "llm_stats": llm_stats,
        }

    t1 = time.time()
    verification_prompt = _build_verification_prompt(
        query, draft["answer"], draft["evidence"]
    )
    raw_verdict = llm_client.generate(
        verification_prompt,
        system=VERIFICATION_SYSTEM_PROMPT,
        temperature=0.0,  # deterministic — this is a checking task, not a creative one
        model=llm_client.VERIFIER_MODEL,  # None = same model as the generator
    )
    verdict = _parse_verdict(raw_verdict)
    verification_time = time.time() - t1
    llm_stats["verification"] = dict(llm_client.LAST_STATS)

    return {
        **draft,
        "llm_stats": llm_stats,
        "verification": verdict,
        "timing": {"reasoning_sec": round(reasoning_time, 3), "verification_sec": round(verification_time, 3)},
    }


if __name__ == "__main__":
    result = verify("What should I do if I see error code E205?")
    print(f"Query: {result['query']}\n")
    print(f"Draft Answer:\n{result['answer']}\n")
    print(f"Verification:")
    print(f"  Verdict: {result['verification']['verdict']}")
    print(f"  Reason:  {result['verification']['reason']}")