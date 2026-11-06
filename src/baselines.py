"""
baselines.py — comparison systems for the evaluation (not used by the app).

  no_retrieval : the same LLM answers the question on its own, with no
                 evidence, no verification and no citations. Shows what
                 the model does without the knowledge base.
  cove         : Chain-of-Verification (Dhuliawala et al., 2023, factored
                 variant), adapted to retrieved evidence: plan verification
                 questions from the draft, answer each from the evidence
                 without seeing the draft, then revise the draft.
  self_check   : retrieval-augmented answer in which the model checks its
                 OWN answer inside the same call (answer + VERDICT line),
                 instead of a separate verification call. The verdict
                 policy and citations are the same as in the full system,
                 so the only difference is where the check happens.

Both return the same dictionary shape as citation_agent.cite(), so the
evaluation harness can treat every configuration the same way.
"""

import re
import time

from src import llm_client
from src import memory
from src.agents import citation_agent
from src.agents import reasoning_agent
from src.agents import retrieval_agent
from src.agents import verification_agent
from eval.refusal import is_refusal

NO_RETRIEVAL_SYSTEM_PROMPT = """You are a practical, concise AI assistant for
engineers and technicians. Answer the question. If you do not know the answer,
say so clearly instead of guessing."""

SELF_CHECK_INSTRUCTION = """After your answer, check it against the evidence
yourself and end your response with exactly one final line:
VERDICT: SUPPORTED or UNSUPPORTED or PARTIALLY_SUPPORTED"""


def no_retrieval(query: str) -> dict:
    t0 = time.time()
    llm_client.LAST_STATS.clear()
    answer = llm_client.generate(query, system=NO_RETRIEVAL_SYSTEM_PROMPT, temperature=0.2)
    return {
        "query": query,
        "answer": answer,
        "verification": {"verdict": "NOT_CHECKED", "reason": "No-retrieval baseline."},
        "declined": False,
        "citations": [],
        "timing": {"reasoning_sec": round(time.time() - t0, 3), "verification_sec": 0.0},
        "evidence_chunks": [],
        "kg_facts": [],
        "llm_stats": {"reasoning": dict(llm_client.LAST_STATS)},
    }


def _split_verdict(raw: str) -> tuple[str, str]:
    """Separates the answer text from the trailing VERDICT line."""
    verdict = verification_agent._parse_verdict(raw)["verdict"]
    answer_lines = [line for line in raw.splitlines()
                    if not line.strip().upper().startswith("VERDICT:")]
    return "\n".join(answer_lines).strip(), verdict


def self_check(query: str, session_id: str = "default", top_k: int = 5,
               mode: str = "hybrid", use_kg: bool = True) -> dict:
    t0 = time.time()
    evidence = retrieval_agent.retrieve(query, top_k=top_k, mode=mode, use_kg=use_kg)
    llm_client.LAST_STATS.clear()

    if not evidence["has_evidence"]:
        answer, verdict = "I don't have enough information in the current knowledge base to answer this question.", "NO_EVIDENCE"
    else:
        prompt = reasoning_agent._build_prompt(query, evidence, memory.format_history_for_prompt(session_id))
        raw = llm_client.generate(prompt, system=reasoning_agent.SYSTEM_PROMPT + "\n\n" + SELF_CHECK_INSTRUCTION)
        answer, verdict = _split_verdict(raw)
    memory.add_turn(session_id, query, answer)

    final_answer, declined = citation_agent._apply_verdict_policy(answer, verdict)
    citations = [] if declined else citation_agent._format_citations(evidence["evidence_chunks"])
    return {
        "query": query,
        "answer": final_answer,
        "verification": {"verdict": verdict, "reason": "Self-check inside the generation call."},
        "declined": declined,
        "citations": citations,
        "timing": {"reasoning_sec": round(time.time() - t0, 3), "verification_sec": 0.0},
        "evidence_chunks": evidence["evidence_chunks"],
        "kg_facts": evidence.get("kg_facts", []),
        "llm_stats": {"reasoning": dict(llm_client.LAST_STATS)},
    }


COVE_PLAN_SYSTEM_PROMPT = """You are checking a DRAFT ANSWER to a QUESTION. Write up to
three short verification questions that test the specific facts stated in the
draft (values, names, steps). Write one question per line and nothing else."""

COVE_EXECUTE_SYSTEM_PROMPT = """Answer the QUESTION using only the EVIDENCE, in one
sentence. If the evidence does not state the answer, reply: NOT IN EVIDENCE"""

COVE_FINAL_SYSTEM_PROMPT = """You revise a DRAFT ANSWER using VERIFICATION RESULTS.
Keep only the statements the results support and correct the ones they contradict.
Reply with the revised answer and nothing else. If the results support nothing that
answers the QUESTION, reply only: The evidence does not contain this information."""

COVE_MAX_QUESTIONS = 3


def _cove_questions(plan: str) -> list[str]:
    lines = [re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip() for line in plan.splitlines()]
    return [line for line in lines if len(line.split()) >= 3 and line.endswith("?")][:COVE_MAX_QUESTIONS]


def cove(query: str, session_id: str = "default", top_k: int = 5, mode: str = "hybrid",
         use_kg: bool = True) -> dict:
    t0 = time.time()
    llm_client.LAST_STATS.clear()
    draft = reasoning_agent.reason(query, session_id=session_id, top_k=top_k, mode=mode, use_kg=use_kg)
    reasoning_time = round(time.time() - t0, 3)
    evidence = draft["evidence"]
    stats = {"reasoning": dict(llm_client.LAST_STATS)}
    answer, steps = draft["answer"], []

    t1 = time.time()
    if evidence["has_evidence"] and not is_refusal(answer):  # a refusal has nothing to verify
        chunks_text = "\n\n".join(f"[Source: {c['metadata'].get('source')}]\n{c['text']}"
                                  for c in evidence["evidence_chunks"])
        plan = llm_client.generate(f"QUESTION: {query}\n\nDRAFT ANSWER:\n{answer}",
                                   system=COVE_PLAN_SYSTEM_PROMPT)
        for question in _cove_questions(plan):
            reply = llm_client.generate(f"QUESTION: {question}\n\nEVIDENCE:\n{chunks_text}",
                                        system=COVE_EXECUTE_SYSTEM_PROMPT, temperature=0.0)
            steps.append({"question": question, "answer": reply.strip()})
        if steps:
            results = "\n".join(f"Q: {s['question']}\nA: {s['answer']}" for s in steps)
            answer = llm_client.generate(f"QUESTION: {query}\n\nDRAFT ANSWER:\n{answer}\n\n"
                                         f"VERIFICATION RESULTS:\n{results}",
                                         system=COVE_FINAL_SYSTEM_PROMPT).strip()
    return {
        "query": query,
        "answer": answer,
        "draft": draft["answer"],
        "verification": {"verdict": "REVISED" if steps else "NOT_CHECKED",
                         "reason": f"{len(steps)} verification questions"},
        "declined": False,
        "citations": [] if is_refusal(answer) else citation_agent._format_citations(evidence["evidence_chunks"]),
        "timing": {"reasoning_sec": reasoning_time, "verification_sec": round(time.time() - t1, 3)},
        "evidence_chunks": evidence["evidence_chunks"],
        "kg_facts": evidence.get("kg_facts", []),
        "llm_stats": stats,
        "claims": steps,
    }
