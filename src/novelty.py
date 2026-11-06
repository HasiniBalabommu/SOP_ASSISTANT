"""
novelty.py — three abstention / verification methods studied in the second
round of experiments (see RESEARCH_DIRECTIONS.md). Each returns the same
dictionary shape as citation_agent.cite(), so the evaluation harness can
compare them directly with the original pipeline.

  identifier gate   : before any LLM call, every exact identifier in the
                      question (fault codes, model numbers, named units such
                      as "Generator C") must appear in the retrieved evidence;
                      otherwise the system declines. No model call, so it is
                      free, deterministic and cannot be talked out of it.
  sufficiency gate  : one LLM call asks whether the evidence contains enough
                      information to answer; if not, the system declines
                      before generating (after "sufficient context", ICLR 2025).
  claim verification: the draft is split into sentences and the verifier
                      checks each one separately, instead of judging the
                      whole answer at once.
  trained checker   : the same per-sentence check done by MiniCheck (Tang et
                      al., EMNLP 2024), a small fact-checking model, instead
                      of an LLM call. Needs requirements-extra.txt.
"""

import os
import re
import time

from src import config
from src import llm_client
from src import pipeline
from src.agents import citation_agent
from src.agents import retrieval_agent
from src.agents import verification_agent

# --- identifier gate --------------------------------------------------------

IDENTIFIER_PATTERNS = [
    re.compile(r"\b[A-Z]\d{2,4}\b"),                     # fault codes, e.g. E205
    re.compile(r"\b[A-Z]{1,4}-\d{2,5}\b"),               # model numbers, e.g. GB-500
    re.compile(r"\b(?i:generator|pump|compressor|motor|fan|unit|model|machine)\s+[A-Z0-9][A-Z0-9-]*\b"),
]

GATE_DECLINE_MESSAGE = (
    "I can't answer this from the current knowledge base: the question refers to "
    "{entities}, which does not appear in any of the retrieved documents."
)


def query_identifiers(query: str) -> list[str]:
    found = []
    for pattern in IDENTIFIER_PATTERNS:
        for m in pattern.finditer(query):
            if m.group(0) not in found:
                found.append(m.group(0))
    return found


def missing_identifiers(query: str, evidence_chunks: list[dict]) -> list[str]:
    text = " ".join(c["text"] for c in evidence_chunks).lower()
    return [e for e in query_identifiers(query) if e.lower() not in text]


def _decline(query, evidence, verdict, reason, message, t0, stats=None) -> dict:
    return {
        "query": query,
        "answer": message,
        "verification": {"verdict": verdict, "reason": reason},
        "declined": True,
        "citations": [],
        "timing": {"reasoning_sec": round(time.time() - t0, 3), "verification_sec": 0.0},
        "evidence_chunks": evidence["evidence_chunks"],
        "kg_facts": evidence.get("kg_facts", []),
        "llm_stats": stats or {},
    }


def gated_ask(query: str, session_id: str = "default", top_k: int = 5, mode: str = "hybrid",
              use_kg: bool = True, use_verification: bool = True) -> dict:
    t0 = time.time()
    evidence = retrieval_agent.retrieve(query, top_k=top_k, mode=mode, use_kg=use_kg)
    missing = missing_identifiers(query, evidence["evidence_chunks"])
    if missing:
        entities = ", ".join(f'"{e}"' for e in missing)
        return _decline(query, evidence, "GATE_DECLINED", f"Identifier(s) not in evidence: {entities}",
                        GATE_DECLINE_MESSAGE.format(entities=entities), t0)
    return pipeline.ask(query, session_id=session_id, top_k=top_k, mode=mode,
                        use_kg=use_kg, use_verification=use_verification)


# --- sufficiency gate -------------------------------------------------------

SUFFICIENCY_SYSTEM_PROMPT = """You check whether some EVIDENCE is enough to answer a
QUESTION. Do not answer the question. Reply with exactly one line:
SUFFICIENT: YES   if the evidence states everything needed for a complete answer
SUFFICIENT: NO    if anything needed is missing, or the evidence is about a
                  different item, code, model or topic than the one asked about"""

SUFFICIENCY_DECLINE_MESSAGE = (
    "I can't answer this from the current knowledge base: the retrieved documents "
    "do not contain enough information to answer it."
)


def _evidence_text(evidence: dict) -> str:
    return "\n\n".join(f"[Source: {c['metadata'].get('source')}]\n{c['text']}"
                       for c in evidence["evidence_chunks"])


def sufficiency_ask(query: str, session_id: str = "default", top_k: int = 5, mode: str = "hybrid",
                    use_kg: bool = True) -> dict:
    t0 = time.time()
    evidence = retrieval_agent.retrieve(query, top_k=top_k, mode=mode, use_kg=use_kg)
    llm_client.LAST_STATS.clear()
    raw = llm_client.generate(f"QUESTION: {query}\n\nEVIDENCE:\n{_evidence_text(evidence)}",
                              system=SUFFICIENCY_SYSTEM_PROMPT, temperature=0.0,
                              model=llm_client.VERIFIER_MODEL)
    stats = {"sufficiency": dict(llm_client.LAST_STATS)}
    gate_time = round(time.time() - t0, 3)
    m = re.search(r"SUFFICIENT:\s*(YES|NO)", raw, re.IGNORECASE)
    sufficient = (m.group(1).upper() == "YES") if m else not re.search(r"\bNO\b", raw, re.IGNORECASE)
    if not sufficient:
        result = _decline(query, evidence, "INSUFFICIENT", raw.strip()[:300],
                          SUFFICIENCY_DECLINE_MESSAGE, t0, stats)
    else:
        result = pipeline.ask(query, session_id=session_id, top_k=top_k, mode=mode, use_kg=use_kg)
        result["llm_stats"] = {**stats, **result.get("llm_stats", {})}
    result["timing"]["gate_sec"] = gate_time
    return result


# --- claim-level verification -----------------------------------------------

CLAIM_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given EVIDENCE and
one CLAIM. Reply with exactly one word: SUPPORTED if the evidence states the claim,
UNSUPPORTED if it does not (even if the claim sounds plausible or generally true)."""

MAX_CLAIMS = 8


def split_claims(answer: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", answer)
    claims = [p.strip(" -*\t") for p in parts]
    return [c for c in claims if len(c.split()) >= 4][:MAX_CLAIMS]


def _llm_claim_labels(claims: list[str], evidence: dict) -> list[bool]:
    labels = []
    for claim in claims:
        raw = llm_client.generate(f"EVIDENCE:\n{_evidence_text(evidence)}\n\nCLAIM: {claim}",
                                  system=CLAIM_SYSTEM_PROMPT, temperature=0.0,
                                  model=llm_client.VERIFIER_MODEL)
        labels.append("UNSUPPORTED" not in raw.upper())
    return labels


_minicheck = None
MINICHECK_MODEL = os.environ.get("SOP_MINICHECK_MODEL", "roberta-large")


def _minicheck_claim_labels(claims: list[str], evidence: dict) -> list[bool]:
    global _minicheck
    if not claims:
        return []
    if _minicheck is None:
        from minicheck.minicheck import MiniCheck  # pip install -r requirements-extra.txt
        _minicheck = MiniCheck(model_name=MINICHECK_MODEL, cache_dir=str(config.BASE_DIR / "ckpts"))
    doc = " ".join(c["text"] for c in evidence["evidence_chunks"])
    pred_labels, _probs, _, _ = _minicheck.score(docs=[doc] * len(claims), claims=claims)
    return [bool(label) for label in pred_labels]


def claim_verified_ask(query: str, session_id: str = "default", top_k: int = 5, mode: str = "hybrid",
                       use_kg: bool = True, checker: str = "llm") -> dict:
    """checker="llm" uses the verifier model per claim; "minicheck" uses MiniCheck."""
    t0 = time.time()
    draft = verification_agent.verify(query, session_id=session_id, top_k=top_k, mode=mode,
                                      use_kg=use_kg, use_verification=False)
    reasoning_time = time.time() - t0
    claims = split_claims(draft["answer"])
    t1 = time.time()
    if checker == "minicheck":
        labels = _minicheck_claim_labels(claims, draft["evidence"])
    else:
        labels = _llm_claim_labels(claims, draft["evidence"])
    if not claims or all(labels):
        verdict = "SUPPORTED"
    elif not any(labels):
        verdict = "UNSUPPORTED"
    else:
        verdict = "PARTIALLY_SUPPORTED"
    final_answer, declined = citation_agent._apply_verdict_policy(draft["answer"], verdict)
    citations = [] if declined else citation_agent._format_citations(draft["evidence"]["evidence_chunks"])
    return {
        "query": query,
        "answer": final_answer,
        "verification": {"verdict": verdict,
                         "reason": f"{sum(labels)}/{len(labels)} claims supported"},
        "declined": declined,
        "citations": citations,
        "timing": {"reasoning_sec": round(reasoning_time, 3), "verification_sec": round(time.time() - t1, 3)},
        "evidence_chunks": draft["evidence"]["evidence_chunks"],
        "kg_facts": draft["evidence"].get("kg_facts", []),
        "llm_stats": {**draft.get("llm_stats", {}), "claims": len(claims)},
        "claims": [{"claim": c, "supported": s} for c, s in zip(claims, labels)],
    }


if __name__ == "__main__":
    for q in ["What should I do if Generator B shows fault code E999?",
              "What is the oil capacity of Generator C?",
              "What should I do if I see error code E205?",
              "What is aliasing in Python lists?"]:
        print(f"{q!r:70} identifiers={query_identifiers(q)}")
