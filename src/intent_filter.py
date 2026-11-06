"""
intent_filter.py — catches greetings and small talk BEFORE the query
hits retrieval/reasoning/verification, so simple inputs like "Hi" get
an instant response instead of running the full (slow, LLM-heavy) pipeline.
"""

import re
from src import llm_client

GREETING_PATTERNS = [
    r"^\s*(hi|hii+|hello+|hey+|yo|sup)\s*[!.?]*\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.?]*\s*$",
    r"^\s*(thanks|thank you|thx|ty)\s*[!.?]*\s*$",
    r"^\s*(bye|goodbye|see\s*you|cya)\s*[!.?]*\s*$",
    r"^\s*(how are you|what's up|whats up)\s*[!.?]*\s*$",
    r"^\s*(ok|okay|cool|nice|great)\s*[!.?]*\s*$",
]
PERSONA_SYSTEM_PROMPT = """You are the Industrial Knowledge Assistant, a friendly
and helpful AI that helps engineers and technicians find information from their
organization's maintenance manuals, SOPs, and troubleshooting guides.

For casual messages (greetings, thanks, small talk), respond warmly and briefly,
then invite the person to ask about equipment, fault codes, or procedures.
Keep it to 1-2 sentences. Do not make up technical information here."""

_compiled = [re.compile(p, re.IGNORECASE) for p in GREETING_PATTERNS]


def is_small_talk(query: str) -> bool:
    """True if the query is a greeting/small talk, not a real technical question."""
    query = query.strip()
    if len(query) == 0:
        return True
    if len(query.split()) > 6:
        return False  # long inputs are almost certainly real questions
    return any(pattern.match(query) for pattern in _compiled)


def small_talk_response(query: str) -> str:
    return llm_client.generate(query, system=PERSONA_SYSTEM_PROMPT, temperature=0.6)


if __name__ == "__main__":
    tests = ["Hi", "hello!", "What should I do if I see error code E205?", "thanks", "  ", "how do I fix the pump"]
    for t in tests:
        print(f"{t!r:45} -> small_talk={is_small_talk(t)}")