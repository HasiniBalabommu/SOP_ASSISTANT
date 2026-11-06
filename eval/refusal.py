"""
refusal.py — phrase-based detection of answers that say the documents do not
contain the answer. Shared by eval/experiment.py (stored with each row) and
eval/analyze.py (re-applied to every stored answer, so results written with an
older pattern list are scored the same way).

Every out-of-scope response should still be read by hand (eval/analyze.py
writes them to a CSV for that purpose).
"""

import re

REFUSAL_PATTERNS = [
    r"does(?: not|n't) (?:contain|mention|provide|include|specify|say)",
    r"(?:is|are) not (?:mentioned|provided|specified|included|covered|available)",
    r"no (?:mention|information|details|data|reference|evidence)",
    r"(?:not|n't) (?:have|find) (?:enough|any|sufficient)? ?information",
    r"(?:can ?not|can't|could ?not|couldn't|unable to) (?:find|answer|determine|provide|confirm)",
    r"not going to be able",
    r"not enough information",
    r"\bi (?:do not|don't) know\b",
    r"outside (?:the|my) (?:scope|knowledge)",
    # added after reading llama3.2:3b answers ("I don't see any evidence about ...")
    r"(?:do not|don't|did not|didn't|am not|'m not) (?:see|seeing|find|finding) any "
    r"(?:evidence|information|mention|details|data|reference)",
]
_refusal_re = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(answer: str) -> bool:
    return bool(_refusal_re.search(answer or ""))
