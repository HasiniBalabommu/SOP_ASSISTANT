"""
memory.py — simple in-memory conversation history per session, so
follow-up questions can be understood in context. Not persisted to
disk (yet) — resets when the app restarts, which is fine for now.
"""

MAX_TURNS = 5  # how many past exchanges to keep, to avoid overloading the prompt

# {session_id: [{"query": ..., "answer": ...}, ...]}
_sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    return _sessions.get(session_id, [])


def add_turn(session_id: str, query: str, answer: str) -> None:
    history = _sessions.setdefault(session_id, [])
    history.append({"query": query, "answer": answer})
    # keep only the most recent MAX_TURNS exchanges
    if len(history) > MAX_TURNS:
        _sessions[session_id] = history[-MAX_TURNS:]


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)


def format_history_for_prompt(session_id: str) -> str:
    """Returns the conversation history as plain text, ready to prepend to a prompt."""
    history = get_history(session_id)
    if not history:
        return ""
    lines = []
    for turn in history:
        lines.append(f"Previous question: {turn['query']}")
        lines.append(f"Previous answer: {turn['answer']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sid = "test_session"
    add_turn(sid, "What should I do if I see error code E205?", "Check coolant level and inspect the radiator fan.")
    add_turn(sid, "How often should I check the coolant?", "Weekly, as part of preventive maintenance.")

    print("History for session:")
    print(format_history_for_prompt(sid))

    clear_session(sid)
    print("\nAfter clearing:", get_history(sid))