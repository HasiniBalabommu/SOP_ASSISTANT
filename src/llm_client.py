"""
llm_client.py — thin wrapper around a locally-running LLM via Ollama.
Every agent (reasoning, verification, citation) calls through this,
so the rest of the system doesn't care which model is underneath.
"""

import ollama

from src import config

MODEL_NAME = config.GENERATOR_MODEL        # set SOP_GENERATOR_MODEL to change
VERIFIER_MODEL = config.VERIFIER_MODEL     # None = use MODEL_NAME for verification too
SEED = None                                # set by the evaluation harness for repeatable runs

# Token counts and timing reported by Ollama for the most recent call, so the
# evaluation harness can log prompt length (and spot context truncation).
LAST_STATS: dict = {}


def generate(prompt: str, system: str = None, temperature: float = 0.2, model: str = None) -> str:
    """
    Sends a prompt to the local LLM and returns the plain text response.
    Low temperature by default — we want grounded, consistent technical
    answers, not creative variation.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {"temperature": temperature}
    if config.NUM_CTX:
        options["num_ctx"] = config.NUM_CTX
    if config.NUM_GPU is not None:
        options["num_gpu"] = config.NUM_GPU
    if config.NUM_PREDICT:
        options["num_predict"] = config.NUM_PREDICT
    if SEED is not None:
        options["seed"] = SEED

    used_model = model or MODEL_NAME
    response = ollama.chat(
        model=used_model,
        messages=messages,
        options=options,
    )
    LAST_STATS.clear()
    LAST_STATS.update({
        "model": used_model,
        "prompt_tokens": response.get("prompt_eval_count"),
        "output_tokens": response.get("eval_count"),
    })
    return response["message"]["content"]


if __name__ == "__main__":
    # manual test — confirms Ollama is reachable and responding
    test_prompt = "In one sentence, what is a compressor used for in industrial systems?"
    print(f"Prompt: {test_prompt}\n")
    answer = generate(test_prompt)
    print(f"Response: {answer}")
