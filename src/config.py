"""
config.py — single source of truth for paths and settings.
Every other module imports from here instead of hardcoding values.

Experiment settings can be overridden with environment variables (set them
BEFORE starting Python), so the same code can build and query several index
variants without editing files. With no variables set, the defaults below
reproduce the evaluated system described in the paper.

  SOP_DATA_DIR               data folder (index, chunks, graph, manifest)
  SOP_CHUNK_SIZE             words per chunk                 (default 300)
  SOP_CHUNK_OVERLAP          words shared by adjacent chunks  (default 50)
  SOP_BM25_STRIP_PUNCTUATION 1 = strip punctuation from BM25 token edges
  SOP_GENERATOR_MODEL        Ollama model for answers        (default llama3.2:1b)
  SOP_VERIFIER_MODEL         Ollama model for verification   (default = generator)
  SOP_NUM_CTX                Ollama context length in tokens  (default: Ollama's own)
  SOP_NUM_GPU                Ollama GPU layers; 0 forces CPU  (default: Ollama's own)
  SOP_NUM_PREDICT            maximum tokens per model reply   (default: unlimited)
"""

import os
from pathlib import Path # pathlin is a pythons modern way of working with paths. It is more readable and less error-prone than using os.path.


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# --- Root paths ---
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = Path(os.environ.get("SOP_DATA_DIR", BASE_DIR / "data"))
if not DATA_DIR.is_absolute():
    DATA_DIR = BASE_DIR / DATA_DIR

INCOMING_DIR = DATA_DIR / "incoming"       # institutions drop files here
PROCESSED_DIR = DATA_DIR / "processed"     # archived after successful ingestion
CHUNKS_DIR = DATA_DIR / "chunks"           # one JSON per processed file
MANIFEST_PATH = DATA_DIR / "manifest.json" # tracks what's been ingested

# --- Chunking settings ---
CHUNK_SIZE = int(os.environ.get("SOP_CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.environ.get("SOP_CHUNK_OVERLAP", 50))

# --- BM25 tokenization ---
# False reproduces the evaluated system: whitespace split, so punctuation
# stays attached and "E205?" does not match "E205". True strips punctuation
# from token edges (measured post hoc in eval/retrieval_tokenizer_check.py).
BM25_STRIP_PUNCTUATION = _env_flag("SOP_BM25_STRIP_PUNCTUATION", False)

# --- Language models (served by Ollama) ---
GENERATOR_MODEL = os.environ.get("SOP_GENERATOR_MODEL", "llama3.2:1b")
VERIFIER_MODEL = os.environ.get("SOP_VERIFIER_MODEL") or None  # None = same as generator
NUM_CTX = int(os.environ["SOP_NUM_CTX"]) if os.environ.get("SOP_NUM_CTX") else None
NUM_GPU = int(os.environ["SOP_NUM_GPU"]) if os.environ.get("SOP_NUM_GPU") else None
# Without a cap a small model can fall into a repetition loop and generate for
# hours (seen with llama3.2:1b: >17,000 tokens for one answer).
NUM_PREDICT = int(os.environ["SOP_NUM_PREDICT"]) if os.environ.get("SOP_NUM_PREDICT") else None

# --- Supported file types ---
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
