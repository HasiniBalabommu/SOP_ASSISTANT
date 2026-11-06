# Separate answer verification in a local knowledge assistant

Code, question sets and stored per-answer results for the paper "Separate Answer Verification in a Local Knowledge
Assistant with Hybrid Retrieval: Effects on Fabrication with 1B and 3B Language Models".

## What is here

- `src/`, `backend_api/`, `frontend/`: the assistant.
- `eval/`: the evaluation harness. `questions_v2.json` holds the 59 institutional questions and
  `squad/questions_squad.json` the 120 SQuAD 2.0 questions (CC BY-SA 4.0).
- `eval/results_v2/`: the 4,282 stored answers behind the paper, one JSON line per answer, with a `_meta.json` file for
  each run.
- `eval/poison/`: the planted test document.
- `analysis/`: the statistics. Run them from this folder, for example `python analysis/review_stats.py out.json`.

## Not included

- The institutional corpus, its chunks and the built indexes, so the institutional runs cannot be repeated from this
  repository alone. The stored answers can still be re-analysed.
- The SQuAD development file, which `eval/make_squad_subset.py` expects as input. It is available from the
  [SQuAD site](https://rajpurkar.github.io/SQuAD-explorer/).
- The MiniCheck checkpoint (see `requirements-extra.txt`).

## Difference from the reported runs

The reported runs broke ties in Reciprocal Rank Fusion by Python set order. `src/hybrid_retriever.py` here breaks them by
chunk identifier, a fix made after those runs and not evaluated, so a repeat run can differ from the stored answers. The
`cove` configuration, `no_kg_no_verify` and `eval/replay_verifiers.py` were added for runs that were not completed; they
produced no result in the paper.

---

# Industrial Knowledge Assistant

A RAG- and Knowledge-Graph-based assistant that lets engineers and technicians
query an organization's SOPs, manuals, and troubleshooting guides in natural
language, with verified, cited answers.

> **Second round of experiments (v3):** see `RUN_EXPERIMENTS.md` to run the
> full experiment plan with one command, `CHANGES_v3.md` for what changed
> (including a session bug fix), and `RESEARCH_DIRECTIONS.md` for the
> research framing and related papers.

## Architecture

```
Document dropped in data/incoming/
        |
        v
   watcher.py  (auto-detects new files)
        |
        v
   ingest.py  (load -> clean -> chunk)
        |
        +--> embed_store.py     (semantic vector search, ChromaDB)
        +--> keyword_index.py   (exact-match search, BM25)
        +--> kg_builder.py      (entities & relationships, networkx)
        |
        v
   User asks a question (via frontend -> backend_api -> pipeline.py)
        |
        v
   retrieval_agent    -> hybrid_retriever + kg_retriever
        |
        v
   reasoning_agent     -> generates answer using local LLM (Ollama), grounded ONLY in retrieved evidence
        |
        v
   verification_agent  -> independently checks the answer against the evidence (hallucination check)
        |
        v
   citation_agent       -> attaches source document + page + excerpt
        |
        v
   Final answer + verdict + citations returned to the user
```

## Tech Stack

| Layer | Tool | Why |
|---|---|---|
| PDF/DOCX parsing | PyMuPDF, python-docx | Fast, accurate text extraction |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Free, local, no API cost |
| Vector store | ChromaDB | Persistent, supports incremental upsert |
| Keyword search | rank_bm25 | Catches exact terms (fault codes) embeddings can miss |
| Knowledge graph | networkx | Lightweight, sufficient for entity/relationship queries |
| LLM | Ollama (llama3) | Local, open-source, no API cost or data leaving the machine |
| Backend API | FastAPI | Simple, fast, auto-generated docs |
| Frontend | Plain HTML/CSS/JS | No build tooling required |

## Setup

```bash
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com) separately and pull a model:
```bash
ollama pull llama3
```

## Running the application

Four terminals, run concurrently:

```bash
# Terminal 1 — watches data/incoming/ and auto-processes new documents
python -m src.watcher

# Terminal 2 — local LLM (if not already running as a background service)
ollama serve

# Terminal 3 — backend API
python -m uvicorn backend_api.main:app --reload --port 8000

# Terminal 4 — frontend
cd frontend
python -m http.server 5500
```

Open **http://localhost:5500** in a browser.

## Usage

- **Upload documents**: drag a PDF/DOCX/TXT file into the sidebar upload box, or drop it directly into `data/incoming/`. It's processed automatically — no manual steps.
- **Ask questions**: type into the chat box. Each answer includes a verification verdict (SUPPORTED / UNSUPPORTED / PARTIALLY_SUPPORTED) and citations with source document + page.

## Evaluation

```bash
python -m eval.evaluate
```

Runs the ground-truth question set (`eval/qa_ground_truth.json`) through the full pipeline and reports:
- Retrieval Precision & Hit Rate
- Citation Accuracy
- Answer Keyword Score (approximate)
- Hallucination Rate
- Out-of-scope Correct Decline Rate
- Average Response Latency

Results are saved to `eval/results.json`.

### BM25 tokenizer check (paper, Section V-E)

```bash
python -m eval.retrieval_tokenizer_check --tag run1
```

Retrieval-only check (no LLM, Ollama not needed). Measures Hit Rate and Precision for hybrid, dense and BM25 retrieval on the 33 in-scope questions, first with the original whitespace tokenizer and then with punctuation stripped from token edges (`BM25_STRIP_PUNCTUATION` in `src/config.py`). Results are saved to `eval/results/retrieval_tokenizer_check_<tag>.json`; the five runs reported in the paper are `run1`–`run5`.

BM25 is deterministic, but dense results can differ each time the stored ChromaDB index is loaded in a new process, so run the check several times, each as a separate command.

`BM25_STRIP_PUNCTUATION` defaults to `False`, which reproduces the evaluated system and every result reported in the paper. Setting it to `True` makes `"E205?"` in a query match `"E205"` in a chunk.

## Known Limitations / Future Work

- Entity extraction in `kg_builder.py` is pattern-based (regex), not a trained NER model
- No true fine-tuning of the LLM; behavior is shaped via system prompts
- Single-tenant: all data lives in one shared `data/` folder (no per-organization isolation yet)
- `auth.py` (role-based access control) is implemented but not yet enforced in `backend_api/main.py`
- CPU-bound local LLM inference is the main latency bottleneck; a GPU or smaller quantized model would improve response time
- No streaming responses yet

## Project Structure

```
sop-assistant/
├── data/                  # ingestion pipeline data (auto-managed)
├── src/                   # core backend: ingestion, retrieval, KG, agents
│   └── agents/            # retrieval, reasoning, verification, citation agents
├── backend_api/           # FastAPI server
├── frontend/               # HTML/CSS/JS chat UI
├── eval/                   # ground-truth Q&A + evaluation harness
└── requirements.txt
```