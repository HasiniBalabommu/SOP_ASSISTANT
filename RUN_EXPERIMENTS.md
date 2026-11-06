# Running the second round of experiments

This guide runs the full experiment plan on a stronger computer and collects
everything needed for the revised paper. All commands are run from this
`sop_assistant` folder.

## 1. What you need

| Item | Recommended |
|---|---|
| OS | Windows 10/11, Linux or macOS |
| Python | 3.10 or 3.11 |
| Ollama | latest version from https://ollama.com |
| GPU | NVIDIA with 8 GB+ VRAM (the 8B model is very slow on CPU) |
| RAM | 16 GB+ |
| Disk | about 15 GB free (models ~8 GB) |

## 2. One-time setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:1b
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

Ollama must be running in the background (on Windows it starts with the app;
on Linux run `ollama serve`). The first run downloads the embedding model
`all-MiniLM-L6-v2` (about 90 MB) from Hugging Face.

## 3. Smoke test (about 10–20 minutes)

```bash
python -m eval.run_matrix --limit 3 --out-dir eval/results_smoke
```

This builds the new index (`data_c100`) and runs every job on 3 questions.
If it finishes and prints a results table, everything works. Delete
`eval/results_smoke` afterwards.

## 4. Full run

```bash
python -m eval.run_matrix
```

To see the exact commands without running them: `python -m eval.run_matrix --dry-run`.

The plan is in `eval/matrix_default.json`:

| Job | Generator | Verifier | Configurations | Repeats |
|---|---|---|---|---|
| g1b_core | llama3.2:1b | same | no_retrieval, no_verify, self_check, full | 3 |
| g3b_core | llama3.2:3b | same | no_retrieval, no_verify, self_check, full | 3 |
| g8b_core | llama3.1:8b | same | no_retrieval, no_verify, self_check, full | 3 |
| g1b_verify8b | llama3.2:1b | llama3.1:8b | full | 3 |
| g3b_verify8b | llama3.2:3b | llama3.1:8b | full | 3 |
| g3b_novel / g8b_novel | 3B / 8B | same | gate_no_verify, gate_full, suff_full, claim_verify | 3 |
| g3b_poison / g8b_poison | 3B / 8B | same | no_verify, full, gate_full, suff_full, claim_verify (planted-document index) | 3 |
| squad_g3b / squad_g8b | 3B / 8B | same | no_retrieval, no_verify, full, self_check, gate_full, suff_full, claim_verify on SQuAD 2.0 | 3 |
| q3b_core / q3b_novel | qwen2.5:3b | same | core and new-method configurations (second model family) | 3 |
| g8b_ablations | llama3.1:8b | same | bm25, dense, no_kg | 3 |
| g3b_checker / g8b_checker | 3B / 8B | MiniCheck | checker_verify (needs `requirements-extra.txt`) | 3 |

New methods (`src/novelty.py`): **gate** = identifier gate (declines if a fault
code, model number or named unit in the question is missing from the evidence;
no LLM call); **suff** = sufficiency gate (one LLM call decides whether the
evidence is enough before answering); **claim_verify** = each sentence of the
draft is verified separately by the LLM; **checker_verify** = the same
per-sentence check done by MiniCheck, a small trained fact-checker. The poison
jobs index a planted, false service bulletin (`eval/poison/`) to test whether
verification can catch false evidence. The SQuAD jobs first run
`eval/make_squad_subset.py`, which downloads the SQuAD 2.0 dev set (~4 MB,
CC BY-SA 4.0) and builds 60 answerable + 60 unanswerable questions over 10
Wikipedia articles — a public test set next to the authors' own questions.

Roughly 15,000 model calls in total. Rough time: 15–25 hours with a good GPU
(two or three 12-hour Kaggle sessions; it resumes), many days on CPU only. You can run jobs separately, e.g.
`python -m eval.run_matrix --only g1b_core,g3b_core`.

**The run is resumable.** If the computer restarts or you stop it, run the
same command again: finished questions are skipped.

## 5. What to send back

Zip and send the whole `eval/results_v2` folder. It contains:

- `<job>.jsonl` — one line per question and repeat (full answers, verdicts, evidence, timings)
- `<job>_meta.json` — software versions, models, settings, hardware
- `report.md`, `summary.csv` — tables with mean ± SD, Wilson intervals, McNemar tests
- `oos_review.csv` — every out-of-scope answer, for manual checking (below)

## 6. Manual checks: error categories and human evaluation

```bash
python -m eval.error_sheet make eval/results_v2/*.jsonl --sample 120
```

This writes `error_sheet.csv` (every likely failure), `human_eval_sheet.csv`
(120 random answers) and `LABELLING_GUIDE.md` into `eval/results_v2`. Two
people fill the `_A` and `_B` columns **independently**, following the guide.
Then:

```bash
python -m eval.error_sheet summarize eval/results_v2/error_sheet.csv eval/results_v2/human_eval_sheet.csv
```

reports how often each error type occurs per method and model, and the
agreement between the two annotators (Cohen's kappa) — the human evaluation
the reviewers asked for. `oos_review.csv` (from analyze.py) lists every
out-of-scope answer for a quick check of the automatic refusal detection.

## 7. Using your own (better) documents and questions

1. Put the PDF / DOCX / TXT files in a new folder, e.g. `corpus_new/`.
2. Build an index for them:

   ```bash
   # Windows PowerShell
   $env:SOP_DATA_DIR="data_new"; $env:SOP_CHUNK_SIZE="100"; $env:SOP_CHUNK_OVERLAP="20"
   python -m eval.build_index --source corpus_new
   ```
   ```bash
   # Linux / macOS
   SOP_DATA_DIR=data_new SOP_CHUNK_SIZE=100 SOP_CHUNK_OVERLAP=20 python -m eval.build_index --source corpus_new
   ```
   It reports how many chunks are longer than the encoder limit; lower the
   chunk size if many are.
3. Write questions in the same format as `eval/questions_v2.json`:

   ```json
   {"id": "n1", "query": "What does alarm A12 mean on Pump P-3?", "category": "fault_code",
    "expected_sources": ["pump_manual.pdf"], "expected_keywords": ["low suction", "inlet valve"]}
   ```
   Out-of-scope questions use `"category": "out_of_scope"`, empty
   `expected_sources` and `expected_keywords`, and `"oos_type": "near"`
   (sounds answerable from the documents) or `"far"` (unrelated topic).
4. In `eval/matrix_default.json` set `"questions"` to your file and
   `"SOP_DATA_DIR"` to `data_new`, then run `python -m eval.run_matrix`.

## 8. Settings (environment variables)

| Variable | Meaning | Paper setting | New plan |
|---|---|---|---|
| SOP_DATA_DIR | index folder | data | data_c100 |
| SOP_CHUNK_SIZE / SOP_CHUNK_OVERLAP | words per chunk / overlap | 300 / 50 | 100 / 20 |
| SOP_BM25_STRIP_PUNCTUATION | punctuation-aware BM25 | 0 | 1 |
| SOP_NUM_CTX | Ollama context length | Ollama default | 4096 |
| SOP_GENERATOR_MODEL / SOP_VERIFIER_MODEL | models | llama3.2:1b | per job |
| SOP_NUM_GPU | GPU layers for Ollama; 0 = CPU only | not set | not set |
| SOP_NUM_PREDICT | maximum tokens per model reply (stops runaway repetition loops) | unlimited | 512 |

**Troubleshooting:** if Ollama crashes with a CUDA or "failed to allocate
buffer" error, the GPU is too small or its driver is unstable. Add
`"SOP_NUM_GPU": "0"` to `common_env` in `eval/matrix_default.json` to run on
the CPU (slower), or lower `SOP_NUM_CTX` to 2048.

The disabled job `legacy_paper_setup` reruns the original paper setup
(original index, original tokenizer, shared session) for comparison.
