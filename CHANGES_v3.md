# Changes in v3 (experiment kit)

## Bug fix that affects the original evaluation
`src/pipeline.py` accepted a `session_id` but did not pass it on, so
`cite() -> verify() -> reason()` always used the session `"default"`. In the
original evaluation every question therefore shared one conversation history
(the previous five questions, from any configuration), not one session per
question as `eval/evaluate.py` intended. The id is now passed through
`citation_agent.cite`, `verification_agent.verify` and `reasoning_agent.reason`.
`eval/experiment.py --legacy-shared-session` reproduces the old behaviour.

## New
- `eval/experiment.py` — new harness: fresh session per question, every repeat
  stored, per-repeat seeds, generator/verifier chosen per run, resumable,
  metrics on the evidence actually used, refusal detection from the answer text,
  software/hardware info saved with the results.
- `eval/run_matrix.py` + `eval/matrix_default.json` — whole plan in one command.
- `eval/analyze.py` — mean ± SD over repeats, Wilson intervals, exact McNemar
  tests, CSV of out-of-scope answers for manual checking.
- `eval/build_index.py` — builds an index variant (e.g. 100-word chunks) and
  reports chunks longer than the encoder's 256 word-piece limit.
- `eval/questions_v2.json` — the original 41 questions plus 18 near-miss
  out-of-scope questions (they sound answerable, but the documents do not
  contain the answer).
- `src/baselines.py` — no-retrieval baseline and same-call self-check baseline.
- `src/novelty.py` — three new abstention/verification methods: identifier
  gate (no LLM call), sufficiency gate, claim-level verification.
- `eval/poison/` + `eval/questions_poison.json` — planted-document test
  (a false service bulletin) for knowledge-poisoning robustness.
- `eval/build_index.py --source` accepts several folders; plan jobs can set
  `index_sources`.
- `eval/make_squad_subset.py` — public test set from SQuAD 2.0 (answerable +
  unanswerable questions over 10 Wikipedia articles), built at run time.
- MiniCheck trained checker as a verifier (`checker_verify`,
  `requirements-extra.txt`).
- Qwen 2.5 3B jobs (second model family) in the default plan.
- `eval/error_sheet.py` — error-category and human-evaluation sheets for two
  annotators, with counts and Cohen's kappa.
- `eval/run_matrix.py` continues with the next job if one fails, and can run a
  preparation step (`prepare_module`) before a job.
- `eval/matrix_laptop.json` — CPU-only plan for a laptop with installed models.
- `SOP_NUM_PREDICT` caps each model reply (512 tokens in the plans). Without it, llama3.2:1b fell into a
  repetition loop and generated >17,000 tokens for one answer; this is the likely cause of the
  5,594 s latency outlier in the original evaluation.
- `eval/analyze.py` separates *echo* answers (the answer only restates the question, e.g.
  "Fault code E207.") from fabrications. llama3.2:1b gives such answers to ~15–19% of
  out-of-scope questions; counting them as fabrications (fabrication = 1 − ENR) overstated the rate.
  Fabrication now counts only out-of-scope answers that state something.
- `eval/refusal.py` — the refusal patterns, shared by `experiment.py`, `analyze.py` and
  `error_sheet.py`. Added "I don't see any information/evidence/mention …" and "no evidence"
  after reading llama3.2:3b answers (6 of 464 3B labels changed, 5 correctly; 2 of 708 1B labels,
  1 of them wrongly). `analyze.py` and `error_sheet.py` re-score every stored answer with the
  current patterns, so all runs are scored the same way.
- `src/agents/verification_agent.py` — the verdict label is normalised ("PARTIALLY SUPPORTED",
  "partially-supported", "**Unsupported**." → PARTIALLY_SUPPORTED / UNSUPPORTED). llama3.2:3b wrote
  "PARTIALLY SUPPORTED" with a space in 12 of 177 `full` verdicts in g3b_core, 9 in gate_full and 12 in
  suff_full. Those answers were shown without the caveat. Declines were not affected, so fabrication and
  false-decline rates are unchanged. Found after g3b_core/g3b_novel/g3b_poison had run; squad_g3b was
  already running with the old parser.
- `RUN_EXPERIMENTS.md` — step-by-step instructions.

## Changed
- `src/config.py` — settings can be overridden with SOP_* environment
  variables (data folder, chunk size, BM25 tokenizer, models, context length).
  With none set, the defaults reproduce the evaluated system.
- `src/llm_client.py` — model per call (separate verifier model), seed and
  num_ctx options, token counts of the last call.
- `src/agents/citation_agent.py` — the result now includes the evidence chunks,
  graph facts and token counts.
- `requirements.txt` — pins transformers 4.57.6 and numpy 1.26.4 (as in the
  original environment).
- The Windows-specific `venv` folder is no longer included; create a new one
  with `pip install -r requirements.txt` (see RUN_EXPERIMENTS.md).

`eval/evaluate.py`, `eval/run_all_experiments.py` and all original result files
are kept unchanged. Because of the session fix, `eval/evaluate.py` now really
uses one session per question; to rerun the paper's setup as it was evaluated,
enable the `legacy_paper_setup` job in `eval/matrix_default.json`.
