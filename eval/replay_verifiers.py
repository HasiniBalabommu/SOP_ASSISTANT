"""
replay_verifiers.py — applies one verifier to drafts already stored from the
no_verify configuration, so that verifiers are compared on identical text and
evidence (a live run regenerates the draft, which mixes the verifier's effect
with run-to-run variation). The evidence is looked up by the stored chunk ids.
Drafts that are refusals, and repeats of an identical draft with identical
evidence, are skipped: the check is at temperature 0 and a refusal is not shown
as an answer either way.

Usage (with SOP_DATA_DIR set to the runs' index):
    python -m eval.replay_verifiers --runs g3b_core,g1b_core --verifier llama3.2:3b --run-name replay_inst_v3b
    python -m eval.replay_verifiers --runs squad_g3b --verifier minicheck --run-name replay_squad_vmc
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from src import keyword_index
from src import llm_client
from src import novelty
from src.agents import citation_agent
from src.agents import verification_agent

RESULTS = Path(__file__).parent / "results_v2"


def load_drafts(runs: list[str]) -> list[dict]:
    drafts, seen = [], set()
    for run in runs:
        for line in (RESULTS / f"{run}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["config"] != "no_verify" or r["refusal_text"] or not r["evidence"]:
                continue
            key = (r["generator"], r["id"], r["answer"], tuple(e["chunk_id"] for e in r["evidence"]))
            if key not in seen:
                seen.add(key)
                drafts.append({**r, "source_run": run})
    return drafts


def check(draft: dict, chunks: list[dict], verifier: str) -> dict:
    evidence = {"evidence_chunks": chunks}
    if verifier == "minicheck":
        claims = novelty.split_claims(draft["answer"])
        labels = novelty._minicheck_claim_labels(claims, evidence)
        verdict = "SUPPORTED" if not claims or all(labels) else \
            "UNSUPPORTED" if not any(labels) else "PARTIALLY_SUPPORTED"
        return {"verdict": verdict, "reason": f"{sum(labels)}/{len(labels)} claims supported"}
    prompt = verification_agent._build_verification_prompt(draft["query"], draft["answer"], evidence)
    raw = llm_client.generate(prompt, system=verification_agent.VERIFICATION_SYSTEM_PROMPT,
                              temperature=0.0, model=verifier)
    return verification_agent._parse_verdict(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, help="comma-separated stored runs")
    parser.add_argument("--verifier", required=True, help="an Ollama model, or minicheck")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default=str(RESULTS))
    args = parser.parse_args()

    index = keyword_index.load_index()
    by_id = {cid: {"text": t, "metadata": m}
             for cid, t, m in zip(index["chunk_ids"], index["texts"], index["metadatas"])}
    drafts = load_drafts(args.runs.split(","))
    if args.limit:
        drafts = drafts[:args.limit]

    out_path = Path(args.out_dir) / f"{args.run_name}.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["generator"], r["id"], r["draft"], tuple(r["evidence_ids"])))
    print(f"{args.run_name}: verifier {args.verifier}, {len(drafts)} drafts, {len(done)} done", flush=True)

    llm_client.SEED = 1000
    with open(out_path, "a", encoding="utf-8") as out:
        for d in drafts:
            ids = [e["chunk_id"] for e in d["evidence"]]
            if (d["generator"], d["id"], d["answer"], tuple(ids)) in done:
                continue
            chunks = [{**by_id[cid], "chunk_id": cid} for cid in ids]
            t0 = time.time()
            result = check(d, chunks, args.verifier)
            _, declined = citation_agent._apply_verdict_policy(d["answer"], result["verdict"])
            row = {"source_run": d["source_run"], "generator": d["generator"], "verifier": args.verifier,
                   "id": d["id"], "repeat": d["repeat"], "category": d["category"], "in_scope": d["in_scope"],
                   "draft": d["answer"], "evidence_ids": ids, "verdict": result["verdict"],
                   "verdict_reason": result["reason"], "declined": declined,
                   "latency_sec": round(time.time() - t0, 3)}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{d['source_run']} {d['id']:>5} {row['verdict']:<20} {row['latency_sec']:.1f}s", flush=True)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
