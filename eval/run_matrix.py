"""
run_matrix.py — runs the whole experiment plan in eval/matrix_default.json
(or another plan file) with one command, then writes the analysis report.

For every enabled job it:
  1. builds the index for the job's data folder if it does not exist yet;
  2. runs eval/experiment.py once per repeat, each in a NEW process (so the
     dense index is reloaded every time and run-to-run variation is real);
  3. finally runs eval/analyze.py on all results.
Everything is resumable: run the same command again after a crash or
restart and finished questions are skipped.

Usage (from the project folder, with Ollama running):
    python -m eval.run_matrix --dry-run          # print what would run
    python -m eval.run_matrix --limit 3 --out-dir eval/results_smoke   # quick smoke test
    python -m eval.run_matrix                    # full plan
    python -m eval.run_matrix --only g1b_core,g3b_core
"""

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def installed_models() -> set:
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            return {m["name"] for m in json.loads(r.read())["models"]}
    except Exception:
        return set()


def run(cmd: list[str], env: dict, dry: bool) -> None:
    shown = " ".join(f"{k}={v}" for k, v in env.items() if k.startswith("SOP_")) + " " + " ".join(cmd)
    print(f"\n>>> {shown}", flush=True)
    if not dry:
        subprocess.run(cmd, env={**os.environ, **env}, cwd=BASE_DIR, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default=str(Path(__file__).parent / "matrix_default.json"))
    parser.add_argument("--only", default=None, help="comma-separated job names to run")
    parser.add_argument("--out-dir", default="eval/results_v2")
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    only = set(args.only.split(",")) if args.only else None
    jobs = [j for j in plan["jobs"] if (j["name"] in only if only else j.get("enabled", True))]

    needed = {m for j in jobs for m in (j.get("generator"), j.get("verifier")) if m}
    have = installed_models()
    missing = sorted(m for m in needed if have and m not in have and f"{m}:latest" not in have)
    if not have:
        print("WARNING: could not reach Ollama at http://localhost:11434 — is it running?")
    elif missing:
        raise SystemExit("Missing Ollama models. Run first:\n" + "\n".join(f"  ollama pull {m}" for m in missing))

    py = sys.executable
    failed = []
    for job in jobs:
        env = {k: str(v) for k, v in {**plan.get("common_env", {}), **job.get("env", {})}.items()}
        questions = job.get("questions", plan["questions"])
        try:
            # a job may need its question set and documents generated first (e.g. SQuAD)
            if job.get("prepare_module") and not (BASE_DIR / questions).exists():
                run([py, "-m", job["prepare_module"]], env, args.dry_run)
            if job.get("module"):  # a job that is not eval.experiment (e.g. eval.replay_verifiers)
                cmd = [py, "-m", job["module"], *job.get("args", []), "--out-dir", args.out_dir]
                if args.limit:
                    cmd += ["--limit", str(args.limit)]
                run(cmd, env, args.dry_run)
                continue
            data_dir = BASE_DIR / env.get("SOP_DATA_DIR", "data")
            if not (data_dir / "chunks").exists():
                sources = [str(BASE_DIR / s) for s in job.get("index_sources", ["data/incoming"])]
                run([py, "-m", "eval.build_index", "--source", *sources], env, args.dry_run)
            start = job.get("start_repeat", 0)
            for r in range(start, start + job["repeats"]):
                cmd = [py, "-m", "eval.experiment", "--configs", ",".join(job["configs"]),
                       "--generator", job["generator"], "--repeats", "1", "--start-repeat", str(r),
                       "--run-name", job["name"], "--questions", questions, "--out-dir", args.out_dir]
                if job.get("verifier"):
                    cmd += ["--verifier", job["verifier"]]
                if job.get("legacy_shared_session"):
                    cmd += ["--legacy-shared-session"]
                if args.limit:
                    cmd += ["--limit", str(args.limit)]
                run(cmd, env, args.dry_run)
        except subprocess.CalledProcessError as e:
            # keep going: one failing job (e.g. a missing optional package) must not stop the rest
            print(f"\n!!! job {job['name']} failed (exit code {e.returncode}); continuing with the next job", flush=True)
            failed.append(job["name"])

    if failed:
        print(f"\nJobs that failed: {', '.join(failed)}. Fix the cause and rerun; finished questions are skipped.")
    out_files = sorted((BASE_DIR / args.out_dir).glob("*.jsonl"))
    if out_files and not args.dry_run:
        run([py, "-m", "eval.analyze", *map(str, out_files)], {}, False)


if __name__ == "__main__":
    main()
