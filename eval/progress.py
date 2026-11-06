"""
progress.py — shows how far an experiment plan has got.

Usage:
    python -m eval.progress                              # laptop plan
    python -m eval.progress --plan eval/matrix_default.json
"""

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="eval/matrix_laptop.json")
    parser.add_argument("--out-dir", default="eval/results_v2")
    args = parser.parse_args()

    plan = json.loads((BASE_DIR / args.plan).read_text(encoding="utf-8"))
    total_done = total_all = 0
    for job in plan["jobs"]:
        if not job.get("enabled", True):
            continue
        qfile = BASE_DIR / job.get("questions", plan["questions"])
        n_q = len(json.loads(qfile.read_text(encoding="utf-8"))) if qfile.exists() else 0
        expected = n_q * len(job["configs"]) * job["repeats"]
        out = BASE_DIR / args.out_dir / f"{job['name']}.jsonl"
        done = sum(1 for l in out.read_text(encoding="utf-8").splitlines() if l.strip()) if out.exists() else 0
        total_done += done
        total_all += expected
        pct = 100 * done / expected if expected else 0
        print(f"{job['name']:16s} {done:5d} / {expected:5d}  ({pct:5.1f}%)")
    if total_all:
        print(f"{'TOTAL':16s} {total_done:5d} / {total_all:5d}  ({100 * total_done / total_all:5.1f}%)")


if __name__ == "__main__":
    main()
