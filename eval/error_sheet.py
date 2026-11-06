"""
error_sheet.py — manual error analysis and human evaluation with two annotators.

  make       writes two CSV sheets from the results:
             error_sheet.csv       every likely failure (answered out-of-scope
                                   questions, low-keyword in-scope answers,
                                   answers that repeat a planted value), to be
                                   labelled with an error category;
             human_eval_sheet.csv  a random sample of answered questions, to be
                                   labelled correct / partial / incorrect.
             Each sheet has columns for annotator A and annotator B. Label them
             independently, without looking at each other's labels.
  summarize  reads the filled sheets and reports, per configuration and model,
             how often each error category occurs and the agreement between
             the annotators (Cohen's kappa).

Usage:
    python -m eval.error_sheet make eval/results_v2/*.jsonl --sample 120
    python -m eval.error_sheet summarize eval/results_v2/error_sheet.csv eval/results_v2/human_eval_sheet.csv
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from eval.refusal import is_refusal

ERROR_CATEGORIES = {
    "code_substitution": "answers about a different fault code or identifier (e.g. E999 answered with the E204 procedure)",
    "entity_swap": "answers with facts of a different item (e.g. Generator C answered with Generator B's values)",
    "invented_number": "states a number, threshold or quantity that no document gives",
    "invented_procedure": "describes steps or actions that no document gives",
    "outside_knowledge": "answers from general knowledge instead of the documents",
    "partial_or_wrong_doc": "uses the right document but misses or misreads the needed part",
    "poison_adopted": "repeats a value from the planted false document",
    "not_an_error": "the answer is actually acceptable (keyword score or refusal detection was wrong)",
    "other": "anything else (explain in notes)",
}
CORRECTNESS = ["correct", "partial", "incorrect", "declined"]


def load_rows(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:  # same refusal scoring as eval/analyze.py
        r["refusal_text"] = is_refusal(r["answer"])
    return rows


def likely_failure(r) -> bool:
    if r["declined"]:
        return False
    if not r["in_scope"]:
        return not r["refusal_text"]
    if r.get("poison_adopted"):
        return True
    return r.get("aks") is not None and r["aks"] < 0.5


def base_fields(r):
    return {"run": r["run_name"], "config": r["config"], "generator": r["generator"], "verifier": r["verifier"],
            "repeat": r["repeat"], "id": r["id"], "category": r["category"], "oos_type": r.get("oos_type") or "",
            "query": r["query"], "expected_keywords": "; ".join(r.get("expected_keywords", [])),
            "evidence_sources": "; ".join(sorted({e["source"] for e in r.get("evidence", [])})),
            "answer": r["answer"]}


def make(paths, sample: int, seed: int):
    rows = load_rows(paths)
    out_dir = Path(paths[0]).parent
    failures = [r for r in rows if likely_failure(r)]
    with open(out_dir / "error_sheet.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(base_fields(rows[0])) + ["error_A", "error_B", "notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in failures:
            w.writerow({**base_fields(r), "error_A": "", "error_B": "", "notes": ""})

    answered = [r for r in rows if not r["declined"]]
    picked = random.Random(seed).sample(answered, min(sample, len(answered)))
    with open(out_dir / "human_eval_sheet.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(base_fields(rows[0])) + ["correct_A", "correct_B", "notes"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in picked:
            w.writerow({**base_fields(r), "correct_A": "", "correct_B": "", "notes": ""})

    guide = ["# Labelling guide", "", "Label independently; do not look at the other annotator's column.", "",
             "## error_sheet.csv — columns error_A / error_B", ""]
    guide += [f"- `{k}`: {v}" for k, v in ERROR_CATEGORIES.items()]
    guide += ["", "## human_eval_sheet.csv — columns correct_A / correct_B", "",
              "- `correct`: fully answers the question and agrees with the documents",
              "- `partial`: right direction but incomplete or partly wrong",
              "- `incorrect`: wrong, invented, or answers a different question",
              "- `declined`: the answer says it cannot answer"]
    (out_dir / "LABELLING_GUIDE.md").write_text("\n".join(guide) + "\n", encoding="utf-8")
    print(f"{len(failures)} likely failures -> {out_dir / 'error_sheet.csv'}")
    print(f"{len(picked)} sampled answers -> {out_dir / 'human_eval_sheet.csv'}")
    print(f"Guide -> {out_dir / 'LABELLING_GUIDE.md'}")


def cohen_kappa(a: list[str], b: list[str]):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return None, 0
    n = len(pairs)
    observed = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (1.0 if expected == 1 else (observed - expected) / (1 - expected)), n


def summarize(paths):
    lines = []
    for p in paths:
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        if not rows:
            continue
        col_a, col_b = ("error_A", "error_B") if "error_A" in rows[0] else ("correct_A", "correct_B")
        kappa, n = cohen_kappa([r[col_a].strip() for r in rows], [r[col_b].strip() for r in rows])
        lines += [f"## {Path(p).name}", "",
                  f"Rows labelled by both annotators: {n}; Cohen's kappa = "
                  + (f"{kappa:.3f}" if kappa is not None else "n/a"), ""]
        groups = defaultdict(Counter)
        for r in rows:
            label = r[col_a].strip() or r[col_b].strip()
            if label:
                groups[(r["config"], r["generator"], r["verifier"])][label] += 1
        labels = sorted({l for c in groups.values() for l in c})
        lines += ["| Config | Generator | Verifier | " + " | ".join(labels) + " |",
                  "|" + "---|" * (3 + len(labels))]
        for key in sorted(groups):
            lines.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(str(groups[key][l]) for l in labels) + " |")
        lines.append("")
    out = Path(paths[0]).parent / "error_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("results", nargs="+")
    m.add_argument("--sample", type=int, default=120)
    m.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("summarize")
    s.add_argument("sheets", nargs="+")
    args = parser.parse_args()
    make(args.results, args.sample, args.seed) if args.cmd == "make" else summarize(args.sheets)
