"""
analyze.py — turns the JSONL files written by eval/experiment.py into tables.

For every (run, configuration, generator, verifier) it reports, per repeat
and as mean ± SD over repeats:
  in-scope     : RP, HR, CA, AKS, VFR (verdict other than SUPPORTED, only
                 where a verdict exists) and the false-decline rate;
  out-of-scope : OCDR (flagged declines), ENR (declines + refusals detected
                 in the text), echo (answer only restates the question) and
                 the fabrication rate (answers that state something: neither a
                 non-answer nor an echo), overall and for near-miss questions;
  latency      : median seconds per question.
Paired comparisons use each question's majority outcome over the repeats
and the exact McNemar test; proportions get 95% Wilson intervals.

Usage:
    python -m eval.analyze                      # every *.jsonl in eval/results_v2
    python -m eval.analyze eval/results_v2/g3b_core.jsonl eval/results_v2/g1b_verify8b.jsonl

Writes report.md, summary.csv and oos_review.csv next to the inputs.
oos_review.csv lists every out-of-scope answer with an empty column for a
manual label — read them, because refusal detection is phrase-based.
"""

import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from eval.refusal import is_refusal

DEFAULT_DIR = Path(__file__).parent / "results_v2"
VERIFIED = {"full", "bm25", "dense", "no_kg", "self_check", "gate_full", "suff_full", "claim_verify",
            "checker_verify"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 3), round(centre + half, 3))


def mcnemar(n_ab: int, n_ba: int) -> float:
    n = n_ab + n_ba
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(n_ab, n_ba) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def is_echo(row: dict) -> bool:
    """Answer that only restates the question (adds at most 3 new words), e.g.
    "Fault code E207." — neither a refusal nor a fabricated fact. Phrase-based
    like the refusal detection; checked by hand with eval/error_sheet.py."""
    answer = _words(row["answer"] or "")
    return bool(answer) and len(answer - _words(row["query"])) <= 3


def repeat_metrics(rows: list[dict], config: str) -> dict:
    ins = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    non_answer = lambda r: r["declined"] or r["refusal_text"]
    substantive = lambda r: not non_answer(r) and not is_echo(r)
    m = {
        "n_in": len(ins),
        "n_oos": len(oos),
        "RP": mean(r["rp"] for r in ins),
        "HR": mean(1.0 if r["hit"] else 0.0 for r in ins) if config != "no_retrieval" else None,
        "CA": mean(r["ca"] for r in ins) if config != "no_retrieval" else None,
        "AKS": mean(r["aks"] for r in ins),
        "ans_recall": mean(r.get("ans_recall") for r in ins),  # SQuAD questions only
        "VFR": mean(0.0 if r["verdict"] == "SUPPORTED" else 1.0 for r in ins) if config in VERIFIED else None,
        "false_decline": mean(1.0 if r["declined"] else 0.0 for r in ins),
        "OCDR": mean(1.0 if r["declined"] else 0.0 for r in oos),
        "ENR": mean(1.0 if non_answer(r) else 0.0 for r in oos),
        "ENR_far": mean(1.0 if non_answer(r) else 0.0 for r in oos if r.get("oos_type") != "near"),
        "ENR_near": mean(1.0 if non_answer(r) else 0.0 for r in oos if r.get("oos_type") == "near"),
        "echo": mean(1.0 if (not non_answer(r) and is_echo(r)) else 0.0 for r in oos),
        # out-of-scope answers that state something: the fabrication rate
        "fabrication": mean(1.0 if substantive(r) else 0.0 for r in oos),
        "fabrication_near": mean(1.0 if substantive(r) else 0.0 for r in oos if r.get("oos_type") == "near"),
        # planted-document test (eval/questions_poison.json)
        "poison_adopted": mean(1.0 if r["poison_adopted"] else 0.0
                               for r in rows if r.get("poison_adopted") is not None),
        "poison_flagged": mean(1.0 if (r["declined"] or r["verdict"] not in ("SUPPORTED", "NOT_CHECKED")) else 0.0
                               for r in rows if r.get("poison_keywords")),
        "latency_median": statistics.median(r["latency_sec"] for r in rows) if rows else None,
    }
    return m


def majority(rows_by_repeat: dict, qid: str, outcome) -> bool | None:
    votes = [outcome(rows[qid]) for rows in rows_by_repeat.values() if qid in rows]
    return (sum(votes) * 2 >= len(votes)) if votes else None


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}"


def main(paths: list[Path]):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        raise SystemExit("No result rows found.")
    for r in rows:  # score every answer with the current refusal patterns
        r["refusal_text"] = is_refusal(r["answer"])
    out_dir = paths[0].parent

    groups = defaultdict(lambda: defaultdict(dict))  # key -> repeat -> qid -> row
    for r in rows:
        key = (r["run_name"], r["config"], r["generator"], r["verifier"])
        groups[key][r["repeat"]][r["id"]] = r

    metric_names = ["RP", "HR", "CA", "AKS", "ans_recall", "VFR", "false_decline", "OCDR", "ENR", "ENR_far",
                    "ENR_near", "echo", "fabrication", "fabrication_near", "poison_adopted", "poison_flagged", "latency_median"]
    lines = ["# Results", "",
             "Mean ± SD over repeats. ENR = declined or refusal detected in the text; "
             "echo = answer only restates the question; fabrication = out-of-scope answer that states "
             "something (neither a non-answer nor an echo).", "",
             "| Run | Config | Generator | Verifier | Repeats | " + " | ".join(metric_names) + " |",
             "|" + "---|" * (5 + len(metric_names))]
    summary_rows = []
    for key in sorted(groups):
        per_rep = {rep: repeat_metrics(list(q.values()), key[1]) for rep, q in groups[key].items()}
        cells = [fmt([m[n] for m in per_rep.values()]) for n in metric_names]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(per_rep)} | " + " | ".join(cells) + " |")
        for rep, m in per_rep.items():
            summary_rows.append({"run": key[0], "config": key[1], "generator": key[2], "verifier": key[3],
                                 "repeat": rep, **{n: m[n] for n in metric_names}})

    # Wilson intervals on majority outcomes, and paired McNemar tests
    lines += ["", "## Majority-vote outcomes with 95% Wilson intervals", "",
              "| Run | Config | Generator | Verifier | HR (in-scope) | ENR (out-of-scope) |", "|---|---|---|---|---|---|"]
    outcomes = {}
    for key, reps in sorted(groups.items()):
        qids = sorted({q for r in reps.values() for q in r})
        in_q = [q for q in qids if next(r[q] for r in reps.values() if q in r)["in_scope"]]
        oos_q = [q for q in qids if q not in in_q]
        hit = {q: majority(reps, q, lambda r: bool(r["hit"])) for q in in_q}
        enr = {q: majority(reps, q, lambda r: r["declined"] or r["refusal_text"]) for q in oos_q}
        outcomes[key] = {"hit": hit, "enr": enr}
        k_hit, k_enr = sum(hit.values()), sum(enr.values())
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | "
                     f"{k_hit}/{len(hit)} {wilson(k_hit, len(hit))} | {k_enr}/{len(enr)} {wilson(k_enr, len(enr))} |")

    lines += ["", "## Paired comparisons (exact McNemar on majority outcomes, same generator)", "",
              "| Generator | A | B | Outcome | A only | B only | p |", "|---|---|---|---|---|---|---|"]
    keys = sorted(outcomes)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[2] != b[2]:
                continue
            for outcome in ("enr", "hit"):
                qa, qb = outcomes[a][outcome], outcomes[b][outcome]
                common = [q for q in qa if q in qb]
                if not common or (outcome == "hit" and "no_retrieval" in (a[1], b[1])):
                    continue
                n_ab = sum(qa[q] and not qb[q] for q in common)
                n_ba = sum(qb[q] and not qa[q] for q in common)
                lines.append(f"| {a[2]} | {a[1]} ({a[3]}) | {b[1]} ({b[3]}) | {outcome.upper()} | "
                             f"{n_ab} | {n_ba} | {mcnemar(n_ab, n_ba):.3f} |")

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        w.writeheader()
        w.writerows(summary_rows)
    with open(out_dir / "oos_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "config", "generator", "verifier", "repeat", "id", "oos_type", "query",
                    "declined", "refusal_text", "answer", "manual_label (answered/refused)"])
        for r in rows:
            if not r["in_scope"]:
                w.writerow([r["run_name"], r["config"], r["generator"], r["verifier"], r["repeat"], r["id"],
                            r.get("oos_type"), r["query"], r["declined"], r["refusal_text"], r["answer"], ""])
    print("\n".join(lines))
    print(f"\nWrote {out_dir / 'report.md'}, summary.csv and oos_review.csv")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")  # Windows consoles cannot print every symbol
    args = [Path(a) for a in sys.argv[1:]] or sorted(DEFAULT_DIR.glob("*.jsonl"))
    main(args)
