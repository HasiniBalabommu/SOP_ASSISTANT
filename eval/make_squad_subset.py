"""
make_squad_subset.py — builds a second, public test set from SQuAD 2.0 so the
results do not rest only on the authors' own questions.

SQuAD 2.0 (Rajpurkar et al., ACL 2018; CC BY-SA 4.0) mixes answerable
questions with unanswerable ones that crowd workers wrote to look answerable
from the same Wikipedia article: the same "near-miss" situation as in
eval/questions_v2.json.

The script picks a fixed set of articles from the SQuAD 2.0 dev set, writes
each article as a TXT document (all its paragraphs, so the other paragraphs
act as distractors), and samples answerable and unanswerable questions from
those articles in this project's question format.

Usage:
    python -m eval.make_squad_subset                         # downloads dev-v2.0.json (~4 MB)
    python -m eval.make_squad_subset --dev path/to/dev-v2.0.json --articles 10 --per-type 60

Output:
    eval/squad/docs/<article>.txt     documents to index (SOP_DATA_DIR=data_squad_c100)
    eval/squad/questions_squad.json   questions (answerable + unanswerable)
"""

import argparse
import json
import random
import re
import urllib.request
from collections import Counter
from pathlib import Path

SQUAD_DEV_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
OUT_DIR = Path(__file__).parent / "squad"


def load_dev(path: str | None) -> dict:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    cached = OUT_DIR / "dev-v2.0.json"
    if not cached.exists():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {SQUAD_DEV_URL}")
        urllib.request.urlretrieve(SQUAD_DEV_URL, cached)
    return json.loads(cached.read_text(encoding="utf-8"))


def safe_name(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:60] + ".txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default=None, help="local dev-v2.0.json (otherwise downloaded)")
    parser.add_argument("--articles", type=int, default=10, help="number of Wikipedia articles to use")
    parser.add_argument("--per-type", type=int, default=60, help="answerable and unanswerable questions each")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    data = load_dev(args.dev)["data"]
    rng = random.Random(args.seed)
    articles = rng.sample(data, args.articles)

    docs_dir = OUT_DIR / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for old in docs_dir.glob("*.txt"):
        old.unlink()

    answerable, unanswerable = [], []
    for art in articles:
        fname = safe_name(art["title"])
        text = "\n\n".join(p["context"] for p in art["paragraphs"])
        (docs_dir / fname).write_text(art["title"].replace("_", " ") + "\n\n" + text, encoding="utf-8")
        for p in art["paragraphs"]:
            for qa in p["qas"]:
                if qa.get("is_impossible"):
                    unanswerable.append((fname, qa))
                elif qa["answers"]:
                    answerable.append((fname, qa))

    questions = []
    for fname, qa in rng.sample(answerable, min(args.per_type, len(answerable))):
        answer = Counter(a["text"] for a in qa["answers"]).most_common(1)[0][0]
        questions.append({"id": f"s{len(questions) + 1}", "query": qa["question"].strip(), "category": "squad",
                          "expected_sources": [fname], "expected_keywords": [answer],
                          "answers": sorted({a["text"] for a in qa["answers"]}), "squad_id": qa["id"]})
    for fname, qa in rng.sample(unanswerable, min(args.per_type, len(unanswerable))):
        questions.append({"id": f"s{len(questions) + 1}", "query": qa["question"].strip(), "category": "out_of_scope",
                          "oos_type": "near", "expected_sources": [], "expected_keywords": [],
                          "squad_id": qa["id"], "article": fname})

    (OUT_DIR / "questions_squad.json").write_text(json.dumps(questions, indent=2), encoding="utf-8")
    print(f"{len(articles)} articles -> {docs_dir}")
    print(f"{sum(q['category'] == 'squad' for q in questions)} answerable + "
          f"{sum(q['category'] == 'out_of_scope' for q in questions)} unanswerable questions -> "
          f"{OUT_DIR / 'questions_squad.json'}")


if __name__ == "__main__":
    main()
