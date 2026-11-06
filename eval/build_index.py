"""
build_index.py — one-shot ingestion of a folder of documents into the data
folder selected by SOP_DATA_DIR, using the chunk settings in SOP_CHUNK_SIZE /
SOP_CHUNK_OVERLAP. Same steps as the watcher (hash deduplication, chunking,
embedding, BM25, knowledge graph), but runs once and then reports how many
chunks are longer than the embedding model's 256 word-piece input limit.

Usage (Windows PowerShell):
    $env:SOP_DATA_DIR="data_c150"; $env:SOP_CHUNK_SIZE="150"; $env:SOP_CHUNK_OVERLAP="25"
    python -m eval.build_index --source data/incoming

Usage (Linux / macOS):
    SOP_DATA_DIR=data_c150 SOP_CHUNK_SIZE=150 SOP_CHUNK_OVERLAP=25 \
        python -m eval.build_index --source data/incoming
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from src import config
from src import embed_store
from src import keyword_index
from src import kg_builder
from src import manifest_store
from src.ingest import ingest_file

ENCODER_LIMIT = 256  # all-MiniLM-L6-v2 max_seq_length, in word pieces


def build(sources: list[Path]) -> None:
    print(f"Source folder(s): {', '.join(map(str, sources))}")
    print(f"Data folder     : {config.DATA_DIR}")
    print(f"Chunking        : {config.CHUNK_SIZE} words, overlap {config.CHUNK_OVERLAP}\n")

    manifest = manifest_store.load_manifest()
    paths = [p for source in sources for p in sorted(source.iterdir())]
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            continue
        if manifest_store.is_already_processed(path, manifest):
            print(f"skip (same content already indexed): {path.name}")
            continue
        chunks = ingest_file(path)
        chunks_json = config.CHUNKS_DIR / f"{path.stem}.json"
        n_emb = embed_store.embed_chunks_file(chunks_json)
        n_kw = keyword_index.add_chunks_file(chunks_json)
        n_kg = kg_builder.add_chunks_file(chunks_json)
        manifest = manifest_store.mark_processed(path, manifest, status="done")
        print(f"indexed {path.name}: {len(chunks)} chunks | embedded {n_emb} | bm25 {n_kw} | kg {n_kg}")

    tokenizer = embed_store.get_model().tokenizer
    lengths = []
    for f in config.CHUNKS_DIR.glob("*.json"):
        for c in json.loads(f.read_text()):
            lengths.append(len(tokenizer(c["text"])["input_ids"]))
    over = sorted(n - ENCODER_LIMIT for n in lengths if n > ENCODER_LIMIT)
    print(f"\nTotal chunks: {len(lengths)}")
    print(f"Chunks over the {ENCODER_LIMIT} word-piece limit: {len(over)}"
          + (f" (median excess {over[len(over) // 2]})" if over else ""))
    if over:
        print("Consider a smaller SOP_CHUNK_SIZE so that dense retrieval sees whole chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", nargs="+", default=[str(config.BASE_DIR / "data" / "incoming")],
                        help="one or more folders with the PDF / DOCX / TXT documents to index")
    args = parser.parse_args()
    build([Path(s) for s in args.source])
