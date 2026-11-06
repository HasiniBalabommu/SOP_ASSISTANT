"""
check_ingestion.py — run this from your project root to see the ground truth:
is generator_b_manual.txt actually in the index or not?

    python check_ingestion.py
"""

import chromadb
import json
from pathlib import Path

# From config.py: config.DATA_DIR / "vector_store", collection "sop_chunks"
PERSIST_DIR = "data/vector_store"

client = chromadb.PersistentClient(path=PERSIST_DIR)
collections = client.list_collections()
print("Collections found:", [c.name for c in collections])

if not collections:
    print(">>> No collections at all in the vector store. Nothing has been embedded yet.")
    raise SystemExit

collection = collections[0]  # adjust index if you have more than one
print(f"Using collection: {collection.name}\n")

data = collection.get()
metadatas = data.get("metadatas", [])

print(f"Total chunks in vector store: {len(metadatas)}")

sources = sorted(set(m.get("source", "UNKNOWN") for m in metadatas))
print("\nUnique source files currently indexed:")
for s in sources:
    print(" -", s)

if not any("generator_b_manual" in s for s in sources):
    print("\n>>> generator_b_manual.txt is NOT in the vector index. Embedding/upsert did not complete for this file.")
else:
    print("\n>>> generator_b_manual.txt IS present in the vector index.")

# Also check the manifest, which should be the authoritative ingestion log
manifest_path = Path("data/manifest.json")
if manifest_path.exists():
    with open(manifest_path) as f:
        manifest = json.load(f)
    print("\nManifest contents:")
    print(json.dumps(manifest, indent=2)[:2000])
else:
    print(f"\nNo manifest found at {manifest_path}")

# Also peek at the chunks file to confirm it has real content, not empty chunks
chunks_path = Path("data/chunks/generator_b_manual.json")
if chunks_path.exists():
    with open(chunks_path) as f:
        chunks = json.load(f)
    print(f"\ndata/chunks/generator_b_manual.json contains {len(chunks)} chunk(s).")
    if chunks:
        preview = chunks[0]
        print("First chunk preview:", json.dumps(preview, indent=2)[:500])
else:
    print("\nNo data/chunks/generator_b_manual.json found.")