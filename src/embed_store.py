"""
embed_store.py — turns text chunks into vector embeddings and stores
them in a persistent, incrementally-updatable vector database (ChromaDB).
"""

import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from src import config

# Loaded once, reused across calls — loading the model is slow,
# so we don't want to reload it every time we embed something.
_model = None
_client = None
_collection = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("Loading embedding model (first time only, may take a moment)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def reset_cache() -> None:
    """Forces the next get_collection() call to reopen a fresh connection —
    needed after the underlying vector store files have been deleted."""
    global _client, _collection
    _client = None
    _collection = None

    
def get_collection():
    """Returns the persistent Chroma collection, creating it if needed."""
    global _client, _collection
    if _collection is None:
        config.DATA_DIR.joinpath("vector_store").mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.DATA_DIR / "vector_store"))
        _collection = _client.get_or_create_collection(name="sop_chunks")
    return _collection


def embed_chunks_file(chunks_json_path: Path) -> int:
    """
    Reads a chunks JSON file (produced by ingest.py) and upserts every
    chunk into the vector store. Upsert = update if chunk_id already
    exists, insert if it's new — this is what makes re-ingestion safe.
    """
    with open(chunks_json_path, "r") as f:
        chunks = json.load(f)

    if not chunks:
        return 0

    model = get_model()
    collection = get_collection()

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"] or 0} for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return len(chunks)


def search(query: str, top_k: int = 5):
    """Semantic search: returns the top_k most similar chunks to the query."""
    model = get_model()
    collection = get_collection()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    return results


if __name__ == "__main__":
    # manual test: embed whatever chunk files already exist, then run a test query
    chunk_files = list(config.CHUNKS_DIR.glob("*.json"))
    if not chunk_files:
        print(f"No chunk files in {config.CHUNKS_DIR} — run ingest.py first.")
    else:
        total = 0
        for cf in chunk_files:
            n = embed_chunks_file(cf)
            print(f"Embedded {cf.name}: {n} chunks")
            total += n
        print(f"\nTotal embedded: {total}")

        print("\nTest query: 'low suction pressure'")
        results = search("low suction pressure", top_k=3)
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            print(f"  [{meta['source']}] {doc[:100]}...")