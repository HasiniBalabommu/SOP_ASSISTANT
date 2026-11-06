"""
cleanup.py — weekly data retention: wipes all ingested document data
(raw files, chunks, vector store, keyword index, knowledge graph,
manifest) to limit how long confidential document content persists
in the system. Chat history (data/conversations/) is NEVER touched —
the Q&A record is preserved even after the underlying documents
that informed it are gone.
"""

import json
import shutil
from datetime import datetime, timezone, timedelta

from src import config
from src import embed_store
from src import keyword_index

CLEANUP_INTERVAL_DAYS = 7
LAST_CLEANUP_PATH = config.DATA_DIR / "last_cleanup.json"


def _read_last_cleanup() -> datetime | None:
    if not LAST_CLEANUP_PATH.exists():
        return None
    with open(LAST_CLEANUP_PATH, "r") as f:
        data = json.load(f)
    return datetime.fromisoformat(data["last_cleanup"])


def _write_last_cleanup(ts: datetime) -> None:
    LAST_CLEANUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_CLEANUP_PATH, "w") as f:
        json.dump({"last_cleanup": ts.isoformat()}, f, indent=2)


def is_due() -> bool:
    """True if a week (or more) has passed since the last cleanup, or
    if cleanup has never run before."""
    last = _read_last_cleanup()
    if last is None:
        return True
    return datetime.now(timezone.utc) >= last + timedelta(days=CLEANUP_INTERVAL_DAYS)


def _safe_rmtree(path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _safe_unlink(path) -> None:
    if path.exists():
        path.unlink()


def run_cleanup(force: bool = False) -> dict:
    """
    Wipes all document-derived data. Does NOT touch data/conversations/.
    Returns a summary of what was removed.
    """
    if not force and not is_due():
        return {"ran": False, "reason": "not due yet"}

    removed = []

    for folder in [config.INCOMING_DIR, config.PROCESSED_DIR]:
        if folder.exists():
            for f in folder.iterdir():
                if f.is_file():
                    f.unlink()
                    removed.append(str(f))

    if config.CHUNKS_DIR.exists():
        for f in config.CHUNKS_DIR.glob("*.json"):
            f.unlink()
            removed.append(str(f))

    vector_store_dir = config.DATA_DIR / "vector_store"
    _safe_rmtree(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    removed.append(str(vector_store_dir))

    kg_path = config.DATA_DIR / "kg" / "graph.pkl"
    _safe_unlink(kg_path)
    removed.append(str(kg_path))

    _safe_unlink(config.MANIFEST_PATH)
    removed.append(str(config.MANIFEST_PATH))

    # Invalidate in-memory caches so the running process doesn't keep
    # serving stale references to now-deleted files
    embed_store.reset_cache()
    keyword_index.reset_cache()

    now = datetime.now(timezone.utc)
    _write_last_cleanup(now)

    print(f"[cleanup] Weekly data retention cleanup ran at {now.isoformat()}")
    print(f"[cleanup] Removed {len(removed)} items. data/conversations/ was NOT touched.")

    return {"ran": True, "timestamp": now.isoformat(), "removed_count": len(removed)}


if __name__ == "__main__":
    # manual test — forces cleanup regardless of schedule
    result = run_cleanup(force=True)
    print(result)