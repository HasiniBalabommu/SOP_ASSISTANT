"""
manifest_store.py — tracks which files have already been ingested,
so the watcher never reprocesses the same document twice.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

from src import config


def _file_hash(path: Path) -> str:
    """Content-based hash — detects duplicates even if renamed,
    and detects edits even if the filename is reused."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    """Returns the manifest dict, creating an empty one if it doesn't exist yet."""
    if not config.MANIFEST_PATH.exists():
        return {}
    with open(config.MANIFEST_PATH, "r") as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def is_already_processed(path: Path, manifest: dict) -> bool:
    file_hash = _file_hash(path)
    return any(
        entry["hash"] == file_hash and entry["status"] == "done"
        for entry in manifest.values()
    )


def mark_processed(path: Path, manifest: dict, status: str = "done") -> dict:
    manifest[path.name] = {
        "hash": _file_hash(path),
        "status": status,  # "done" | "failed" | "pending"
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest)
    return manifest


if __name__ == "__main__":
    # quick manual test
    m = load_manifest()
    print("Current manifest:", m)