"""
watcher.py — polls data/incoming/ for new files, ingests them automatically,
and archives them into data/processed/. This is what makes the pipeline
"plug-and-play": institutions just drop files in, nothing else needed.

"""
import time 
import shutil # this is used to move files from incoming to processed
from pathlib import Path

from src import config
from src import manifest_store
from src.ingest import ingest_file

from src import embed_store
from src import keyword_index
from src import kg_builder

from src import cleanup
POLL_INTERVAL_SECONDS = 5

def scan_and_process() -> int:
    manifest = manifest_store.load_manifest() # load the manifest of already processed files
    processed_count = 0

    for path in config.INCOMING_DIR.iterdir():
        if not path.is_file():
            continue # skip directories, only process files
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            print(f"Skipping unsupported file type: {path.name}")
            continue
        if manifest_store.is_already_processed(path, manifest): # is_already_processed checks if the file has already been processed by looking it up in the manifest its a built in function
            print(f"Skipping already processed file: {path.name}")
            continue

        print(f"new file detected: {path.name}, ingesting...")
        try:
            chunks = ingest_file(path)
            chunks_json_path = config.CHUNKS_DIR / f"{path.stem}.json"

            n_embedded = embed_store.embed_chunks_file(chunks_json_path)
            n_keyword = keyword_index.add_chunks_file(chunks_json_path)
            n_kg = kg_builder.add_chunks_file(chunks_json_path)

            manifest = manifest_store.mark_processed(path, manifest, status="done")
            print(f"  ingested -> {len(chunks)} chunks | embedded: {n_embedded} | "
                  f"keyword-indexed: {n_keyword} | kg relationships: {n_kg}")

        except Exception as e:
            print(f"  FAILED to ingest {path.name}: {e}")
            manifest_store.mark_processed(path, manifest, status="failed")

    return processed_count

def run_forever():
    print(f"Watching {config.INCOMING_DIR} every {POLL_INTERVAL_SECONDS}s. Ctrl+C to stop.")
    while True:
        scan_and_process()
        # if cleanup.is_due():
        #     cleanup.run_cleanup()
        # time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()
