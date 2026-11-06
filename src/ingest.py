"""
this document does the processing of data
cleans and splits it into chunks and saves them in json format
and returns it
"""
import json
from pathlib import Path
import re  # regular expression library for text processing
from dataclasses import dataclass, asdict

import fitz  # PyMuPDF
from docx import Document as DocxDocument

from src import config


@dataclass  # create a class called chunk where each chunk stores info
class Chunk:
    chunk_id: str   # attributes of the chunk class
    source: str
    page: int | None
    text: str


def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Extracts text from a PDF file, returns list of (page_number, text) tuples."""
    doc = fitz.open(str(path))
    pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    doc.close()
    return pages
## in the above, built-in functions like get_text are used to extract text from each page and enumerate is used to number the pages


def load_docx(path: Path) -> list[tuple[int, str]]:
    """Extracts text from a DOCX file, returns list of (page_number, text) tuples."""
    docs = DocxDocument(str(path))  # fixed: was Document(...), which is not imported under that name
    text = "\n".join(p.text for p in docs.paragraphs)  # docx files are made up of paragraphs, so we join them together with newlines
    return [(1, text)]  # treat the whole document as a single "page"


def load_txt(path: Path) -> list[tuple[int, str]]:
    """Reads a TXT file, returns list of (page_number, text) tuples."""
    return [(1, path.read_text(errors="ignore"))]  # treat the whole document as a single "page"


def clean_txt(text: str) -> str:
    text = re.sub(r"\s+", " ", text)  # collapse whitespace ("\s" means spaces)
    text = re.sub(r"[^\x20-\x7E]", "", text)  # remove non-ASCII characters
    return text.strip()  # removes the spaces from the beginning and end of the string


def chunk_words(text: str, source: str, page: int | None, start_idx: int) -> list[Chunk]:
    words = text.split()  # split() method is used to split the text into words based on whitespace
    chunks = []
    i, idx = 0, start_idx
    while i < len(words):
        window = words[i:i + config.CHUNK_SIZE]  # create a window of words based on the chunk size
        if len(window) < 20:
            break  # don't create tiny chunks at the end
        chunks.append(Chunk(
            chunk_id=f"{source}_{idx:04d}",
            source=source,
            page=page,
            text=" ".join(window),
        ))
        idx += 1
        i += config.CHUNK_SIZE - config.CHUNK_OVERLAP  # move the window forward with overlap
    return chunks


def ingest_file(path: Path) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = load_pdf(path)
    elif suffix == ".docx":
        pages = load_docx(path)
    elif suffix == ".txt":
        pages = load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    all_chunks, idx = [], 0

    for page_num, raw_text in pages:
        text = clean_txt(raw_text)
        if not text:
            continue  # skip empty pages
        page_chunks = chunk_words(text, path.name, page_num, idx)
        idx += len(page_chunks)
        all_chunks.extend(page_chunks)

    # Save chunks for this file specifically
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.CHUNKS_DIR / f"{path.stem}.json"
    with open(out_path, "w") as f:
        json.dump([asdict(c) for c in all_chunks], f, indent=2)

    return all_chunks


if __name__ == "__main__":
    # manual test: drop one file in data/incoming/ and point to it here
    test_files = list(config.INCOMING_DIR.glob("*"))
    if not test_files:
        print(f"No files in {config.INCOMING_DIR} — drop one in to test.")
    else:
        chunks = ingest_file(test_files[0])
        print(f"Ingested {test_files[0].name}: {len(chunks)} chunks")