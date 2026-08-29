"""Helper functions: PDF text extraction and chunking."""

import uuid
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from every page of a PDF."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks for embedding.

    chunk_size / overlap are in words, not characters, so chunks stay a
    reasonable size for the embedding model regardless of formatting.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += step
    return chunks


def generate_doc_id() -> str:
    return str(uuid.uuid4())[:8]
