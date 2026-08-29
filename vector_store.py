"""
Local vector store for AI Study Assistant.

Uses Sentence-Transformers to create embeddings and stores them locally.
No Pinecone API key or cloud service is required.
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
STORE_FILE = os.path.join("uploads", "vector_store.json")

_embed_model = None


def index_is_configured() -> bool:
    """Return True because the local vector store is always available."""
    return True


def get_embed_model() -> SentenceTransformer:
    global _embed_model

    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    return _embed_model


def _load_store() -> list:
    """Load stored document chunks."""
    if not os.path.exists(STORE_FILE):
        return []

    try:
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_store(data: list) -> None:
    """Save document chunks locally."""
    os.makedirs(os.path.dirname(STORE_FILE), exist_ok=True)

    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def add_documents(
    chunks: list[str],
    doc_id: str,
    source_name: str = ""
) -> int:
    """Embed and store document chunks locally."""

    model = get_embed_model()
    store = _load_store()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        store.append({
            "id": f"{doc_id}-{i}",
            "text": chunk,
            "embedding": embedding.tolist(),
            "doc_id": doc_id,
            "source": source_name,
            "chunk_index": i,
        })

    _save_store(store)

    return len(chunks)


def search_documents(query: str, top_k: int = 5) -> list[dict]:
    """Return the most relevant document chunks."""

    store = _load_store()

    if not store:
        return []

    model = get_embed_model()

    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    results = []

    for item in store:
        embedding = np.array(item["embedding"])

        score = float(np.dot(query_embedding, embedding))

        results.append({
            "text": item["text"],
            "score": score,
            "source": item.get("source", ""),
            "doc_id": item.get("doc_id"),
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]