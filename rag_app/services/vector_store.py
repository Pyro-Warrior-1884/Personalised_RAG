"""ChromaDB vector store access.

Each knowledge base owns exactly one Chroma collection whose name is stored
on the ``KnowledgeBase`` record, so the mapping between Django rows and
vector data stays stable across restarts.

The collection is created with cosine space so that the distance reported by
Chroma has a well defined meaning: ``score = 1 - distance`` is the cosine
similarity between the query and the chunk.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from langchain_core.documents import Document

from .embeddings import get_embedding_model

logger = logging.getLogger(__name__)

#: Chroma's process-wide shared client must be created exactly once per
#: persistence directory.  Creating it concurrently from several ingestion
#: threads corrupts its internal rust bindings, so initialization is guarded
#: and the client is reused everywhere.
_client_lock = threading.Lock()
_shared_client = None
_shared_client_path = None


class VectorStoreError(Exception):
    """Raised when the vector store cannot be used."""


@dataclass
class SearchHit:
    """A single retrieved chunk with its score and metadata."""

    document: Document
    score: float


def ensure_persist_directory() -> Path:
    """Create and return the Chroma persistence directory."""
    directory = Path(settings.CHROMA_PERSIST_DIRECTORY)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _get_chroma_client(directory: Path):
    """Return the shared ``PersistentClient`` for ``directory``, created once."""
    global _shared_client, _shared_client_path
    path = str(directory)
    if _shared_client is not None and _shared_client_path == path:
        return _shared_client
    with _client_lock:
        if _shared_client is None or _shared_client_path != path:
            import chromadb

            _shared_client = chromadb.PersistentClient(path=path)
            _shared_client_path = path
    return _shared_client


def build_collection_name(knowledge_base_id) -> str:
    """Return a stable, Chroma-safe collection name for a knowledge base."""
    prefix = settings.CHROMA_COLLECTION_PREFIX
    safe = "".join(char for char in str(knowledge_base_id) if char.isalnum() or char in {"-", "_"})
    return f"{prefix}_{safe}"[:60]


def get_vector_store(knowledge_base):
    """Open (or create) the Chroma collection that belongs to a knowledge base."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise VectorStoreError("langchain-chroma is not installed.") from exc

    directory = ensure_persist_directory()
    collection_name = knowledge_base.collection_name
    try:
        client = _get_chroma_client(directory)
        return Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=get_embedding_model(),
            collection_metadata={"hnsw:space": settings.CHROMA_SPACE},
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not open the Chroma collection %s", collection_name)
        raise VectorStoreError(f"The vector store could not be opened: {exc}") from exc


def _clean_metadata(metadata: dict) -> dict:
    """Drop empty metadata values, which Chroma rejects."""
    cleaned: dict = {}
    for key, value in (metadata or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[str(key)] = value
    return cleaned


def upsert_chunks(store, chunks: list[Document]) -> int:
    """Insert or replace chunks using their stable ids."""
    if not chunks:
        return 0

    ids: list[str] = []
    texts: list[str] = []
    metadatas: list[dict] = []
    for chunk in chunks:
        chunk_id = str((chunk.metadata or {}).get("chunk_id", "")).strip()
        if not chunk_id:
            continue
        ids.append(chunk_id)
        texts.append(chunk.page_content)
        metadatas.append(_clean_metadata(chunk.metadata))

    if not ids:
        return 0

    try:
        # Remove any previous rows with these ids so the operation is a true
        # upsert and never fails on duplicates.
        store.delete(ids=ids)
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to store %s chunks", len(ids))
        raise VectorStoreError(f"The chunks could not be stored: {exc}") from exc
    return len(ids)


def list_chunk_ids_for_file(store, relative_path: str) -> list[str]:
    """Return the ids of every chunk that belongs to one file."""
    try:
        result = store.get(
            where={"relative_path": {"$eq": relative_path}},
            include=["metadatas"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not list chunks for %s: %s", relative_path, exc)
        return []
    ids = result.get("ids") or []
    return [str(item) for item in ids]


def delete_file_chunks(store, relative_path: str) -> int:
    """Delete every chunk that belongs to one file.  Returns the count."""
    ids = list_chunk_ids_for_file(store, relative_path)
    if not ids:
        return 0
    try:
        store.delete(ids=ids)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not delete chunks for %s", relative_path)
        raise VectorStoreError(f"The chunks for {relative_path} could not be deleted: {exc}") from exc
    return len(ids)


def delete_knowledge_base_collection(knowledge_base) -> bool:
    """Remove the whole Chroma collection of a knowledge base."""
    try:
        from langchain_chroma import Chroma
    except ImportError as exc:  # pragma: no cover
        raise VectorStoreError("langchain-chroma is not installed.") from exc

    directory = ensure_persist_directory()
    try:
        client = _get_chroma_client(directory)
        store = Chroma(
            client=client,
            collection_name=knowledge_base.collection_name,
            embedding_function=get_embedding_model(),
            collection_metadata={"hnsw:space": settings.CHROMA_SPACE},
        )
        store.delete_collection()
        return True
    except Exception as exc:  # noqa: BLE001
        message = str(exc).lower()
        if "not exist" in message or "does not exist" in message or "not found" in message:
            logger.info("Collection %s was already gone", knowledge_base.collection_name)
            return False
        logger.exception("Could not delete the collection %s", knowledge_base.collection_name)
        raise VectorStoreError(f"The vector collection could not be deleted: {exc}") from exc


def similarity_search(store, question: str, k: int = 5) -> list[SearchHit]:
    """Search the collection and return the ``k`` closest chunks.

    Scores are cosine similarities in the range ``-1..1`` (higher is more
    similar) because the collection is created with cosine space.
    """
    try:
        raw = store.similarity_search_with_score(question, k=k)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Similarity search failed")
        raise VectorStoreError(f"The search failed: {exc}") from exc

    hits: list[SearchHit] = []
    for document, distance in raw:
        try:
            distance_value = float(distance)
        except (TypeError, ValueError):
            distance_value = 0.0
        if settings.CHROMA_SPACE == "cosine":
            score = 1.0 - distance_value
        else:
            # Euclidean distance is unbounded; map it to a 0..1 heuristic.
            score = 1.0 / (1.0 + distance_value)
        hits.append(SearchHit(document=document, score=score))
    return hits


def count_chunks(store) -> int:
    """Return the number of chunks currently stored in the collection."""
    try:
        return int(store._collection.count())  # noqa: SLF001 - chroma exposes no public count
    except Exception:  # noqa: BLE001
        return 0
