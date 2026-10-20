"""Chunking of parsed documents.

Chunks keep every piece of metadata produced by the parsers and receive a
deterministic identifier so that re-indexing the same content never creates
duplicates.
"""

from __future__ import annotations

import hashlib
import logging

from django.conf import settings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

#: Separators used by the recursive splitter, from most to least preferred.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def build_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None) -> RecursiveCharacterTextSplitter:
    """Create the character splitter using the configured sizes."""
    size = int(chunk_size or settings.CHUNK_SIZE)
    overlap = int(chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP)
    if overlap >= size:
        overlap = max(0, size // 2)
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=DEFAULT_SEPARATORS,
        length_function=len,
        keep_separator=True,
    )


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into chunks while preserving their metadata."""
    if not documents:
        return []
    splitter = build_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    cleaned: list[Document] = []
    for chunk in chunks:
        content = (chunk.page_content or "").strip()
        if not content:
            continue
        metadata = dict(chunk.metadata or {})
        cleaned.append(Document(page_content=content, metadata=metadata))
    return cleaned


def make_chunk_id(knowledge_base_id: str, relative_path: str, file_hash: str, chunk_index: int) -> str:
    """Build a deterministic chunk id.

    The same knowledge base, file, content hash and chunk position always
    produce the same id, which is what makes re-indexing idempotent.
    """
    payload = f"{knowledge_base_id}|{relative_path}|{file_hash}|{chunk_index}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def assign_chunk_ids(chunks: list[Document], knowledge_base_id: str) -> list[Document]:
    """Attach ``chunk_id`` and ``chunk_index`` metadata to each chunk."""
    per_file_counter: dict[str, int] = {}
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        relative_path = str(metadata.get("relative_path", ""))
        file_hash = str(metadata.get("file_hash", ""))
        index = per_file_counter.get(relative_path, 0)
        per_file_counter[relative_path] = index + 1
        metadata["chunk_index"] = index
        metadata["chunk_id"] = make_chunk_id(str(knowledge_base_id), relative_path, file_hash, index)
        metadata["knowledge_base_id"] = str(knowledge_base_id)
        chunk.metadata = metadata
    return chunks
