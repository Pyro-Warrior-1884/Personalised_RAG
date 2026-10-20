"""Retrieval of relevant chunks from the vector store."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

from django.conf import settings

from .vector_store import VectorStoreError, get_vector_store, similarity_search

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk that was returned by the similarity search."""

    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0

    @property
    def relative_path(self) -> str:
        return str(self.metadata.get("relative_path", "unknown file"))

    @property
    def source_label(self) -> str:
        """A human readable label such as ``report.pdf (page 3)``."""
        label = self.relative_path
        page = self.metadata.get("page_number")
        if page:
            label = f"{label} (page {page})"
        sheet = self.metadata.get("sheet")
        if sheet:
            row_start = self.metadata.get("row_start")
            row_end = self.metadata.get("row_end")
            if row_start and row_end and row_end != row_start:
                label = f"{label} (sheet {sheet}, rows {row_start}-{row_end})"
            else:
                label = f"{label} (sheet {sheet})"
        elif self.metadata.get("file_type") == "code":
            language = self.metadata.get("language")
            if language:
                label = f"{label} ({language})"
        return label

    def to_source_dict(self) -> dict:
        """Return the JSON friendly representation used in API responses."""
        return {
            "label": self.source_label,
            "relative_path": self.relative_path,
            "file_type": str(self.metadata.get("file_type", "")),
            "page_number": self.metadata.get("page_number"),
            "sheet": self.metadata.get("sheet"),
            "row_start": self.metadata.get("row_start"),
            "row_end": self.metadata.get("row_end"),
            "score": round(float(self.score), 4),
            "rank": self.rank,
            "chunk_id": str(self.metadata.get("chunk_id", "")),
            "excerpt": (self.content or "")[:280],
        }


def _content_key(text: str) -> str:
    """Hash used to detect duplicated chunks."""
    normalised = " ".join((text or "").split()).lower()
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def retrieve_context(
    knowledge_base,
    question: str,
    k: int | None = None,
    deduplicate: bool = True,
) -> list[RetrievedChunk]:
    """Search the knowledge base and return the most relevant chunks."""
    top_k = int(k or settings.RETRIEVAL_K)
    if top_k <= 0:
        return []

    try:
        store = get_vector_store(knowledge_base)
    except VectorStoreError as exc:
        logger.warning("Could not open the vector store: %s", exc)
        return []

    hits = similarity_search(store, question, k=top_k)

    results: list[RetrievedChunk] = []
    seen: set[str] = set()
    for hit in hits:
        content = hit.document.page_content or ""
        if deduplicate:
            key = _content_key(content)
            if key in seen:
                continue
            seen.add(key)
        metadata = dict(hit.document.metadata or {})
        metadata.setdefault("relative_path", "unknown file")
        results.append(
            RetrievedChunk(
                content=content,
                metadata=metadata,
                score=float(hit.score),
                rank=len(results) + 1,
            )
        )
    return results
