"""Local embedding model management.

The embedding model runs locally through ``sentence-transformers``; only the
generated answer is sent to Groq.  The model is created lazily and cached so
that the first request does not pay the load cost on every call.
"""

from __future__ import annotations

import logging
import threading

from django.conf import settings

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


class EmbeddingError(Exception):
    """Raised when the local embedding model cannot be used."""


def get_embedding_model():
    """Return the cached ``HuggingFaceEmbeddings`` instance."""
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise EmbeddingError(
                "langchain-huggingface is not installed, so embeddings are unavailable."
            ) from exc

        model_name = settings.EMBEDDING_MODEL
        logger.info("Loading embedding model %s", model_name)
        try:
            _model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": settings.EMBEDDING_DEVICE},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": settings.EMBEDDING_BATCH_SIZE,
                },
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a clear error
            logger.exception("Could not load the embedding model %s", model_name)
            raise EmbeddingError(
                f"The embedding model '{model_name}' could not be loaded: {exc}"
            ) from exc
        logger.info("Embedding model %s ready", model_name)
        return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document chunks."""
    if not texts:
        return []
    return get_embedding_model().embed_documents(texts)


def embed_query(question: str) -> list[float]:
    """Embed a question with the same model used for the documents."""
    if not question or not question.strip():
        raise EmbeddingError("The question is empty, so it cannot be embedded.")
    return get_embedding_model().embed_query(question)


def reset_embedding_model() -> None:
    """Drop the cached model (used by the test suite)."""
    global _model
    with _model_lock:
        _model = None
