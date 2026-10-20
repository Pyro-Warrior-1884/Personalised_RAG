"""The RAG orchestration layer.

``answer_question`` ties the pieces together: it validates that the
knowledge base is ready, retrieves the top chunks, decides whether the
evidence is strong enough to answer from, calls Groq and returns the answer
together with structured citations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from django.conf import settings

from rag_app.models import KnowledgeBase, KnowledgeBaseStatus

from .llm_service import LLMError, generate_answer
from .retrieval_service import RetrievedChunk, retrieve_context

logger = logging.getLogger(__name__)

INSUFFICIENT_EVIDENCE_ANSWER = (
    "I could not find enough information in the indexed material to answer that question. "
    "Try rephrasing it, or add the relevant documents to the knowledge base and re-index it."
)


class RAGError(Exception):
    """Raised when a question cannot be answered."""


@dataclass
class RAGResult:
    """The outcome of answering a question."""

    answer: str
    sources: list[dict] = field(default_factory=list)
    insufficient_evidence: bool = False
    retrieval: dict = field(default_factory=dict)


def ensure_ready(knowledge_base: KnowledgeBase) -> None:
    """Raise when the knowledge base cannot be used for answering."""
    if knowledge_base.status == KnowledgeBaseStatus.INDEXING:
        raise RAGError("This knowledge base is still being indexed. Please wait a moment.")
    if knowledge_base.status == KnowledgeBaseStatus.FAILED:
        raise RAGError(
            "This knowledge base could not be indexed"
            + (f": {knowledge_base.last_error}" if knowledge_base.last_error else ".")
        )
    if knowledge_base.status == KnowledgeBaseStatus.EMPTY:
        raise RAGError("This knowledge base is empty, so there is nothing to search yet.")
    if knowledge_base.status != KnowledgeBaseStatus.READY:
        raise RAGError("This knowledge base is not ready yet. Please index it first.")


def answer_question(
    knowledge_base: KnowledgeBase,
    question: str,
    history: list[dict] | None = None,
) -> RAGResult:
    """Answer a question using the knowledge base."""
    ensure_ready(knowledge_base)

    question = (question or "").strip()
    if not question:
        raise RAGError("The question is empty.")

    top_k = int(settings.RETRIEVAL_K)
    chunks: list[RetrievedChunk] = retrieve_context(knowledge_base, question, k=top_k)

    retrieval_info = {
        "k": top_k,
        "returned": len(chunks),
        "threshold": settings.RETRIEVAL_SCORE_THRESHOLD,
        "top_score": round(chunks[0].score, 4) if chunks else None,
    }

    if not chunks:
        logger.info("No chunks were retrieved for knowledge base %s", knowledge_base.id)
        return RAGResult(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=[],
            insufficient_evidence=True,
            retrieval=retrieval_info,
        )

    threshold = float(settings.RETRIEVAL_SCORE_THRESHOLD)
    best_score = chunks[0].score
    if threshold > 0 and best_score < threshold:
        logger.info(
            "Top score %.4f is below the threshold %.4f for knowledge base %s",
            best_score,
            threshold,
            knowledge_base.id,
        )
        return RAGResult(
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            sources=[chunk.to_source_dict() for chunk in chunks],
            insufficient_evidence=True,
            retrieval=retrieval_info,
        )

    try:
        answer = generate_answer(question, chunks, history=history)
    except LLMError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected failure while generating an answer")
        raise RAGError(f"The answer could not be generated: {type(exc).__name__}") from exc

    return RAGResult(
        answer=answer,
        sources=[chunk.to_source_dict() for chunk in chunks],
        insufficient_evidence=False,
        retrieval=retrieval_info,
    )
