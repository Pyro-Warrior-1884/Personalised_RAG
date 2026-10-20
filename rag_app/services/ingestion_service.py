"""The ingestion pipeline.

``run_ingestion`` walks a source directory, compares the content hash of
every file with the previous run, parses and embeds only what changed and
removes the vectors of files that disappeared.  The pipeline is idempotent:
running it twice over unchanged files does not duplicate any chunk.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from django.db import transaction
from django.utils import timezone as django_timezone

from rag_app.models import (
    IngestionJob,
    IngestionStage,
    IngestionStatus,
    IndexedFile,
    IndexedFileStatus,
    KnowledgeBase,
    KnowledgeBaseStatus,
)

from .chunking import assign_chunk_ids, split_documents
from .document_parser import load_file
from .file_discovery import discover_supported_files
from .source_manager import SourceValidationError, prepare_source
from .vector_store import (
    VectorStoreError,
    delete_file_chunks,
    get_vector_store,
    upsert_chunks,
)

logger = logging.getLogger(__name__)

HASH_CHUNK_SIZE = 1024 * 1024


@dataclass
class FileOutcome:
    """What happened to a single file during ingestion."""

    relative_path: str
    status: str
    chunk_count: int = 0
    error: str = ""


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hash of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(HASH_CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _set_stage(job: IngestionJob, stage: str) -> None:
    """Persist a stage change only when the stage actually moved."""
    if job.stage != stage:
        job.stage = stage
        job.save(update_fields=["stage", "updated_at"])


def _safe_error(exc: Exception) -> str:
    """Build a short, safe error summary (no stack traces, no file contents)."""
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    message = message[:300]
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def run_ingestion(job_id) -> IngestionJob:
    """Execute an ingestion job.  Returns the finished job."""
    try:
        job = IngestionJob.objects.select_related("knowledge_base").get(pk=job_id)
    except IngestionJob.DoesNotExist:
        logger.error("Ingestion job %s no longer exists", job_id)
        raise

    knowledge_base = job.knowledge_base
    job.status = IngestionStatus.RUNNING
    job.stage = IngestionStage.PREPARING
    job.started_at = job.started_at or _utcnow()
    job.error_summary = ""
    job.save(update_fields=["status", "stage", "started_at", "error_summary", "updated_at"])

    knowledge_base.status = KnowledgeBaseStatus.INDEXING
    knowledge_base.last_error = ""
    knowledge_base.save(update_fields=["status", "last_error", "updated_at"])

    try:
        source = prepare_source(knowledge_base.source_type, knowledge_base.source_location)
    except SourceValidationError as exc:
        return _fail_job(job, knowledge_base, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not prepare the source for job %s", job.id)
        return _fail_job(job, knowledge_base, _safe_error(exc))

    knowledge_base.source_metadata = {**knowledge_base.source_metadata, **source.metadata}
    knowledge_base.save(update_fields=["source_metadata", "updated_at"])

    root = source.root
    try:
        store = get_vector_store(knowledge_base)
    except VectorStoreError as exc:
        return _fail_job(job, knowledge_base, str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not open the vector store for job %s", job.id)
        return _fail_job(job, knowledge_base, _safe_error(exc))

    try:
        _set_stage(job, IngestionStage.DISCOVERING)
        discovered = discover_supported_files(root)
        job.files_discovered = len(discovered)
        job.save(update_fields=["files_discovered", "updated_at"])

        existing = {
            record.relative_path: record
            for record in IndexedFile.objects.filter(knowledge_base=knowledge_base)
        }
        discovered_set = set(discovered)

        processed = 0
        skipped = 0
        failed = 0
        chunks_created = 0
        errors: list[str] = []

        for relative_path in discovered:
            absolute_path = root / relative_path
            try:
                stat = absolute_path.stat()
                content_hash = compute_file_hash(absolute_path)
            except OSError as exc:
                failed += 1
                errors.append(f"{relative_path}: {exc.strerror or 'could not be read'}")
                _record_file(
                    knowledge_base,
                    relative_path,
                    status=IndexedFileStatus.FAILED,
                    error_message="The file could not be read.",
                    content_hash="",
                    size_bytes=0,
                    chunk_count=0,
                )
                continue

            previous = existing.get(relative_path)
            if (
                previous is not None
                and previous.status == IndexedFileStatus.INDEXED
                and previous.content_hash == content_hash
                and previous.chunk_count > 0
            ):
                # Unchanged file: reuse the existing vectors.
                skipped += 1
                previous.size_bytes = stat.st_size
                previous.modified_at = django_timezone.now()
                previous.save(update_fields=["size_bytes", "modified_at"])
                continue

            _set_stage(job, IngestionStage.PARSING)
            parse_result = load_file(absolute_path, relative_path)

            if not parse_result.ok:
                failed += 1
                reason = parse_result.reason or "unknown"
                message = parse_result.error or "The file could not be parsed."
                errors.append(f"{relative_path}: {message}")
                _record_file(
                    knowledge_base,
                    relative_path,
                    status=(
                        IndexedFileStatus.SKIPPED
                        if reason in {"unsupported", "empty"}
                        else IndexedFileStatus.FAILED
                    ),
                    error_message=message,
                    content_hash=content_hash,
                    size_bytes=stat.st_size,
                    chunk_count=0,
                )
                continue

            for document in parse_result.documents:
                document.metadata = {
                    **(document.metadata or {}),
                    "file_hash": content_hash,
                    "knowledge_base_id": str(knowledge_base.id),
                }

            _set_stage(job, IngestionStage.CHUNKING)
            chunks = split_documents(parse_result.documents)
            chunks = assign_chunk_ids(chunks, knowledge_base.id)

            _set_stage(job, IngestionStage.EMBEDDING)
            try:
                # Replace any previous vectors for this file before storing.
                delete_file_chunks(store, relative_path)
                stored = upsert_chunks(store, chunks)
            except VectorStoreError as exc:
                failed += 1
                errors.append(f"{relative_path}: {exc}")
                _record_file(
                    knowledge_base,
                    relative_path,
                    status=IndexedFileStatus.FAILED,
                    error_message="The file could not be stored in the vector database.",
                    content_hash=content_hash,
                    size_bytes=stat.st_size,
                    chunk_count=0,
                )
                continue

            _record_file(
                knowledge_base,
                relative_path,
                status=IndexedFileStatus.INDEXED,
                error_message="",
                content_hash=content_hash,
                size_bytes=stat.st_size,
                chunk_count=stored,
                file_type=parse_result.file_type,
            )
            processed += 1
            chunks_created += stored

        # ------------------------------------------------------------------
        # Remove vectors for files that are no longer present in the source.
        # ------------------------------------------------------------------
        _set_stage(job, IngestionStage.CLEANUP)
        chunks_deleted = 0
        for relative_path, record in existing.items():
            if relative_path in discovered_set:
                continue
            try:
                chunks_deleted += delete_file_chunks(store, relative_path)
            except VectorStoreError as exc:
                logger.warning("Could not remove stale chunks for %s: %s", relative_path, exc)
            record.delete()
            skipped += 1

        job.files_processed = processed
        job.files_skipped = skipped
        job.files_failed = failed
        job.chunks_created = chunks_created
        job.chunks_deleted = chunks_deleted
        job.status = IngestionStatus.COMPLETED
        job.stage = IngestionStage.COMPLETED
        job.finished_at = _utcnow()
        job.error_summary = summarise_errors(errors)
        job.save()

        knowledge_base.status = (
            KnowledgeBaseStatus.READY if (processed or skipped) else KnowledgeBaseStatus.EMPTY
        )
        knowledge_base.last_indexed_at = django_timezone.now()
        knowledge_base.last_error = job.error_summary
        knowledge_base.save(
            update_fields=["status", "last_indexed_at", "last_error", "updated_at"]
        )
        logger.info(
            "Ingestion %s finished: %s processed, %s skipped, %s failed, %s chunks.",
            job.id,
            processed,
            skipped,
            failed,
            chunks_created,
        )
        return job

    except Exception as exc:  # noqa: BLE001 - a job must never crash the worker
        logger.exception("Ingestion job %s failed", job.id)
        return _fail_job(job, knowledge_base, _safe_error(exc))


def summarise_errors(errors: list[str], limit: int = 10) -> str:
    """Build a concise error summary for the UI."""
    if not errors:
        return ""
    shown = errors[:limit]
    text = "; ".join(shown)
    if len(errors) > limit:
        text += f" (+{len(errors) - limit} more)"
    return text[:2000]


def _record_file(
    knowledge_base: KnowledgeBase,
    relative_path: str,
    *,
    status: str,
    error_message: str = "",
    content_hash: str = "",
    size_bytes: int = 0,
    chunk_count: int = 0,
    file_type: str = "",
) -> None:
    """Create or update the ``IndexedFile`` bookkeeping row."""
    record, _ = IndexedFile.objects.update_or_create(
        knowledge_base=knowledge_base,
        relative_path=relative_path,
        defaults={
            "status": status,
            "error_message": error_message[:1000],
            "content_hash": content_hash,
            "size_bytes": size_bytes,
            "chunk_count": chunk_count,
            "file_type": file_type,
            "modified_at": django_timezone.now(),
            "last_indexed_at": django_timezone.now(),
        },
    )
    return record


def _fail_job(job: IngestionJob, knowledge_base: KnowledgeBase, message: str) -> IngestionJob:
    """Mark a job (and its knowledge base) as failed."""
    job.status = IngestionStatus.FAILED
    job.stage = IngestionStage.FAILED
    job.error_summary = message[:2000]
    job.finished_at = _utcnow()
    job.save(update_fields=["status", "stage", "error_summary", "finished_at", "updated_at"])

    knowledge_base.status = KnowledgeBaseStatus.FAILED
    knowledge_base.last_error = message[:2000]
    knowledge_base.save(update_fields=["status", "last_error", "updated_at"])
    logger.error("Ingestion job %s failed: %s", job.id, message)
    return job


def reindex_knowledge_base(knowledge_base: KnowledgeBase) -> IngestionJob:
    """Create a new ingestion job for an existing knowledge base."""
    return IngestionJob.objects.create(
        knowledge_base=knowledge_base,
        status=IngestionStatus.QUEUED,
        stage=IngestionStage.QUEUED,
        trigger="reindex",
    )
