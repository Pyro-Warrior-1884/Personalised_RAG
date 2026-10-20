"""Background execution of ingestion jobs."""

from __future__ import annotations

import logging
import threading

from django.db import close_old_connections

from rag_app.models import IngestionJob, IngestionStage, IngestionStatus

logger = logging.getLogger(__name__)

#: Keeps a reference to the worker threads so they are not garbage collected.
_active_threads: dict[str, threading.Thread] = {}

#: Only one ingestion job may touch the vector store at a time.  Overlapping
#: jobs raced on Chroma's shared client initialization and on SQLite writes,
#: so every worker waits for this lock before running.
_ingestion_lock = threading.Lock()


def _run_job(job_id: str) -> None:
    """Thread body: run the ingestion and clean up the DB connection."""
    from rag_app.services.ingestion_service import run_ingestion

    close_old_connections()
    try:
        # Show "Queued" in the UI while waiting for a previous job to finish.
        IngestionJob.objects.filter(pk=job_id).update(stage=IngestionStage.QUEUED)
        with _ingestion_lock:
            run_ingestion(job_id)
    except Exception:  # noqa: BLE001 - the thread must never die silently
        logger.exception("Ingestion job %s crashed", job_id)
        try:
            IngestionJob.objects.filter(pk=job_id, status=IngestionStatus.RUNNING).update(
                status=IngestionStatus.FAILED,
                stage=IngestionStage.FAILED,
                error_summary="The ingestion job crashed unexpectedly.",
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("Could not mark job %s as failed", job_id)
    finally:
        close_old_connections()
        _active_threads.pop(str(job_id), None)


def enqueue_ingestion(job: IngestionJob) -> IngestionJob:
    """Start a background worker for an ingestion job."""
    job.status = IngestionStatus.RUNNING
    job.stage = IngestionStage.PREPARING
    job.save(update_fields=["status", "stage", "updated_at"])

    thread = threading.Thread(
        target=_run_job,
        args=(str(job.id),),
        daemon=True,
        name=f"ingestion-{job.id}",
    )
    _active_threads[str(job.id)] = thread
    thread.start()
    logger.info("Queued ingestion job %s", job.id)
    return job


def is_job_running(job_id) -> bool:
    """Return True when a worker thread is still running for the job."""
    thread = _active_threads.get(str(job_id))
    return bool(thread and thread.is_alive())


def reconcile_abandoned_jobs() -> int:
    """Mark jobs that were running when the process stopped as failed.

    Called on application startup.  Returns the number of jobs that were
    reconciled.
    """
    try:
        from django.db import connection

        if "rag_app_ingestionjob" not in connection.introspection.table_names():
            # The database has not been migrated yet.
            return 0
        stale = IngestionJob.objects.filter(
            status__in=[IngestionStatus.RUNNING, IngestionStatus.QUEUED]
        )
        count = stale.count()
        if count:
            stale.update(
                status=IngestionStatus.FAILED,
                stage=IngestionStage.FAILED,
                error_summary=(
                    "The server restarted while this job was running. "
                    "Please retry the indexing."
                ),
            )
            logger.info("Marked %s abandoned ingestion job(s) as failed.", count)
        return count
    except Exception:  # pragma: no cover - startup must never crash
        logger.exception("Could not reconcile abandoned ingestion jobs")
        return 0
