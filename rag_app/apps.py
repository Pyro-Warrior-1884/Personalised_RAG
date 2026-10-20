"""Application configuration for the ``rag_app`` Django application."""

import logging

from django.apps import AppConfig
from django.core.signals import request_started
from django.dispatch import receiver

logger = logging.getLogger(__name__)


class RagAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "rag_app"
    verbose_name = "Personalised RAG"

    def ready(self) -> None:
        """Register the startup hook.

        The reconciliation is bound to the first incoming request instead of
        running inside ``ready()`` so that the database is never touched
        while the application registry is still being populated.
        """
        request_started.connect(
            _reconcile_once,
            dispatch_uid="rag_app.reconcile_abandoned_jobs",
            weak=False,
        )


@receiver(request_started, dispatch_uid="rag_app.reconcile_abandoned_jobs")
def _reconcile_once(sender, **kwargs):
    """Mark jobs that were running when the process stopped as failed.

    The MVP ingestion runner is an in-process thread, so a job that was
    running when the process died will never resume.  Marking those jobs as
    failed keeps the UI honest instead of showing a job that is stuck in
    "running" forever.  This is not a durable queue.
    """
    request_started.disconnect(
        _reconcile_once, dispatch_uid="rag_app.reconcile_abandoned_jobs"
    )
    try:
        from rag_app.tasks.ingestion_tasks import reconcile_abandoned_jobs

        reconcile_abandoned_jobs()
    except Exception:  # pragma: no cover - startup must never crash
        logger.exception("Failed to reconcile abandoned ingestion jobs on startup")
