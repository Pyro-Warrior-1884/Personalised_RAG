"""Background task layer.

The MVP runs ingestion jobs in a background thread inside the Django
process.  This is intentionally isolated behind :func:`enqueue_ingestion` so
it can be replaced by Celery (or another durable queue) later without
touching the services.

The thread is **not** durable: if the server process dies, the job dies with
it.  :func:`reconcile_abandoned_jobs` marks such jobs as failed on the next
startup so the UI never shows a job that is stuck in "running".
"""
