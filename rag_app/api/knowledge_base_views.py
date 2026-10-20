"""API endpoints for knowledge bases and their ingestion jobs."""

from __future__ import annotations

import logging
import uuid

from django.conf import settings
from django.db.models import Q
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from rag_app.models import (
    IngestionJob,
    IngestionStatus,
    IndexedFile,
    IndexedFileStatus,
    KnowledgeBase,
    KnowledgeBaseStatus,
    SourceType,
)
from rag_app.serializers import (
    IndexedFileSerializer,
    IngestionJobSerializer,
    KnowledgeBaseSerializer,
)
from rag_app.services.ingestion_service import reindex_knowledge_base
from rag_app.services.source_manager import SourceValidationError
from rag_app.services.vector_store import (
    VectorStoreError,
    build_collection_name,
    delete_knowledge_base_collection,
)
from rag_app.tasks.ingestion_tasks import enqueue_ingestion

from .serializers import (
    DeleteKnowledgeBaseSerializer,
    IngestionStatusResponseSerializer,
    KnowledgeBaseCreateSerializer,
    KnowledgeBaseResponseSerializer,
)

logger = logging.getLogger(__name__)


def _get_knowledge_base(kb_id) -> KnowledgeBase | None:
    return KnowledgeBase.objects.filter(pk=kb_id).first()


def _latest_job(knowledge_base: KnowledgeBase) -> IngestionJob | None:
    return knowledge_base.ingestion_jobs.order_by("-created_at").first()


def _status_payload(knowledge_base: KnowledgeBase) -> dict:
    job = _latest_job(knowledge_base)
    serializer = IngestionStatusResponseSerializer(
        {
            "job_id": job.id if job else None,
            "status": job.status if job else IngestionStatus.QUEUED,
            "stage": job.stage if job else "queued",
            "stage_display": job.get_stage_display() if job else "Queued",
            "files_discovered": job.files_discovered if job else 0,
            "files_processed": job.files_processed if job else 0,
            "files_skipped": job.files_skipped if job else 0,
            "files_failed": job.files_failed if job else 0,
            "chunks_created": job.chunks_created if job else 0,
            "error_summary": job.error_summary if job else "",
            "started_at": job.started_at if job else None,
            "finished_at": job.finished_at if job else None,
            "knowledge_base_status": knowledge_base.status,
        }
    )
    return serializer.data


class KnowledgeBaseListCreateView(APIView):
    """``GET`` lists knowledge bases, ``POST`` creates and indexes a new one."""

    def get(self, request):
        knowledge_bases = KnowledgeBase.objects.all()
        serializer = KnowledgeBaseSerializer(knowledge_bases, many=True)
        return Response({"results": serializer.data})

    def post(self, request):
        serializer = KnowledgeBaseCreateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        source_type = data["source_type"]
        source_location = data["source"]
        name = data["name"]

        if source_type == SourceType.GITHUB:
            source_location = data.get("resolved_source", source_location)

        knowledge_base = KnowledgeBase(
            name=name,
            source_type=source_type,
            source_location=source_location,
            collection_name=build_collection_name(uuid.uuid4()),
            status=KnowledgeBaseStatus.PENDING,
            source_metadata={},
        )
        knowledge_base.save()

        job = IngestionJob.objects.create(
            knowledge_base=knowledge_base,
            status=IngestionStatus.QUEUED,
            stage="queued",
            trigger="create",
        )

        try:
            enqueue_ingestion(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not start the ingestion job")
            job.status = IngestionStatus.FAILED
            job.stage = "failed"
            job.error_summary = f"The ingestion job could not be started: {type(exc).__name__}"
            job.save(update_fields=["status", "stage", "error_summary", "updated_at"])
            knowledge_base.status = KnowledgeBaseStatus.FAILED
            knowledge_base.last_error = job.error_summary
            knowledge_base.save(update_fields=["status", "last_error", "updated_at"])

        return Response(
            {
                "knowledge_base": KnowledgeBaseResponseSerializer(knowledge_base).data,
                "job": IngestionJobSerializer(job).data,
            },
            status=status.HTTP_202_ACCEPTED,
        )


class KnowledgeBaseDetailView(APIView):
    """Retrieve, and delete, a single knowledge base."""

    def get(self, request, kb_id):
        knowledge_base = _get_knowledge_base(kb_id)
        if knowledge_base is None:
            return Response(
                {"error": "not_found", "detail": "That knowledge base does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )
        payload = KnowledgeBaseSerializer(knowledge_base).data
        payload["latest_job"] = (
            IngestionJobSerializer(_latest_job(knowledge_base)).data
            if _latest_job(knowledge_base)
            else None
        )
        return Response(payload)

    def delete(self, request, kb_id):
        knowledge_base = _get_knowledge_base(kb_id)
        if knowledge_base is None:
            return Response(
                {"error": "not_found", "detail": "That knowledge base does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = DeleteKnowledgeBaseSerializer(data=request.data or {})
        serializer.is_valid(raise_exception=True)

        collection_name = knowledge_base.collection_name
        try:
            delete_knowledge_base_collection(knowledge_base)
        except VectorStoreError as exc:
            return Response(
                {"error": "vector_store_error", "detail": str(exc)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        knowledge_base.delete()
        logger.info("Deleted knowledge base %s (collection %s)", kb_id, collection_name)
        return Response(status=status.HTTP_204_NO_CONTENT)


class KnowledgeBaseStatusView(APIView):
    """Polling endpoint for ingestion progress."""

    def get(self, request, kb_id):
        knowledge_base = _get_knowledge_base(kb_id)
        if knowledge_base is None:
            return Response(
                {"error": "not_found", "detail": "That knowledge base does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(_status_payload(knowledge_base))


class KnowledgeBaseReindexView(APIView):
    """Queue an incremental re-index of an existing knowledge base."""

    def post(self, request, kb_id):
        knowledge_base = _get_knowledge_base(kb_id)
        if knowledge_base is None:
            return Response(
                {"error": "not_found", "detail": "That knowledge base does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            job = reindex_knowledge_base(knowledge_base)
            enqueue_ingestion(job)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Could not queue a re-index for %s", kb_id)
            return Response(
                {
                    "error": "reindex_failed",
                    "detail": f"The re-index could not be started: {type(exc).__name__}",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return Response(
            {"job": IngestionJobSerializer(job).data, "knowledge_base_id": str(knowledge_base.id)},
            status=status.HTTP_202_ACCEPTED,
        )


class KnowledgeBaseFilesView(APIView):
    """List the files that were indexed, skipped or failed."""

    def get(self, request, kb_id):
        knowledge_base = _get_knowledge_base(kb_id)
        if knowledge_base is None:
            return Response(
                {"error": "not_found", "detail": "That knowledge base does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )

        file_status = request.query_params.get("status")
        queryset = IndexedFile.objects.filter(knowledge_base=knowledge_base)
        if file_status:
            valid = {choice.value for choice in IndexedFileStatus.choices}
            if file_status not in valid:
                return Response(
                    {
                        "error": "invalid_status",
                        "detail": f"Unknown status '{file_status}'.",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
            queryset = queryset.filter(status=file_status)

        search = request.query_params.get("search")
        if search:
            queryset = queryset.filter(relative_path__icontains=search)

        try:
            page = int(request.query_params.get("page", 1))
        except (TypeError, ValueError):
            page = 1
        try:
            page_size = min(int(request.query_params.get("page_size", 100)), 500)
        except (TypeError, ValueError):
            page_size = 100
        page = max(page, 1)
        page_size = max(page_size, 1)

        total = queryset.count()
        offset = (page - 1) * page_size
        rows = list(queryset[offset : offset + page_size])
        serializer = IndexedFileSerializer(rows, many=True)

        counts = {
            value: IndexedFile.objects.filter(
                knowledge_base=knowledge_base, status=value
            ).count()
            for value, _label in IndexedFileStatus.choices
        }
        return Response(
            {
                "results": serializer.data,
                "total": total,
                "page": page,
                "page_size": page_size,
                "counts": counts,
            }
        )
