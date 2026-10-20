"""Request and response serializers for the knowledge base and chat APIs."""

from __future__ import annotations

from django.conf import settings
from rest_framework import serializers

from rag_app.models import KnowledgeBase, SourceType
from rag_app.services.source_manager import (
    SourceValidationError,
    validate_github_url,
    validate_local_directory,
)


class KnowledgeBaseCreateSerializer(serializers.Serializer):
    """Validates the body of ``POST /api/knowledge-bases/``."""

    name = serializers.CharField(max_length=200, allow_blank=False, trim_whitespace=True)
    source_type = serializers.ChoiceField(
        choices=[choice[0] for choice in SourceType.choices],
        default=SourceType.LOCAL_DIRECTORY,
    )
    source = serializers.CharField(max_length=2000, allow_blank=False, trim_whitespace=True)

    def validate_name(self, value: str) -> str:
        value = value.strip()
        if not value:
            raise serializers.ValidationError("Please give the knowledge base a name.")
        return value

    def validate(self, attrs):
        source_type = attrs.get("source_type")
        source = attrs.get("source", "")
        if source_type == SourceType.LOCAL_DIRECTORY:
            try:
                attrs["resolved_path"] = str(validate_local_directory(source))
            except SourceValidationError as exc:
                raise serializers.ValidationError({"source": str(exc)}) from exc
        elif source_type == SourceType.GITHUB:
            try:
                attrs["resolved_source"] = validate_github_url(source)
            except SourceValidationError as exc:
                raise serializers.ValidationError({"source": str(exc)}) from exc
        return attrs


class KnowledgeBaseResponseSerializer(serializers.ModelSerializer):
    """Compact representation returned after creating a knowledge base."""

    class Meta:
        model = KnowledgeBase
        fields = ["id", "name", "source_type", "source_location", "status", "created_at"]


class IngestionStatusResponseSerializer(serializers.Serializer):
    """The payload used by the UI to poll ingestion progress."""

    job_id = serializers.UUIDField(allow_null=True)
    status = serializers.CharField()
    stage = serializers.CharField()
    stage_display = serializers.CharField()
    files_discovered = serializers.IntegerField()
    files_processed = serializers.IntegerField()
    files_skipped = serializers.IntegerField()
    files_failed = serializers.IntegerField()
    chunks_created = serializers.IntegerField()
    error_summary = serializers.CharField(allow_blank=True)
    started_at = serializers.DateTimeField(allow_null=True)
    finished_at = serializers.DateTimeField(allow_null=True)
    knowledge_base_status = serializers.CharField()


class DeleteKnowledgeBaseSerializer(serializers.Serializer):
    """Confirms the deletion of a knowledge base."""

    confirm = serializers.BooleanField()

    def validate_confirm(self, value: bool) -> bool:
        if not value:
            raise serializers.ValidationError(
                "Set confirm to true to delete this knowledge base and its index."
            )
        return value


class ConversationCreateSerializer(serializers.Serializer):
    """Validates the body of ``POST /api/conversations/``."""

    knowledge_base_id = serializers.UUIDField(required=False, allow_null=True)
    title = serializers.CharField(max_length=200, required=False, allow_blank=True)

    def validate_knowledge_base_id(self, value):
        if value is None:
            return value
        if not KnowledgeBase.objects.filter(pk=value).exists():
            raise serializers.ValidationError("That knowledge base does not exist.")
        return value


class ConversationUpdateSerializer(serializers.Serializer):
    """Validates ``PATCH /api/conversations/{id}/``."""

    title = serializers.CharField(max_length=200, required=False, allow_blank=False)
    knowledge_base_id = serializers.UUIDField(required=False, allow_null=True)

    def validate_title(self, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise serializers.ValidationError("The title cannot be empty.")
        return value

    def validate_knowledge_base_id(self, value):
        if value is None:
            return value
        if not KnowledgeBase.objects.filter(pk=value).exists():
            raise serializers.ValidationError("That knowledge base does not exist.")
        return value


class MessageCreateSerializer(serializers.Serializer):
    """Validates ``POST /api/conversations/{id}/messages/``."""

    content = serializers.CharField(
        max_length=settings.MAX_QUESTION_LENGTH,
        allow_blank=False,
        trim_whitespace=True,
        error_messages={
            "blank": "Please type a question.",
            "max_length": (
                f"The question is too long (maximum {settings.MAX_QUESTION_LENGTH} characters)."
            ),
        },
    )

    def validate_content(self, value: str) -> str:
        value = value.strip()
        if not value:
            raise serializers.ValidationError("Please type a question.")
        return value


class ChatResponseSerializer(serializers.Serializer):
    """The payload returned after a question was answered."""

    message = serializers.DictField()
    sources = serializers.ListField()
    insufficient_evidence = serializers.BooleanField()
    retrieval = serializers.DictField()
