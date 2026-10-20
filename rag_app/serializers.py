"""Serializers shared by the API and the HTML views."""

from __future__ import annotations

from django.conf import settings
from rest_framework import serializers

from rag_app.models import (
    Conversation,
    IngestionJob,
    IndexedFile,
    KnowledgeBase,
    Message,
)


class KnowledgeBaseSerializer(serializers.ModelSerializer):
    """Read serializer for a knowledge base."""

    source_type_display = serializers.CharField(source="get_source_type_display", read_only=True)
    display_location = serializers.CharField(read_only=True)
    file_count = serializers.SerializerMethodField()
    chunk_count = serializers.SerializerMethodField()
    conversation_count = serializers.SerializerMethodField()
    is_ready = serializers.BooleanField(read_only=True)

    class Meta:
        model = KnowledgeBase
        fields = [
            "id",
            "name",
            "source_type",
            "source_type_display",
            "source_location",
            "display_location",
            "collection_name",
            "status",
            "is_ready",
            "created_at",
            "updated_at",
            "last_indexed_at",
            "last_error",
            "source_metadata",
            "file_count",
            "chunk_count",
            "conversation_count",
        ]
        read_only_fields = fields

    def get_file_count(self, obj: KnowledgeBase) -> int:
        return obj.indexed_files.count()

    def get_chunk_count(self, obj: KnowledgeBase) -> int:
        return sum(record.chunk_count for record in obj.indexed_files.all())

    def get_conversation_count(self, obj: KnowledgeBase) -> int:
        return obj.conversations.count()


class IngestionJobSerializer(serializers.ModelSerializer):
    """Read serializer for an ingestion job (used for progress polling)."""

    status_display = serializers.CharField(source="get_status_display", read_only=True)
    stage_display = serializers.CharField(source="get_stage_display", read_only=True)
    is_terminal = serializers.BooleanField(read_only=True)

    class Meta:
        model = IngestionJob
        fields = [
            "id",
            "knowledge_base_id",
            "status",
            "status_display",
            "stage",
            "stage_display",
            "trigger",
            "files_discovered",
            "files_processed",
            "files_skipped",
            "files_failed",
            "chunks_created",
            "chunks_deleted",
            "started_at",
            "finished_at",
            "error_summary",
            "created_at",
            "is_terminal",
        ]
        read_only_fields = fields


class IndexedFileSerializer(serializers.ModelSerializer):
    """Read serializer for one indexed file."""

    status_display = serializers.CharField(source="get_status_display", read_only=True)

    class Meta:
        model = IndexedFile
        fields = [
            "id",
            "relative_path",
            "content_hash",
            "size_bytes",
            "modified_at",
            "status",
            "status_display",
            "chunk_count",
            "last_indexed_at",
            "error_message",
            "file_type",
        ]
        read_only_fields = fields


class MessageSerializer(serializers.ModelSerializer):
    """Read serializer for a chat message."""

    class Meta:
        model = Message
        fields = [
            "id",
            "role",
            "content",
            "sources",
            "status",
            "error_message",
            "created_at",
        ]
        read_only_fields = fields


class ConversationSerializer(serializers.ModelSerializer):
    """Read serializer for a conversation."""

    knowledge_base_name = serializers.CharField(
        source="knowledge_base.name", read_only=True, default=None
    )
    message_count = serializers.SerializerMethodField()
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "knowledge_base_id",
            "knowledge_base_name",
            "created_at",
            "updated_at",
            "message_count",
            "messages",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "message_count", "messages"]

    def get_message_count(self, obj: Conversation) -> int:
        return obj.messages.count()


class ConversationListSerializer(serializers.ModelSerializer):
    """Lighter serializer used for the sidebar list."""

    knowledge_base_name = serializers.CharField(
        source="knowledge_base.name", read_only=True, default=None
    )
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = [
            "id",
            "title",
            "knowledge_base_id",
            "knowledge_base_name",
            "created_at",
            "updated_at",
            "message_count",
        ]
        read_only_fields = fields

    def get_message_count(self, obj: Conversation) -> int:
        return obj.messages.count()


class ChatRequestSerializer(serializers.Serializer):
    """Validates the body of ``POST /api/conversations/{id}/messages/``."""

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
