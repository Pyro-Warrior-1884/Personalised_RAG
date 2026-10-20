"""Django admin registration for local inspection and troubleshooting."""

from django.contrib import admin

from rag_app.models import (
    Conversation,
    IngestionJob,
    IndexedFile,
    KnowledgeBase,
    Message,
)


@admin.register(KnowledgeBase)
class KnowledgeBaseAdmin(admin.ModelAdmin):
    list_display = ("name", "source_type", "status", "last_indexed_at", "created_at")
    list_filter = ("source_type", "status")
    search_fields = ("name", "source_location", "collection_name")
    readonly_fields = ("id", "collection_name", "created_at", "updated_at")


@admin.register(IngestionJob)
class IngestionJobAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "knowledge_base",
        "status",
        "stage",
        "files_discovered",
        "files_processed",
        "files_skipped",
        "files_failed",
        "chunks_created",
        "created_at",
    )
    list_filter = ("status", "stage")
    readonly_fields = ("id", "created_at", "updated_at")


@admin.register(IndexedFile)
class IndexedFileAdmin(admin.ModelAdmin):
    list_display = (
        "relative_path",
        "knowledge_base",
        "status",
        "chunk_count",
        "size_bytes",
        "last_indexed_at",
    )
    list_filter = ("status", "file_type")
    search_fields = ("relative_path",)


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("title", "knowledge_base", "created_at", "updated_at")
    search_fields = ("title",)
    readonly_fields = ("id", "created_at", "updated_at")


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("conversation", "role", "status", "created_at")
    list_filter = ("role", "status")
    readonly_fields = ("id", "created_at")
