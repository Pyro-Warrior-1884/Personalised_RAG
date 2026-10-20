"""Database models for Personalised RAG.

All application state that is not vector data lives in SQLite through the
Django ORM: knowledge bases, ingestion jobs, per-file index bookkeeping,
conversations and chat messages.
"""

from __future__ import annotations

import uuid

from django.db import models
from django.utils import timezone


class SourceType(models.TextChoices):
    """The kinds of sources that can be indexed."""

    LOCAL_DIRECTORY = "local_directory", "Local directory"
    GITHUB = "github", "GitHub repository"


class KnowledgeBaseStatus(models.TextChoices):
    """Lifecycle of a knowledge base."""

    PENDING = "pending", "Pending"
    INDEXING = "indexing", "Indexing"
    READY = "ready", "Ready"
    FAILED = "failed", "Failed"
    EMPTY = "empty", "Empty"


class IngestionStatus(models.TextChoices):
    """Lifecycle of a single ingestion job."""

    QUEUED = "queued", "Queued"
    RUNNING = "running", "Running"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"
    CANCELLED = "cancelled", "Cancelled"


class IngestionStage(models.TextChoices):
    """Progress stage of an ingestion job."""

    QUEUED = "queued", "Queued"
    PREPARING = "preparing", "Preparing source"
    DISCOVERING = "discovering", "Discovering files"
    PARSING = "parsing", "Parsing files"
    CHUNKING = "chunking", "Chunking documents"
    EMBEDDING = "embedding", "Embedding and storing"
    CLEANUP = "cleanup", "Cleaning up stale files"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class IndexedFileStatus(models.TextChoices):
    """Per-file indexing outcome."""

    INDEXED = "indexed", "Indexed"
    SKIPPED = "skipped", "Skipped"
    FAILED = "failed", "Failed"
    REMOVED = "removed", "Removed"


class MessageRole(models.TextChoices):
    """Who produced a chat message."""

    USER = "user", "User"
    ASSISTANT = "assistant", "Assistant"
    SYSTEM = "system", "System"


class MessageStatus(models.TextChoices):
    """Delivery status of a chat message."""

    COMPLETE = "complete", "Complete"
    ERROR = "error", "Error"


class KnowledgeBase(models.Model):
    """An indexable collection of documents backed by one Chroma collection."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    source_type = models.CharField(max_length=32, choices=SourceType.choices)
    source_location = models.TextField(help_text="Local path or repository URL as entered by the user.")
    collection_name = models.CharField(max_length=128, unique=True)
    status = models.CharField(
        max_length=32, choices=KnowledgeBaseStatus.choices, default=KnowledgeBaseStatus.PENDING
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    last_indexed_at = models.DateTimeField(null=True, blank=True)
    last_error = models.TextField(blank=True, default="")
    source_metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["status"]),
            models.Index(fields=["-updated_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.name} ({self.get_source_type_display()})"

    @property
    def is_ready(self) -> bool:
        return self.status == KnowledgeBaseStatus.READY

    @property
    def display_location(self) -> str:
        if self.source_type == SourceType.GITHUB:
            return self.source_location
        return str(self.source_metadata.get("resolved_path", self.source_location))


class IngestionJob(models.Model):
    """A single run of the ingestion pipeline, used for progress reporting."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    knowledge_base = models.ForeignKey(
        KnowledgeBase, on_delete=models.CASCADE, related_name="ingestion_jobs"
    )
    status = models.CharField(
        max_length=32, choices=IngestionStatus.choices, default=IngestionStatus.QUEUED
    )
    stage = models.CharField(
        max_length=32, choices=IngestionStage.choices, default=IngestionStage.QUEUED
    )
    trigger = models.CharField(max_length=32, default="create")
    files_discovered = models.PositiveIntegerField(default=0)
    files_processed = models.PositiveIntegerField(default=0)
    files_skipped = models.PositiveIntegerField(default=0)
    files_failed = models.PositiveIntegerField(default=0)
    chunks_created = models.PositiveIntegerField(default=0)
    chunks_deleted = models.PositiveIntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    error_summary = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["knowledge_base", "-created_at"]),
            models.Index(fields=["status"]),
        ]

    def __str__(self) -> str:
        return f"Job {self.id} for {self.knowledge_base_id} [{self.status}/{self.stage}]"

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            IngestionStatus.COMPLETED,
            IngestionStatus.FAILED,
            IngestionStatus.CANCELLED,
        }


class IndexedFile(models.Model):
    """Bookkeeping for one file inside a knowledge base."""

    id = models.BigAutoField(primary_key=True)
    knowledge_base = models.ForeignKey(
        KnowledgeBase, on_delete=models.CASCADE, related_name="indexed_files"
    )
    relative_path = models.CharField(max_length=1024)
    content_hash = models.CharField(max_length=64, blank=True, default="")
    size_bytes = models.BigIntegerField(default=0)
    modified_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=32, choices=IndexedFileStatus.choices, default=IndexedFileStatus.INDEXED
    )
    chunk_count = models.PositiveIntegerField(default=0)
    last_indexed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True, default="")
    file_type = models.CharField(max_length=32, blank=True, default="")

    class Meta:
        ordering = ["relative_path"]
        constraints = [
            models.UniqueConstraint(
                fields=["knowledge_base", "relative_path"], name="unique_file_per_knowledge_base"
            )
        ]
        indexes = [
            models.Index(fields=["knowledge_base", "status"]),
        ]

    def __str__(self) -> str:
        return f"{self.relative_path} [{self.status}]"


class Conversation(models.Model):
    """A chat session bound to an optional knowledge base."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=200, default="New chat")
    knowledge_base = models.ForeignKey(
        KnowledgeBase,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="conversations",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        indexes = [
            models.Index(fields=["-updated_at"]),
        ]

    def __str__(self) -> str:
        return self.title

    def touch(self) -> None:
        """Bump ``updated_at`` so the sidebar sorts by recent activity."""
        self.updated_at = timezone.now()
        self.save(update_fields=["updated_at"])


class Message(models.Model):
    """One message inside a conversation."""

    id = models.BigAutoField(primary_key=True)
    conversation = models.ForeignKey(
        Conversation, on_delete=models.CASCADE, related_name="messages"
    )
    role = models.CharField(max_length=16, choices=MessageRole.choices)
    content = models.TextField(blank=True, default="")
    sources = models.JSONField(default=list, blank=True)
    status = models.CharField(
        max_length=16, choices=MessageStatus.choices, default=MessageStatus.COMPLETE
    )
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [
            models.Index(fields=["conversation", "created_at"]),
        ]

    def __str__(self) -> str:
        preview = (self.content or "")[:40].replace("\n", " ")
        return f"{self.get_role_display()}: {preview}"
