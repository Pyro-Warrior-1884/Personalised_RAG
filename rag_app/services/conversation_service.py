"""Conversation and message management backed by the Django ORM."""

from __future__ import annotations

import re
from typing import Iterable

from django.db import transaction
from django.utils import timezone

from rag_app.models import (
    Conversation,
    KnowledgeBase,
    Message,
    MessageRole,
    MessageStatus,
)


def create_conversation(
    knowledge_base: KnowledgeBase | None = None,
    title: str = "New chat",
) -> Conversation:
    """Create a blank conversation, optionally bound to a knowledge base."""
    return Conversation.objects.create(
        title=title or "New chat",
        knowledge_base=knowledge_base,
    )


def build_title_from_first_question(question: str, max_length: int = 60) -> str:
    """Derive a short title from the first question of a conversation."""
    text = " ".join((question or "").split())
    if not text:
        return "New chat"
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    if " " in truncated:
        truncated = truncated[: truncated.rfind(" ")]
    return truncated.rstrip(" .,;:") + "…"


def add_user_message(conversation: Conversation, content: str) -> Message:
    """Persist the user's question."""
    message = Message.objects.create(
        conversation=conversation,
        role=MessageRole.USER,
        content=content,
        status=MessageStatus.COMPLETE,
    )
    conversation.touch()
    return message


def add_assistant_message(
    conversation: Conversation,
    content: str,
    sources: Iterable[dict] | None = None,
    status: str = MessageStatus.COMPLETE,
    error_message: str = "",
) -> Message:
    """Persist the assistant's answer together with its sources."""
    message = Message.objects.create(
        conversation=conversation,
        role=MessageRole.ASSISTANT,
        content=content or "",
        sources=list(sources or []),
        status=status,
        error_message=error_message[:1000],
    )
    conversation.touch()
    return message


def load_conversation_history(conversation: Conversation, limit: int = 12) -> list[dict]:
    """Return the most recent messages of a conversation as plain dicts."""
    messages = list(conversation.messages.order_by("-created_at", "-id")[: max(0, limit)])
    messages.reverse()
    return [
        {
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at.isoformat() if message.created_at else None,
        }
        for message in messages
    ]


def rename_conversation(conversation: Conversation, title: str) -> Conversation:
    """Rename a conversation."""
    cleaned = " ".join((title or "").split())
    if not cleaned:
        raise ValueError("The title cannot be empty.")
    conversation.title = cleaned[:200]
    conversation.save(update_fields=["title", "updated_at"])
    return conversation


def set_conversation_knowledge_base(
    conversation: Conversation, knowledge_base: KnowledgeBase | None
) -> Conversation:
    """Bind a conversation to a knowledge base."""
    conversation.knowledge_base = knowledge_base
    conversation.save(update_fields=["knowledge_base", "updated_at"])
    return conversation


def delete_conversation(conversation: Conversation) -> None:
    """Delete a conversation and its messages."""
    conversation.delete()


@transaction.atomic
def prepare_conversation_for_message(
    conversation: Conversation, question: str
) -> None:
    """Set the title from the first question when the conversation is new."""
    if conversation.title in {"", "New chat"} and not conversation.messages.exists():
        conversation.title = build_title_from_first_question(question)
        conversation.save(update_fields=["title", "updated_at"])


def recent_conversations(limit: int = 50) -> list[Conversation]:
    """Return the most recently updated conversations."""
    return list(
        Conversation.objects.select_related("knowledge_base").order_by("-updated_at")[:limit]
    )


def conversation_to_dict(conversation: Conversation, with_messages: bool = False) -> dict:
    """Serialise a conversation for the API."""
    data = {
        "id": str(conversation.id),
        "title": conversation.title,
        "knowledge_base_id": str(conversation.knowledge_base_id) if conversation.knowledge_base_id else None,
        "knowledge_base_name": (
            conversation.knowledge_base.name if conversation.knowledge_base_id else None
        ),
        "created_at": conversation.created_at.isoformat() if conversation.created_at else None,
        "updated_at": conversation.updated_at.isoformat() if conversation.updated_at else None,
        "message_count": conversation.messages.count(),
    }
    if with_messages:
        data["messages"] = [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "sources": message.sources or [],
                "status": message.status,
                "error_message": message.error_message,
                "created_at": message.created_at.isoformat() if message.created_at else None,
            }
            for message in conversation.messages.order_by("created_at", "id")
        ]
    return data


def prune_empty_conversations(older_than_days: int = 7) -> int:
    """Delete old conversations that never received a message."""
    from datetime import timedelta

    cutoff = timezone.now() - timedelta(days=older_than_days)
    stale = Conversation.objects.filter(
        created_at__lt=cutoff,
        title="New chat",
        messages__isnull=True,
    )
    count = stale.count()
    stale.delete()
    return count
