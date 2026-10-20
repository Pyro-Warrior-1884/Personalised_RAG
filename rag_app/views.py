"""HTML views.

The views only prepare the context for the templates; all business logic
lives in the service layer.
"""

from __future__ import annotations

from django.conf import settings
from django.shortcuts import render

from rag_app.models import Conversation, KnowledgeBase, KnowledgeBaseStatus
from rag_app.services import conversation_service


def index(request):
    """Render the single page chat application."""
    knowledge_bases = list(KnowledgeBase.objects.all())

    ready_knowledge_bases = [
        kb for kb in knowledge_bases if kb.status == KnowledgeBaseStatus.READY
    ]

    selected_kb_id = request.GET.get("knowledge_base")
    selected_knowledge_base = None
    if selected_kb_id:
        selected_knowledge_base = next(
            (kb for kb in knowledge_bases if str(kb.id) == selected_kb_id), None
        )
    if selected_knowledge_base is None and ready_knowledge_bases:
        selected_knowledge_base = ready_knowledge_bases[0]

    conversations = list(
        Conversation.objects.select_related("knowledge_base").order_by("-updated_at")[
            : settings.CONVERSATION_PAGE_SIZE
        ]
    )

    active_conversation = None
    conversation_id = request.GET.get("conversation")
    if conversation_id:
        active_conversation = next(
            (item for item in conversations if str(item.id) == conversation_id), None
        )

    context = {
        "knowledge_bases": knowledge_bases,
        "ready_knowledge_bases": ready_knowledge_bases,
        "selected_knowledge_base": selected_knowledge_base,
        "conversations": conversations,
        "active_conversation": active_conversation,
        "allowed_source_roots": [str(root) for root in settings.ALLOWED_SOURCE_ROOTS],
        "supported_formats": [
            "PDF",
            "Excel (.xlsx)",
            "HTML",
            "Word (.docx)",
            "Text",
            "Markdown",
            "CSV",
            "Source code",
        ],
        "groq_model": settings.GROQ_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "retrieval_k": settings.RETRIEVAL_K,
    }
    return render(request, "rag_app/index.html", context)
