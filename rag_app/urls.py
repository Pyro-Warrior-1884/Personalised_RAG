"""URL routing for the Personalised RAG app.

The project includes this module under ``/api/`` (see ``config/urls.py``).
"""

from django.urls import path

from rag_app.api.chat_views import (
    ConversationDetailView,
    ConversationListCreateView,
    ConversationMessageView,
)
from rag_app.api.knowledge_base_views import (
    KnowledgeBaseDetailView,
    KnowledgeBaseFilesView,
    KnowledgeBaseListCreateView,
    KnowledgeBaseReindexView,
    KnowledgeBaseStatusView,
)

app_name = "rag_app"

urlpatterns = [
    path(
        "knowledge-bases/",
        KnowledgeBaseListCreateView.as_view(),
        name="knowledge-base-list",
    ),
    path(
        "knowledge-bases/<uuid:kb_id>/",
        KnowledgeBaseDetailView.as_view(),
        name="knowledge-base-detail",
    ),
    path(
        "knowledge-bases/<uuid:kb_id>/status/",
        KnowledgeBaseStatusView.as_view(),
        name="knowledge-base-status",
    ),
    path(
        "knowledge-bases/<uuid:kb_id>/reindex/",
        KnowledgeBaseReindexView.as_view(),
        name="knowledge-base-reindex",
    ),
    path(
        "knowledge-bases/<uuid:kb_id>/files/",
        KnowledgeBaseFilesView.as_view(),
        name="knowledge-base-files",
    ),
    path(
        "conversations/",
        ConversationListCreateView.as_view(),
        name="conversation-list",
    ),
    path(
        "conversations/<uuid:conversation_id>/",
        ConversationDetailView.as_view(),
        name="conversation-detail",
    ),
    path(
        "conversations/<uuid:conversation_id>/messages/",
        ConversationMessageView.as_view(),
        name="conversation-messages",
    ),
]
