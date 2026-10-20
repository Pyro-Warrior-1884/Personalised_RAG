"""API endpoints for conversations, messages and asking questions."""

from __future__ import annotations

import logging

from django.conf import settings
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from rag_app.models import Conversation, KnowledgeBase, MessageRole, MessageStatus
from rag_app.serializers import ConversationListSerializer, ConversationSerializer
from rag_app.services import conversation_service
from rag_app.services.llm_service import LLMError
from rag_app.services.rag_service import RAGError, answer_question

from .serializers import (
    ConversationCreateSerializer,
    ConversationUpdateSerializer,
    MessageCreateSerializer,
)

logger = logging.getLogger(__name__)


def _get_conversation(conversation_id) -> Conversation | None:
    return (
        Conversation.objects.select_related("knowledge_base")
        .filter(pk=conversation_id)
        .first()
    )


class ConversationListCreateView(APIView):
    """``GET`` lists conversations, ``POST`` creates a blank one."""

    def get(self, request):
        try:
            limit = min(int(request.query_params.get("limit", settings.CONVERSATION_PAGE_SIZE)), 200)
        except (TypeError, ValueError):
            limit = settings.CONVERSATION_PAGE_SIZE
        conversations = Conversation.objects.select_related("knowledge_base").order_by(
            "-updated_at"
        )[: max(1, limit)]
        serializer = ConversationListSerializer(conversations, many=True)
        return Response({"results": serializer.data})

    def post(self, request):
        serializer = ConversationCreateSerializer(data=request.data or {})
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        knowledge_base = None
        kb_id = data.get("knowledge_base_id")
        if kb_id:
            knowledge_base = KnowledgeBase.objects.filter(pk=kb_id).first()
            if knowledge_base is None:
                return Response(
                    {"error": "not_found", "detail": "That knowledge base does not exist."},
                    status=status.HTTP_404_NOT_FOUND,
                )

        title = (data.get("title") or "New chat").strip() or "New chat"
        conversation = conversation_service.create_conversation(
            knowledge_base=knowledge_base, title=title
        )
        return Response(
            ConversationSerializer(conversation).data,
            status=status.HTTP_201_CREATED,
        )


class ConversationDetailView(APIView):
    """Load, rename or delete a single conversation."""

    def get(self, request, conversation_id):
        conversation = _get_conversation(conversation_id)
        if conversation is None:
            return Response(
                {"error": "not_found", "detail": "That conversation does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )
        return Response(ConversationSerializer(conversation).data)

    def patch(self, request, conversation_id):
        conversation = _get_conversation(conversation_id)
        if conversation is None:
            return Response(
                {"error": "not_found", "detail": "That conversation does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = ConversationUpdateSerializer(data=request.data or {}, partial=True)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        if "title" in data:
            conversation = conversation_service.rename_conversation(conversation, data["title"])
        if "knowledge_base_id" in data:
            kb_id = data["knowledge_base_id"]
            knowledge_base = (
                KnowledgeBase.objects.filter(pk=kb_id).first() if kb_id else None
            )
            conversation = conversation_service.set_conversation_knowledge_base(
                conversation, knowledge_base
            )
        return Response(ConversationSerializer(conversation).data)

    def delete(self, request, conversation_id):
        conversation = _get_conversation(conversation_id)
        if conversation is None:
            return Response(
                {"error": "not_found", "detail": "That conversation does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )
        conversation_service.delete_conversation(conversation)
        return Response(status=status.HTTP_204_NO_CONTENT)


class ConversationMessageView(APIView):
    """Ask a question inside a conversation."""

    def post(self, request, conversation_id):
        conversation = _get_conversation(conversation_id)
        if conversation is None:
            return Response(
                {"error": "not_found", "detail": "That conversation does not exist."},
                status=status.HTTP_404_NOT_FOUND,
            )

        serializer = MessageCreateSerializer(data=request.data or {})
        serializer.is_valid(raise_exception=True)
        question = serializer.validated_data["content"]

        knowledge_base = conversation.knowledge_base
        if knowledge_base is None:
            return Response(
                {
                    "error": "knowledge_base_required",
                    "detail": "Select a knowledge base before asking a question.",
                },
                status=status.HTTP_409_CONFLICT,
            )

        # The user's message is stored first so a failed answer never loses it.
        conversation_service.prepare_conversation_for_message(conversation, question)
        user_message = conversation_service.add_user_message(conversation, question)

        history = conversation_service.load_conversation_history(conversation, limit=10)
        # The current question is already the last entry of the history.
        history = history[:-1] if history else []

        try:
            result = answer_question(knowledge_base, question, history=history)
        except LLMError as exc:
            logger.warning("The LLM request failed: %s", exc)
            conversation_service.add_assistant_message(
                conversation,
                content="",
                sources=[],
                status=MessageStatus.ERROR,
                error_message=str(exc),
            )
            return Response(
                {
                    "error": "llm_error",
                    "detail": str(exc),
                    "retryable": True,
                    "message": {
                        "id": user_message.id,
                        "role": MessageRole.USER,
                        "content": question,
                    },
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )
        except RAGError as exc:
            conversation_service.add_assistant_message(
                conversation,
                content="",
                sources=[],
                status=MessageStatus.ERROR,
                error_message=str(exc),
            )
            return Response(
                {
                    "error": "rag_error",
                    "detail": str(exc),
                    "retryable": False,
                    "message": {
                        "id": user_message.id,
                        "role": MessageRole.USER,
                        "content": question,
                    },
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        assistant_message = conversation_service.add_assistant_message(
            conversation,
            content=result.answer,
            sources=result.sources,
            status=MessageStatus.COMPLETE,
        )

        return Response(
            {
                "message": {
                    "id": assistant_message.id,
                    "role": assistant_message.role,
                    "content": assistant_message.content,
                    "sources": result.sources,
                    "status": assistant_message.status,
                    "created_at": assistant_message.created_at.isoformat(),
                },
                "sources": result.sources,
                "insufficient_evidence": result.insufficient_evidence,
                "retrieval": result.retrieval,
                "user_message": {
                    "id": user_message.id,
                    "role": user_message.role,
                    "content": user_message.content,
                    "created_at": user_message.created_at.isoformat(),
                },
            },
            status=status.HTTP_200_OK,
        )
