"""Tests for the REST API."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from rag_app.models import (
    Conversation,
    IngestionJob,
    IndexedFile,
    KnowledgeBase,
    KnowledgeBaseStatus,
    Message,
    MessageStatus,
)


class KnowledgeBaseApiTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.source = Path(self._tmp.name).resolve() / "source"
        self.source.mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)
        self.settings = override_settings(ALLOWED_SOURCE_ROOTS=[self.source])
        self.settings.enable()
        self.addCleanup(self.settings.disable)

    def _payload(self, **overrides):
        data = {
            "name": "My documents",
            "source_type": "local_directory",
            "source": str(self.source),
        }
        data.update(overrides)
        return data

    def test_creating_a_knowledge_base_queues_an_ingestion_job(self):
        with patch("rag_app.api.knowledge_base_views.enqueue_ingestion") as enqueue:
            response = self.client.post(
                reverse("rag_app:knowledge-base-list"),
                data=json.dumps(self._payload()),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 202)
        self.assertIn("knowledge_base", response.json())
        self.assertIn("job", response.json())
        self.assertTrue(enqueue.called)
        self.assertEqual(KnowledgeBase.objects.count(), 1)
        self.assertEqual(IngestionJob.objects.count(), 1)

    def test_creating_a_knowledge_base_rejects_a_path_outside_the_roots(self):
        response = self.client.post(
            reverse("rag_app:knowledge-base-list"),
            data=json.dumps(self._payload(source="/etc")),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("approved roots", str(response.json()["detail"]))
        self.assertEqual(KnowledgeBase.objects.count(), 0)

    def test_creating_a_knowledge_base_rejects_a_missing_path(self):
        response = self.client.post(
            reverse("rag_app:knowledge-base-list"),
            data=json.dumps(self._payload(source=str(self.source / "nope"))),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_creating_a_knowledge_base_rejects_a_bad_github_url(self):
        response = self.client.post(
            reverse("rag_app:knowledge-base-list"),
            data=json.dumps(self._payload(source_type="github", source="https://evil.example/x/y")),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_listing_knowledge_bases(self):
        KnowledgeBase.objects.create(
            name="KB",
            source_type="local_directory",
            source_location=str(self.source),
            collection_name="kb_list_test",
            status=KnowledgeBaseStatus.READY,
        )
        response = self.client.get(reverse("rag_app:knowledge-base-list"))
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "KB")

    def test_knowledge_base_detail_and_status(self):
        kb = KnowledgeBase.objects.create(
            name="KB",
            source_type="local_directory",
            source_location=str(self.source),
            collection_name="kb_detail_test",
        )
        job = IngestionJob.objects.create(knowledge_base=kb, stage="parsing", files_discovered=4)

        detail = self.client.get(reverse("rag_app:knowledge-base-detail", args=[kb.id]))
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["latest_job"]["id"], str(job.id))

        status = self.client.get(reverse("rag_app:knowledge-base-status", args=[kb.id]))
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["stage"], "parsing")
        self.assertEqual(status.json()["files_discovered"], 4)

    def test_status_of_an_unknown_knowledge_base_is_404(self):
        response = self.client.get(
            reverse("rag_app:knowledge-base-status", args=["00000000-0000-0000-0000-000000000000"])
        )
        self.assertEqual(response.status_code, 404)

    def test_listing_indexed_files(self):
        kb = KnowledgeBase.objects.create(
            name="KB",
            source_type="local_directory",
            source_location=str(self.source),
            collection_name="kb_files_test",
        )
        IndexedFile.objects.create(
            knowledge_base=kb, relative_path="a.md", status="indexed", chunk_count=2
        )
        IndexedFile.objects.create(
            knowledge_base=kb, relative_path="b.pdf", status="failed", error_message="broken"
        )
        response = self.client.get(reverse("rag_app:knowledge-base-files", args=[kb.id]))
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["counts"]["indexed"], 1)
        self.assertEqual(payload["counts"]["failed"], 1)

    def test_deleting_a_knowledge_base_requires_confirmation(self):
        kb = KnowledgeBase.objects.create(
            name="KB",
            source_type="local_directory",
            source_location=str(self.source),
            collection_name="kb_delete_test",
        )
        response = self.client.delete(
            reverse("rag_app:knowledge-base-detail", args=[kb.id]),
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertTrue(KnowledgeBase.objects.filter(pk=kb.id).exists())

    def test_deleting_a_knowledge_base_removes_the_records(self):
        kb = KnowledgeBase.objects.create(
            name="KB",
            source_type="local_directory",
            source_location=str(self.source),
            collection_name="kb_delete_test_2",
        )
        with patch("rag_app.api.knowledge_base_views.delete_knowledge_base_collection") as delete:
            response = self.client.delete(
                reverse("rag_app:knowledge-base-detail", args=[kb.id]),
                data=json.dumps({"confirm": True}),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 204)
        self.assertTrue(delete.called)
        self.assertFalse(KnowledgeBase.objects.filter(pk=kb.id).exists())

    def test_reindex_queues_a_new_job(self):
        kb = KnowledgeBase.objects.create(
            name="KB",
            source_type="local_directory",
            source_location=str(self.source),
            collection_name="kb_reindex_test",
        )
        with patch("rag_app.api.knowledge_base_views.enqueue_ingestion") as enqueue:
            response = self.client.post(reverse("rag_app:knowledge-base-reindex", args=[kb.id]))
        self.assertEqual(response.status_code, 202)
        self.assertTrue(enqueue.called)
        self.assertEqual(IngestionJob.objects.filter(knowledge_base=kb).count(), 1)


class ConversationApiTests(TestCase):
    def setUp(self):
        self.knowledge_base = KnowledgeBase.objects.create(
            name="KB",
            source_type="local_directory",
            source_location="/tmp",
            collection_name="kb_conversation_test",
            status=KnowledgeBaseStatus.READY,
        )

    def test_creating_and_listing_conversations(self):
        response = self.client.post(
            reverse("rag_app:conversation-list"),
            data=json.dumps({"knowledge_base_id": str(self.knowledge_base.id)}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 201)
        conversation_id = response.json()["id"]
        self.assertEqual(Conversation.objects.count(), 1)

        listing = self.client.get(reverse("rag_app:conversation-list"))
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(len(listing.json()["results"]), 1)

        detail = self.client.get(reverse("rag_app:conversation-detail", args=[conversation_id]))
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["knowledge_base_id"], str(self.knowledge_base.id))

    def test_renaming_a_conversation(self):
        conversation = Conversation.objects.create(title="Old title")
        response = self.client.patch(
            reverse("rag_app:conversation-detail", args=[conversation.id]),
            data=json.dumps({"title": "New title"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        conversation.refresh_from_db()
        self.assertEqual(conversation.title, "New title")

    def test_renaming_to_an_empty_title_is_rejected(self):
        conversation = Conversation.objects.create(title="Old title")
        response = self.client.patch(
            reverse("rag_app:conversation-detail", args=[conversation.id]),
            data=json.dumps({"title": "   "}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_deleting_a_conversation_removes_its_messages(self):
        conversation = Conversation.objects.create(title="Chat")
        Message.objects.create(conversation=conversation, role="user", content="hi")
        response = self.client.delete(reverse("rag_app:conversation-detail", args=[conversation.id]))
        self.assertEqual(response.status_code, 204)
        self.assertFalse(Conversation.objects.filter(pk=conversation.id).exists())
        self.assertEqual(Message.objects.count(), 0)

    def test_sending_a_message_requires_a_knowledge_base(self):
        conversation = Conversation.objects.create(title="Chat")
        response = self.client.post(
            reverse("rag_app:conversation-messages", args=[conversation.id]),
            data=json.dumps({"content": "Hello?"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 409)

    def test_sending_an_empty_message_is_rejected(self):
        conversation = Conversation.objects.create(
            title="Chat", knowledge_base=self.knowledge_base
        )
        response = self.client.post(
            reverse("rag_app:conversation-messages", args=[conversation.id]),
            data=json.dumps({"content": "   "}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_sending_a_message_stores_the_question_and_the_answer(self):
        conversation = Conversation.objects.create(
            title="New chat", knowledge_base=self.knowledge_base
        )
        with patch("rag_app.api.chat_views.answer_question") as answer:
            answer.return_value = type(
                "Result",
                (),
                {
                    "answer": "Paris is the capital of France.",
                    "sources": [
                        {
                            "label": "paris.md",
                            "relative_path": "paris.md",
                            "score": 0.8,
                            "excerpt": "Paris is the capital of France.",
                        }
                    ],
                    "insufficient_evidence": False,
                    "retrieval": {"k": 5, "returned": 1},
                },
            )()
            response = self.client.post(
                reverse("rag_app:conversation-messages", args=[conversation.id]),
                data=json.dumps({"content": "What is the capital of France?"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["message"]["content"], "Paris is the capital of France.")
        self.assertEqual(len(payload["sources"]), 1)

        conversation.refresh_from_db()
        self.assertEqual(conversation.title, "What is the capital of France?")
        self.assertEqual(conversation.messages.count(), 2)
        self.assertEqual(conversation.messages.filter(role="user").count(), 1)

    def test_a_groq_failure_keeps_the_question_and_stores_no_fake_answer(self):
        conversation = Conversation.objects.create(
            title="New chat", knowledge_base=self.knowledge_base
        )
        with patch("rag_app.api.chat_views.answer_question") as answer:
            answer.side_effect = __import__(
                "rag_app.services.llm_service", fromlist=["LLMError"]
            ).LLMError("Groq rate limit reached.")
            response = self.client.post(
                reverse("rag_app:conversation-messages", args=[conversation.id]),
                data=json.dumps({"content": "Anything?"}),
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("rate limit", response.json()["detail"])

        messages = list(conversation.messages.order_by("created_at"))
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[0].content, "Anything?")
        self.assertEqual(messages[1].role, "assistant")
        self.assertEqual(messages[1].content, "")
        self.assertEqual(messages[1].status, MessageStatus.ERROR)
        self.assertIn("rate limit", messages[1].error_message)


class HomePageTests(TestCase):
    def test_the_home_page_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Personalised RAG")
        self.assertContains(response, "Welcome to Personalised RAG")

    def test_home_page_shows_chat_state_when_a_kb_is_ready(self):
        KnowledgeBase.objects.create(
            name="Ready KB",
            source_type="local_directory",
            source_location="/tmp",
            collection_name="kb_home_ready",
            status="ready",
        )
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "How can I help you today?")
