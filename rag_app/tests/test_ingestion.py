"""Tests for the ingestion pipeline (incremental indexing)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase, override_settings

from rag_app.models import (
    IngestionJob,
    IngestionStatus,
    IndexedFile,
    IndexedFileStatus,
    KnowledgeBase,
    KnowledgeBaseStatus,
)
from rag_app.services.ingestion_service import run_ingestion
from rag_app.services.vector_store import get_vector_store, similarity_search

from .fakes import FakeEmbeddings


class IngestionTests(TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.source = Path(self._tmp.name).resolve() / "source"
        self.source.mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)

        self.persist = Path(self._tmp.name).resolve() / "chroma"
        self.settings = override_settings(
            CHROMA_PERSIST_DIRECTORY=str(self.persist),
            ALLOWED_SOURCE_ROOTS=[self.source],
        )
        self.settings.enable()
        self.addCleanup(self.settings.disable)

        self.knowledge_base = KnowledgeBase.objects.create(
            name="Test KB",
            source_type="local_directory",
            source_location=str(self.source),
            collection_name="kb_test_ingestion",
        )

        self.embeddings = FakeEmbeddings()
        self.embed_patch = patch(
            "rag_app.services.vector_store.get_embedding_model", return_value=self.embeddings
        )
        self.embed_patch.start()
        self.addCleanup(self.embed_patch.stop)

    def _job(self) -> IngestionJob:
        return IngestionJob.objects.create(knowledge_base=self.knowledge_base)

    def _ingest(self) -> IngestionJob:
        job = self._job()
        return run_ingestion(job.id)

    def _store(self):
        return get_vector_store(self.knowledge_base)

    def test_indexes_files_and_creates_chunks(self):
        (self.source / "alpha.md").write_text(
            "The capital of France is Paris. It is a beautiful city.", encoding="utf-8"
        )
        (self.source / "beta.txt").write_text("Python is a programming language.", encoding="utf-8")

        job = self._ingest()

        self.assertEqual(job.status, IngestionStatus.COMPLETED)
        self.assertEqual(job.files_discovered, 2)
        self.assertEqual(job.files_processed, 2)
        self.assertGreater(job.chunks_created, 0)

        self.knowledge_base.refresh_from_db()
        self.assertEqual(self.knowledge_base.status, KnowledgeBaseStatus.READY)
        self.assertIsNotNone(self.knowledge_base.last_indexed_at)

        self.assertEqual(IndexedFile.objects.filter(knowledge_base=self.knowledge_base).count(), 2)
        record = IndexedFile.objects.get(knowledge_base=self.knowledge_base, relative_path="alpha.md")
        self.assertEqual(record.status, IndexedFileStatus.INDEXED)
        self.assertGreater(record.chunk_count, 0)
        self.assertTrue(record.content_hash)

    def test_second_run_is_idempotent(self):
        (self.source / "alpha.md").write_text("The capital of France is Paris.", encoding="utf-8")
        self._ingest()
        first = IndexedFile.objects.get(relative_path="alpha.md")

        job = self._ingest()

        self.assertEqual(job.status, IngestionStatus.COMPLETED)
        self.assertEqual(job.files_processed, 0)
        self.assertEqual(job.files_skipped, 1)
        self.assertEqual(job.chunks_created, 0)

        first.refresh_from_db()
        self.assertEqual(first.chunk_count, first.chunk_count)
        hits = similarity_search(self._store(), "capital of France", k=10)
        self.assertEqual(len(hits), 1)

    def test_changed_file_replaces_its_chunks(self):
        target = self.source / "alpha.md"
        target.write_text("The capital of France is Paris.", encoding="utf-8")
        self._ingest()

        target.write_text("The capital of Japan is Tokyo.", encoding="utf-8")
        job = self._ingest()

        self.assertEqual(job.files_processed, 1)
        record = IndexedFile.objects.get(relative_path="alpha.md")
        self.assertNotEqual(record.content_hash, "")
        hits = similarity_search(self._store(), "capital", k=10)
        self.assertEqual(len(hits), 1)
        self.assertIn("Tokyo", hits[0].document.page_content)

    def test_removed_file_loses_its_chunks(self):
        target = self.source / "alpha.md"
        target.write_text("The capital of France is Paris.", encoding="utf-8")
        self._ingest()

        target.unlink()
        job = self._ingest()

        self.assertEqual(job.status, IngestionStatus.COMPLETED)
        self.assertFalse(
            IndexedFile.objects.filter(
                knowledge_base=self.knowledge_base, relative_path="alpha.md"
            ).exists()
        )
        hits = similarity_search(self._store(), "capital of France", k=10)
        self.assertEqual(len(hits), 0)

    def test_a_broken_file_is_recorded_but_does_not_fail_the_job(self):
        (self.source / "good.md").write_text("Perfectly fine content here.", encoding="utf-8")
        (self.source / "broken.pdf").write_bytes(b"%PDF-1.4 not a real pdf")

        job = self._ingest()

        self.assertEqual(job.status, IngestionStatus.COMPLETED)
        self.assertEqual(job.files_discovered, 2)
        self.assertEqual(job.files_processed, 1)
        self.assertEqual(job.files_failed, 1)
        self.assertTrue(job.error_summary)

        failed = IndexedFile.objects.get(relative_path="broken.pdf")
        self.assertEqual(failed.status, IndexedFileStatus.FAILED)
        self.assertTrue(failed.error_message)

    def test_an_invalid_source_fails_the_job(self):
        self.knowledge_base.source_location = str(self.source.parent / "missing")
        self.knowledge_base.save()

        job = self._ingest()

        self.assertEqual(job.status, IngestionStatus.FAILED)
        self.assertTrue(job.error_summary)
        self.knowledge_base.refresh_from_db()
        self.assertEqual(self.knowledge_base.status, KnowledgeBaseStatus.FAILED)

    def test_the_job_records_the_stage_progress(self):
        (self.source / "alpha.md").write_text("Some content for the chunker.", encoding="utf-8")
        job = self._ingest()
        self.assertEqual(job.stage, "completed")
        self.assertIsNotNone(job.started_at)
        self.assertIsNotNone(job.finished_at)
