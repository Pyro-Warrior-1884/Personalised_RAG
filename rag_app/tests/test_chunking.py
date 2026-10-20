"""Tests for chunking, chunk ids and the vector store."""

from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import SimpleTestCase, override_settings
from langchain_core.documents import Document

from rag_app.services.chunking import (
    assign_chunk_ids,
    make_chunk_id,
    split_documents,
)
from rag_app.services.vector_store import (
    delete_file_chunks,
    get_vector_store,
    similarity_search,
    upsert_chunks,
)

from .fakes import FakeEmbeddings


class ChunkingTests(SimpleTestCase):
    def test_split_preserves_metadata_and_bounds_the_size(self):
        document = Document(
            page_content="word " * 800,
            metadata={"relative_path": "a.txt", "file_hash": "abc"},
        )
        chunks = split_documents([document], chunk_size=200, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["relative_path"], "a.txt")
            self.assertLessEqual(len(chunk.page_content), 220)

    def test_chunk_ids_are_deterministic(self):
        first = make_chunk_id("kb-1", "a.txt", "hash", 0)
        second = make_chunk_id("kb-1", "a.txt", "hash", 0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, make_chunk_id("kb-1", "a.txt", "hash", 1))
        self.assertNotEqual(first, make_chunk_id("kb-2", "a.txt", "hash", 0))

    def test_assign_chunk_ids_numbers_chunks_per_file(self):
        documents = [
            Document(page_content="a", metadata={"relative_path": "a.txt", "file_hash": "h"}),
            Document(page_content="b", metadata={"relative_path": "a.txt", "file_hash": "h"}),
            Document(page_content="c", metadata={"relative_path": "b.txt", "file_hash": "h"}),
        ]
        chunks = assign_chunk_ids(split_documents(documents, chunk_size=2000, chunk_overlap=0), "kb-1")
        by_file = {}
        for chunk in chunks:
            by_file.setdefault(chunk.metadata["relative_path"], []).append(chunk)
        self.assertEqual([c.metadata["chunk_index"] for c in by_file["a.txt"]], [0, 1])
        self.assertEqual([c.metadata["chunk_index"] for c in by_file["b.txt"]], [0])
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        self.assertEqual(len(ids), len(set(ids)))


class VectorStoreTests(SimpleTestCase):
    """The Chroma integration is exercised with a fake embedding model."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.persist = Path(self._tmp.name) / "chroma"
        self.settings = override_settings(CHROMA_PERSIST_DIRECTORY=str(self.persist))
        self.settings.enable()
        self.addCleanup(self.settings.disable)
        self.embeddings = FakeEmbeddings()

    def _store(self, name: str = "kb_test"):
        with override_settings():
            from unittest.mock import patch

            with patch(
                "rag_app.services.vector_store.get_embedding_model", return_value=self.embeddings
            ):
                return get_vector_store(_FakeKnowledgeBase(name))

    def test_upsert_and_search_returns_the_matching_chunk(self):
        store = self._store()
        chunks = [
            Document(
                page_content="The Eiffel tower is in Paris France",
                metadata={"relative_path": "paris.md", "chunk_id": "c1", "chunk_index": 0},
            ),
            Document(
                page_content="Penguins live in the Antarctic",
                metadata={"relative_path": "animals.md", "chunk_id": "c2", "chunk_index": 0},
            ),
        ]
        self.assertEqual(upsert_chunks(store, chunks), 2)

        hits = similarity_search(store, "Where is the Eiffel tower?", k=2)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].document.metadata["relative_path"], "paris.md")
        self.assertGreater(hits[0].score, hits[1].score)
        # With cosine space the score is a cosine similarity in [-1, 1].
        self.assertLessEqual(hits[0].score, 1.0)
        self.assertGreaterEqual(hits[0].score, -1.0)

    def test_delete_file_chunks_removes_only_that_file(self):
        store = self._store()
        upsert_chunks(
            store,
            [
                Document(page_content="alpha text", metadata={"relative_path": "a.md", "chunk_id": "a1"}),
                Document(page_content="beta text", metadata={"relative_path": "b.md", "chunk_id": "b1"}),
            ],
        )
        self.assertEqual(delete_file_chunks(store, "a.md"), 1)
        hits = similarity_search(store, "alpha", k=5)
        paths = {hit.document.metadata.get("relative_path") for hit in hits}
        self.assertNotIn("a.md", paths)
        self.assertIn("b.md", paths)

    def test_upsert_twice_does_not_duplicate(self):
        store = self._store()
        chunks = [
            Document(page_content="alpha text", metadata={"relative_path": "a.md", "chunk_id": "a1"})
        ]
        upsert_chunks(store, chunks)
        upsert_chunks(store, chunks)
        hits = similarity_search(store, "alpha", k=10)
        self.assertEqual(len(hits), 1)


class _FakeKnowledgeBase:
    """Minimal stand-in for the model used by the vector store."""

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.id = "fake-kb"
