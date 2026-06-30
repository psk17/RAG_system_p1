"""
tests/unit/test_document_processor.py
───────────────────────────────────────
Tests for DocumentChunk value object and the BaseDocumentProcessor interface.
A MinimalProcessor stub lets us test the shared `chunk()` logic in isolation.
"""

import pytest
from pathlib import Path
from typing import BinaryIO
from unittest.mock import MagicMock

from rag_system.core.interfaces.document_processor import BaseDocumentProcessor, DocumentChunk


# ──────────────────────────────────────────────────────────────────────────────
# Stubs
# ──────────────────────────────────────────────────────────────────────────────

class MinimalProcessor(BaseDocumentProcessor):
    """Concrete stub — lets us control extract_text() output in tests."""

    def __init__(self, pages: list[tuple[str, int | None]] | None = None):
        super().__init__()
        self._pages = pages or []

    def load(self, source: Path | BinaryIO, *, file_name: str = "") -> None:
        self._raw_content = b"stub"
        self._file_name = file_name or "stub.txt"

    def extract_text(self) -> list[tuple[str, int | None]]:
        self._assert_loaded()
        return self._pages


# ──────────────────────────────────────────────────────────────────────────────
# DocumentChunk Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDocumentChunk:
    def test_create_generates_unique_ids(self):
        c1 = DocumentChunk.create(
            document_id="doc1", source_file="a.pdf",
            page_number=1, chunk_index=0, text="hello"
        )
        c2 = DocumentChunk.create(
            document_id="doc1", source_file="a.pdf",
            page_number=1, chunk_index=1, text="world"
        )
        assert c1.chunk_id != c2.chunk_id

    def test_to_metadata_dict_contains_required_keys(self):
        chunk = DocumentChunk.create(
            document_id="doc42", source_file="report.pdf",
            page_number=3, chunk_index=7, text="Some content"
        )
        meta = chunk.to_metadata_dict()
        for key in ("chunk_id", "document_id", "source_file", "page_number", "chunk_index"):
            assert key in meta

    def test_extra_metadata_forwarded(self):
        chunk = DocumentChunk.create(
            document_id="d1", source_file="f.pdf",
            page_number=None, chunk_index=0, text="x",
            extra_metadata={"department": "hr"}
        )
        assert chunk.to_metadata_dict()["department"] == "hr"

    def test_chunk_is_immutable(self):
        chunk = DocumentChunk.create(
            document_id="d1", source_file="f.pdf",
            page_number=1, chunk_index=0, text="immutable"
        )
        with pytest.raises(Exception):  # frozen dataclass raises FrozenInstanceError
            chunk.text = "mutated"  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────────
# BaseDocumentProcessor.chunk() Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBaseDocumentProcessorChunk:
    def _make_processor(self, pages):
        p = MinimalProcessor(pages=pages)
        p.load(MagicMock(), file_name="test.txt")
        return p

    def test_chunk_returns_document_chunks(self):
        p = self._make_processor([("Hello world " * 100, 1)])
        chunks = p.chunk(document_id="doc1", chunk_size=100, chunk_overlap=10)
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_chunk_indices_are_sequential(self):
        long_text = "word " * 500
        p = self._make_processor([(long_text, 1)])
        chunks = p.chunk(document_id="doc1", chunk_size=200, chunk_overlap=20)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_pages_produce_no_chunks(self):
        p = self._make_processor([("   ", 1), ("", 2)])
        chunks = p.chunk(document_id="doc1")
        assert chunks == []

    def test_source_file_propagated_to_chunks(self):
        p = MinimalProcessor(pages=[("Content here", 1)])
        p.load(MagicMock(), file_name="manual.pdf")
        chunks = p.chunk(document_id="docX")
        assert all(c.source_file == "manual.pdf" for c in chunks)

    def test_extract_text_raises_before_load(self):
        p = MinimalProcessor(pages=[("x", 1)])
        with pytest.raises(RuntimeError, match="load()"):
            p.extract_text()


# ──────────────────────────────────────────────────────────────────────────────
# BaseVectorStore Interface Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestBaseVectorStoreInterface:
    """Confirm the abstract interface cannot be instantiated directly."""

    def test_cannot_instantiate_abstract_class(self):
        from rag_system.core.interfaces.vector_store import BaseVectorStore
        with pytest.raises(TypeError):
            BaseVectorStore()  # type: ignore[abstract]

    def test_search_result_score_validation(self):
        from rag_system.core.interfaces.vector_store import SearchResult
        chunk = DocumentChunk.create(
            document_id="d", source_file="f.txt",
            page_number=None, chunk_index=0, text="t"
        )
        with pytest.raises(ValueError, match="score"):
            SearchResult(chunk=chunk, score=1.5)
