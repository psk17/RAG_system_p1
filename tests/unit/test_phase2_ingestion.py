"""
tests/unit/test_phase2_ingestion.py
──────────────────────────────────────
Phase 2 unit tests verifying:
  • PDFProcessor: multi-page extraction, metadata correctness, guard rails
  • MarkdownProcessor: header-aware chunking, section extraction
  • ChunkingService: processor dispatch, directory scan, document_id stability
  • ChromaAdapter (fake embeddings): upsert, search, delete, list_collections
  • IngestionService: end-to-end pipeline with in-memory Chroma

All tests are self-contained — no API keys, no disk persistence, no network.
PDF test files are generated in-memory using PyMuPDF.
"""

from __future__ import annotations

import asyncio
import io
import textwrap
from pathlib import Path

import fitz
import pytest

from rag_system.core.interfaces.document_processor import DocumentChunk
from rag_system.ingestion.chunking_service import ChunkingService
from rag_system.ingestion.markdown_processor import MarkdownProcessor
from rag_system.ingestion.pdf_processor import PDFProcessor
from rag_system.ingestion.vector_store_chroma import ChromaAdapter


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — in-memory fixture builders
# ──────────────────────────────────────────────────────────────────────────────

def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a minimal multi-page PDF in memory using PyMuPDF."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_md_file(tmp_path: Path, content: str, name: str = "test.md") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_pdf_file(tmp_path: Path, pages: list[str], name: str = "test.pdf") -> Path:
    p = tmp_path / name
    p.write_bytes(_make_pdf_bytes(pages))
    return p


# ──────────────────────────────────────────────────────────────────────────────
# PDFProcessor Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPDFProcessor:

    def test_load_from_path(self, tmp_path):
        pdf_path = _make_pdf_file(tmp_path, ["Page one text.", "Page two text."])
        p = PDFProcessor()
        p.load(pdf_path)
        assert p.page_count == 2

    def test_load_from_bytes_io(self):
        raw = _make_pdf_bytes(["Hello world"])
        stream = io.BytesIO(raw)
        p = PDFProcessor()
        p.load(stream, file_name="streamed.pdf")
        assert p._file_name == "streamed.pdf"
        assert p.page_count == 1

    def test_extract_text_returns_correct_page_count(self, tmp_path):
        pdf_path = _make_pdf_file(tmp_path, ["Page A", "Page B", "Page C"])
        p = PDFProcessor()
        p.load(pdf_path)
        pages = p.extract_text()
        assert len(pages) == 3

    def test_page_numbers_are_one_based(self, tmp_path):
        pdf_path = _make_pdf_file(tmp_path, ["First", "Second"])
        p = PDFProcessor()
        p.load(pdf_path)
        pages = p.extract_text()
        assert pages[0][1] == 1
        assert pages[1][1] == 2

    def test_chunk_carries_source_and_page_metadata(self, tmp_path):
        long_text = "word " * 300  # forces multiple chunks
        pdf_path = _make_pdf_file(tmp_path, [long_text, "Page 2 content"])
        p = PDFProcessor()
        p.load(pdf_path)
        chunks = p.chunk(document_id="doc_test", chunk_size=200, chunk_overlap=20)
        assert all(c.source_file == pdf_path.name for c in chunks)
        assert all(c.page_number is not None for c in chunks)

    def test_chunk_indices_are_sequential(self, tmp_path):
        pdf_path = _make_pdf_file(tmp_path, ["word " * 400])
        p = PDFProcessor()
        p.load(pdf_path)
        chunks = p.chunk(document_id="d1", chunk_size=200, chunk_overlap=20)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_file_not_found_raises(self):
        p = PDFProcessor()
        with pytest.raises(FileNotFoundError):
            p.load(Path("/nonexistent/file.pdf"))

    def test_invalid_pdf_bytes_raises(self):
        p = PDFProcessor()
        with pytest.raises(ValueError, match="valid PDF"):
            p.load(io.BytesIO(b"not a pdf"), file_name="bad.pdf")

    def test_oversized_file_raises(self, tmp_path):
        pdf_path = _make_pdf_file(tmp_path, ["text"])
        p = PDFProcessor(max_bytes=10)  # tiny limit
        with pytest.raises(ValueError, match="exceeds limit"):
            p.load(pdf_path)

    def test_extract_before_load_raises(self):
        p = PDFProcessor()
        with pytest.raises(RuntimeError, match="load()"):
            p.extract_text()


# ──────────────────────────────────────────────────────────────────────────────
# MarkdownProcessor Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMarkdownProcessor:

    SAMPLE_MD = textwrap.dedent("""\
        # Introduction

        This is the introduction paragraph with enough text to matter in chunking.

        ## Background

        The background section provides historical context for the document.

        ### Details

        Fine-grained details go here and include multiple sentences for splitting.
    """)

    def test_load_from_path(self, tmp_path):
        md_path = _make_md_file(tmp_path, self.SAMPLE_MD)
        p = MarkdownProcessor()
        p.load(md_path)
        assert p._file_name == "test.md"

    def test_extract_text_returns_single_page_none(self, tmp_path):
        md_path = _make_md_file(tmp_path, self.SAMPLE_MD)
        p = MarkdownProcessor()
        p.load(md_path)
        pages = p.extract_text()
        assert len(pages) == 1
        assert pages[0][1] is None  # no page number for markdown

    def test_header_aware_chunk_preserves_context(self, tmp_path):
        md_path = _make_md_file(tmp_path, self.SAMPLE_MD)
        p = MarkdownProcessor()
        p.load(md_path)
        chunks = p.chunk(document_id="doc_md", chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 0
        assert all(isinstance(c, DocumentChunk) for c in chunks)

    def test_sections_property(self, tmp_path):
        md_path = _make_md_file(tmp_path, self.SAMPLE_MD)
        p = MarkdownProcessor()
        p.load(md_path)
        sections = p.sections
        headings = [h for h, _ in sections]
        assert "Introduction" in headings
        assert "Background" in headings

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "doc.docx"
        bad.write_bytes(b"fake")
        p = MarkdownProcessor()
        with pytest.raises(ValueError, match="Unsupported extension"):
            p.load(bad)

    def test_txt_file_accepted(self, tmp_path):
        txt = tmp_path / "notes.txt"
        txt.write_text("Plain text notes here.\n" * 10)
        p = MarkdownProcessor()
        p.load(txt)
        pages = p.extract_text()
        assert len(pages) == 1


# ──────────────────────────────────────────────────────────────────────────────
# ChunkingService Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestChunkingService:

    def test_process_pdf_file(self, tmp_path):
        pdf_path = _make_pdf_file(
            tmp_path, ["word " * 200, "more words " * 150], name="report.pdf"
        )
        svc = ChunkingService(chunk_size=300, chunk_overlap=30)
        chunks = svc.process_file(pdf_path, collection_id="test_col")
        assert len(chunks) > 0
        assert all(c.extra_metadata.get("collection_id") == "test_col" for c in chunks)

    def test_process_markdown_file(self, tmp_path):
        md_path = _make_md_file(
            tmp_path, "# Title\n\n" + "Content sentence. " * 100
        )
        svc = ChunkingService(chunk_size=300, chunk_overlap=30)
        chunks = svc.process_file(md_path, collection_id="docs")
        assert len(chunks) > 0

    def test_document_id_is_deterministic(self, tmp_path):
        pdf_path = _make_pdf_file(tmp_path, ["stable"])
        id1 = ChunkingService._make_document_id(pdf_path)
        id2 = ChunkingService._make_document_id(pdf_path)
        assert id1 == id2

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "data.csv"
        bad.write_text("a,b,c")
        with pytest.raises(ValueError, match="No processor registered"):
            ChunkingService().process_file(bad)

    def test_process_directory_finds_all_supported_files(self, tmp_path):
        _make_pdf_file(tmp_path, ["PDF content " * 100], name="a.pdf")
        _make_md_file(tmp_path, "# MD\n\n" + "text " * 100, name="b.md")
        svc = ChunkingService(chunk_size=200, chunk_overlap=20)
        chunks = svc.process_directory(tmp_path, collection_id="batch")
        assert len(chunks) > 0

    def test_extra_tags_propagated(self, tmp_path):
        pdf_path = _make_pdf_file(tmp_path, ["content " * 100])
        svc = ChunkingService(chunk_size=200, chunk_overlap=20)
        chunks = svc.process_file(pdf_path, extra_tags={"department": "legal"})
        assert all(c.extra_metadata.get("department") == "legal" for c in chunks)

    def test_register_custom_processor(self, tmp_path):
        """Registering a new extension at runtime should work."""
        svc = ChunkingService()
        svc.register_processor(".txt2", lambda: MarkdownProcessor(strict_extension=False))
        f = tmp_path / "custom.txt2"
        f.write_text("custom " * 100)
        chunks = svc.process_file(f, collection_id="custom")
        assert len(chunks) > 0

    def test_metadata_contains_required_fields(self, tmp_path):
        """FR-ING-3: every chunk must have source, page_number, chunk_id."""
        pdf_path = _make_pdf_file(tmp_path, ["test " * 200])
        svc = ChunkingService(chunk_size=200, chunk_overlap=20)
        chunks = svc.process_file(pdf_path, collection_id="col")
        for c in chunks:
            meta = c.to_metadata_dict()
            assert "chunk_id" in meta
            assert "source_file" in meta
            assert "page_number" in meta


# ──────────────────────────────────────────────────────────────────────────────
# ChromaAdapter Tests (fake embeddings, in-memory)
# ──────────────────────────────────────────────────────────────────────────────

def _make_chunks(n: int = 5, doc_id: str = "doc_test") -> list[DocumentChunk]:
    return [
        DocumentChunk.create(
            document_id=doc_id,
            source_file="sample.pdf",
            page_number=i + 1,
            chunk_index=i,
            text=f"This is chunk number {i} with some searchable content about policies.",
            extra_metadata={"collection_id": "test_col"},
        )
        for i in range(n)
    ]


class TestChromaAdapter:

    def test_upsert_returns_correct_count(self):
        adapter = ChromaAdapter.for_testing()
        chunks = _make_chunks(5)
        count = asyncio.get_event_loop().run_until_complete(
            adapter.upsert(chunks, collection_name="test_col")
        )
        assert count == 5

    def test_upsert_empty_list_returns_zero(self):
        adapter = ChromaAdapter.for_testing()
        count = asyncio.get_event_loop().run_until_complete(
            adapter.upsert([], collection_name="test_col")
        )
        assert count == 0

    def test_similarity_search_returns_results(self):
        adapter = ChromaAdapter.for_testing()
        chunks = _make_chunks(10)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(adapter.upsert(chunks, collection_name="policies"))
        results = loop.run_until_complete(
            adapter.similarity_search(
                "searchable content policies",
                collection_name="policies",
                top_k=3,
            )
        )
        assert len(results) <= 3
        assert all(hasattr(r, "score") for r in results)

    def test_search_score_in_valid_range(self):
        adapter = ChromaAdapter.for_testing()
        chunks = _make_chunks(5)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(adapter.upsert(chunks, collection_name="score_test"))
        results = loop.run_until_complete(
            adapter.similarity_search("chunk policies", collection_name="score_test")
        )
        for r in results:
            assert 0.0 <= r.score <= 1.0, f"Score out of range: {r.score}"

    def test_search_result_preserves_metadata(self):
        adapter = ChromaAdapter.for_testing()
        chunks = _make_chunks(3)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(adapter.upsert(chunks, collection_name="meta_test"))
        results = loop.run_until_complete(
            adapter.similarity_search("chunk content", collection_name="meta_test")
        )
        for r in results:
            assert r.chunk.source_file == "sample.pdf"
            assert r.chunk.page_number is not None

    def test_upsert_is_idempotent(self):
        adapter = ChromaAdapter.for_testing()
        chunks = _make_chunks(3)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(adapter.upsert(chunks, collection_name="idem_test"))
        # Same chunk_ids → should update not duplicate
        loop.run_until_complete(adapter.upsert(chunks, collection_name="idem_test"))
        results = loop.run_until_complete(
            adapter.similarity_search("chunk", collection_name="idem_test", top_k=10)
        )
        assert len(results) <= 3

    def test_delete_document(self):
        adapter = ChromaAdapter.for_testing()
        chunks = _make_chunks(4, doc_id="doc_to_delete")
        loop = asyncio.get_event_loop()
        loop.run_until_complete(adapter.upsert(chunks, collection_name="del_test"))
        deleted = loop.run_until_complete(
            adapter.delete_document("doc_to_delete", collection_name="del_test")
        )
        assert deleted == 4

    def test_list_collections(self):
        adapter = ChromaAdapter.for_testing()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            adapter.upsert(_make_chunks(2), collection_name="col_alpha")
        )
        loop.run_until_complete(
            adapter.upsert(_make_chunks(2), collection_name="col_beta")
        )
        cols = loop.run_until_complete(adapter.list_collections())
        assert "col_alpha" in cols
        assert "col_beta" in cols

    def test_collection_exists(self):
        adapter = ChromaAdapter.for_testing()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            adapter.upsert(_make_chunks(1), collection_name="exists_col")
        )
        assert loop.run_until_complete(adapter.collection_exists("exists_col"))
        assert not loop.run_until_complete(adapter.collection_exists("ghost_col"))


# ──────────────────────────────────────────────────────────────────────────────
# IngestionService End-to-End Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestIngestionServiceE2E:

    def test_ingest_pdf_file_end_to_end(self, tmp_path):
        """Full pipeline: PDF → chunks → Chroma (fake embeddings)."""
        pdf_path = _make_pdf_file(
            tmp_path,
            ["word " * 200, "more content " * 150, "final page " * 100],
            name="handbook.pdf",
        )
        adapter = ChromaAdapter.for_testing()
        from rag_system.ingestion.ingestion_service import IngestionService
        svc = IngestionService(
            vector_store=adapter,
            chunking_service=ChunkingService(chunk_size=300, chunk_overlap=30),
        )
        result = asyncio.get_event_loop().run_until_complete(
            svc.ingest_file(pdf_path, collection_id="e2e_test")
        )
        assert result.success
        assert result.chunks_upserted > 0
        assert result.chunks_processed == result.chunks_upserted

    def test_ingest_directory_end_to_end(self, tmp_path):
        _make_pdf_file(tmp_path, ["PDF data " * 100], name="file1.pdf")
        _make_md_file(tmp_path, "# Title\n\n" + "text " * 100, name="file2.md")

        adapter = ChromaAdapter.for_testing()
        from rag_system.ingestion.ingestion_service import IngestionService
        svc = IngestionService(
            vector_store=adapter,
            chunking_service=ChunkingService(chunk_size=200, chunk_overlap=20),
        )
        results = asyncio.get_event_loop().run_until_complete(
            svc.ingest_directory(tmp_path, collection_id="dir_test")
        )
        assert len(results) == 2
        assert all(r.success for r in results)
        total_upserted = sum(r.chunks_upserted for r in results)
        assert total_upserted > 0

    def test_ingest_result_has_correct_source_name(self, tmp_path):
        pdf_path = _make_pdf_file(tmp_path, ["content " * 50], name="specific.pdf")
        adapter = ChromaAdapter.for_testing()
        from rag_system.ingestion.ingestion_service import IngestionService
        svc = IngestionService(vector_store=adapter)
        result = asyncio.get_event_loop().run_until_complete(
            svc.ingest_file(pdf_path, collection_id="name_test")
        )
        assert result.source == "specific.pdf"
