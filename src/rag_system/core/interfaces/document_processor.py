"""
app/core/interfaces/document_processor.py
──────────────────────────────────────────
Abstract contract that every concrete document processor must fulfill.

Concrete implementations (Phase 2):
  - PDFProcessor   (PyMuPDF)
  - TxtProcessor   (plain text)
  - DocxProcessor  (python-docx)
  - MarkdownProcessor

Design Notes
────────────
• `load()` is responsible only for I/O — reading raw bytes from disk / memory.
• `extract_text()` does the format-specific parsing; it must be idempotent.
• `chunk()` calls LangChain's RecursiveCharacterTextSplitter; the default
  parameters are pulled from settings so they stay in one place.
• `DocumentChunk` is a pure data container (dataclass) — no business logic.
  It carries all metadata required by FR-ING-3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
import uuid


# ──────────────────────────────────────────────────────────────────────────────
# Value Objects
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DocumentChunk:
    """
    Immutable container for a single text chunk and its provenance metadata.

    Attributes
    ──────────
    chunk_id:       Globally unique identifier for this chunk (UUID4).
    document_id:    ID of the parent document (set by the ingestion service).
    source_file:    Original filename as uploaded.
    page_number:    1-based page index; None for formats without pages (TXT, MD).
    chunk_index:    0-based position of this chunk within the document.
    text:           The raw text content of the chunk.
    extra_metadata: Arbitrary key/value pairs for downstream filtering
                    (e.g. collection_id, department, language).
    """

    chunk_id: str
    document_id: str
    source_file: str
    page_number: int | None
    chunk_index: int
    text: str
    extra_metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        source_file: str,
        page_number: int | None,
        chunk_index: int,
        text: str,
        extra_metadata: dict[str, str] | None = None,
    ) -> "DocumentChunk":
        """Factory method that auto-generates a unique chunk_id."""
        return cls(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
            text=text,
            extra_metadata=extra_metadata or {},
        )

    def to_metadata_dict(self) -> dict[str, str | int | None]:
        """Flat dict suitable for storing in a vector database metadata field."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            **self.extra_metadata,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Base Class
# ──────────────────────────────────────────────────────────────────────────────

class BaseDocumentProcessor(ABC):
    """
    Interface that every file-format processor must implement.

    Subclasses MUST override:
        load()          — read raw content from a path or file-like object
        extract_text()  — parse the raw content into (text, page_number) pairs

    Subclasses MAY override:
        chunk()         — default uses RecursiveCharacterTextSplitter;
                          override if the format has natural split points
                          (e.g. Markdown headers, DOCX paragraph styles).

    Typical usage (by the IngestionOrchestrator):
        processor = PDFProcessor()
        processor.load(file_path)
        chunks = processor.chunk(document_id="doc_001", source_file="report.pdf")
    """

    def __init__(self) -> None:
        self._raw_content: bytes | None = None
        self._file_name: str = ""

    # ── Abstract Methods ──────────────────────────────────────────────────────

    @abstractmethod
    def load(self, source: Path | BinaryIO, *, file_name: str = "") -> None:
        """
        Read raw bytes from `source` into internal state.

        Args:
            source:     A filesystem path or any readable binary stream.
            file_name:  Original filename (used in chunk metadata). When
                        `source` is a Path, the stem is used if not provided.

        Raises:
            FileNotFoundError:  If `source` is a Path that does not exist.
            ValueError:         If `source` is neither a Path nor a BinaryIO.
        """
        ...

    @abstractmethod
    def extract_text(self) -> list[tuple[str, int | None]]:
        """
        Parse the loaded content into a list of (text, page_number) tuples.

        Must be called after `load()`.

        Returns:
            A list where each element is:
                (text_block: str, page_number: int | None)
            Page numbers are 1-based. Non-paginated formats return None.

        Raises:
            RuntimeError: If called before `load()`.
        """
        ...

    # ── Concrete Methods ──────────────────────────────────────────────────────

    def chunk(
        self,
        *,
        document_id: str,
        source_file: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extra_metadata: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        """
        Split extracted text into overlapping chunks with metadata.

        Pulls chunk_size / chunk_overlap from settings by default;
        callers can override per-call for testing or special collections.

        Args:
            document_id:    Identifier for the parent document.
            source_file:    Filename stored in metadata; defaults to the name
                            captured during `load()`.
            chunk_size:     Max characters per chunk.
            chunk_overlap:  Character overlap between consecutive chunks.
            extra_metadata: Forwarded verbatim to every DocumentChunk.

        Returns:
            Ordered list of DocumentChunk objects ready for embedding.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        pages = self.extract_text()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        fname = source_file or self._file_name
        chunks: list[DocumentChunk] = []
        global_index = 0

        for text, page_num in pages:
            if not text.strip():
                continue
            splits = splitter.split_text(text)
            for split in splits:
                chunks.append(
                    DocumentChunk.create(
                        document_id=document_id,
                        source_file=fname,
                        page_number=page_num,
                        chunk_index=global_index,
                        text=split,
                        extra_metadata=extra_metadata,
                    )
                )
                global_index += 1

        return chunks

    def _assert_loaded(self) -> None:
        """Guard: raise if extract_text() is called before load()."""
        if self._raw_content is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.load() must be called before extract_text()."
            )
