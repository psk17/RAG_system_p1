"""
app/ingestion/pdf_processor.py
────────────────────────────────
Concrete DocumentProcessor for PDF files using PyMuPDF (fitz).

Features
────────
• Extracts text page-by-page with 1-based page numbers (FR-ING-3).
• Preserves reading order via PyMuPDF's `get_text("blocks")` layout sort.
• Falls back to raw `get_text()` if block extraction yields nothing
  (scanned PDFs, image-only pages).
• Accepts both filesystem paths and in-memory BytesIO streams (for
  FastAPI UploadFile objects — no temp files on disk; NFR-SEC-1).
• File-size guard prevents loading files that exceed MAX_UPLOAD_MB.

Usage
─────
    processor = PDFProcessor(max_bytes=settings.max_upload_bytes)
    processor.load(Path("report.pdf"))
    chunks = processor.chunk(document_id="doc_001")
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import fitz  # PyMuPDF

from rag_system.core.interfaces.document_processor import BaseDocumentProcessor


class PDFProcessor(BaseDocumentProcessor):
    """PyMuPDF-backed PDF parser."""

    # ── Constructor ───────────────────────────────────────────────────────────

    def __init__(self, max_bytes: int = 50 * 1024 * 1024) -> None:
        """
        Args:
            max_bytes: Hard limit on acceptable file size. Defaults to 50 MB
                       (FR-ING-1). Pass `settings.max_upload_bytes` in prod.
        """
        super().__init__()
        self._max_bytes = max_bytes
        self._doc: fitz.Document | None = None

    # ── Abstract Method Implementations ──────────────────────────────────────

    def load(self, source: Path | BinaryIO, *, file_name: str = "") -> None:
        """
        Open a PDF from a path or a file-like binary stream.

        Args:
            source:    Path to a .pdf file OR any readable BinaryIO
                       (e.g. FastAPI UploadFile.file, BytesIO).
            file_name: Stored in chunk metadata. Auto-derived from path stem
                       when source is a Path and file_name is not given.

        Raises:
            FileNotFoundError: Path does not exist.
            ValueError:        Source type is unsupported, file is not a PDF,
                               or file exceeds max_bytes.
        """
        if isinstance(source, Path):
            if not source.exists():
                raise FileNotFoundError(f"PDF not found: {source}")
            if source.stat().st_size > self._max_bytes:
                raise ValueError(
                    f"File {source.name} exceeds limit "
                    f"({source.stat().st_size} > {self._max_bytes} bytes)"
                )
            self._raw_content = source.read_bytes()
            self._file_name = file_name or source.name

        elif hasattr(source, "read"):
            data = source.read()
            if len(data) > self._max_bytes:
                raise ValueError(
                    f"Stream exceeds size limit ({len(data)} > {self._max_bytes} bytes)"
                )
            self._raw_content = data
            self._file_name = file_name or "upload.pdf"

        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        # Validate it's actually a PDF before storing
        if not self._raw_content.startswith(b"%PDF"):
            raise ValueError(f"'{self._file_name}' does not appear to be a valid PDF.")

        self._doc = fitz.open(stream=self._raw_content, filetype="pdf")

    def extract_text(self) -> list[tuple[str, int | None]]:
        """
        Extract text from every page, preserving reading order.

        Returns:
            List of (text, 1-based page number) tuples.
            Empty / image-only pages are included as ("", page_num) and
            filtered out later by BaseDocumentProcessor.chunk().

        Raises:
            RuntimeError: If called before load().
        """
        self._assert_loaded()
        assert self._doc is not None

        pages: list[tuple[str, int | None]] = []

        for page_index in range(len(self._doc)):
            page = self._doc[page_index]
            page_num = page_index + 1  # 1-based

            # Attempt layout-aware block extraction first
            blocks = page.get_text("blocks", sort=True)  # sorted top→bottom, left→right
            block_texts = [b[4].strip() for b in blocks if b[4].strip()]
            text = "\n".join(block_texts)

            # Fallback: raw text (handles simple single-column PDFs)
            if not text:
                text = page.get_text().strip()

            pages.append((text, page_num))

        return pages

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def page_count(self) -> int:
        """Number of pages in the loaded document; 0 if not yet loaded."""
        return len(self._doc) if self._doc else 0
