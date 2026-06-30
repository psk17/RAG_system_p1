"""
app/ingestion/markdown_processor.py
─────────────────────────────────────
Concrete DocumentProcessor for plain-text and Markdown files.

Features
────────
• Header-aware splitting: the ChunkingService can detect Markdown headings
  (# / ## / ###) as natural semantic boundaries so context stays intact.
• Exposes `sections` property — list of (heading, body) pairs — useful
  for downstream metadata enrichment or structured retrieval.
• Handles both .md and .txt files; page_number is always None (no concept
  of pages) as per the interface contract.
• Accepts Path or BinaryIO with configurable encoding (default UTF-8).

Usage
─────
    processor = MarkdownProcessor()
    processor.load(Path("docs/overview.md"))
    chunks = processor.chunk(document_id="doc_002")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO

from rag_system.core.interfaces.document_processor import BaseDocumentProcessor


# Heading pattern: lines starting with 1–6 `#` characters
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class MarkdownProcessor(BaseDocumentProcessor):
    """Header-aware Markdown / plain-text parser."""

    SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".text"}

    def __init__(
        self,
        encoding: str = "utf-8",
        max_bytes: int = 50 * 1024 * 1024,
        strict_extension: bool = True,
    ) -> None:
        """
        Args:
            strict_extension: When False, skip the extension allowlist check.
                              Set to False when registering the processor for a
                              custom extension via ChunkingService.register_processor().
        """
        super().__init__()
        self._encoding = encoding
        self._max_bytes = max_bytes
        self._strict_extension = strict_extension
        self._text: str = ""

    # ── Abstract Method Implementations ──────────────────────────────────────

    def load(self, source: Path | BinaryIO, *, file_name: str = "") -> None:
        """
        Read a Markdown or plain-text file.

        Args:
            source:    Filesystem Path or readable BinaryIO stream.
            file_name: Name stored in chunk metadata; auto-derived from path.

        Raises:
            FileNotFoundError: Path does not exist.
            ValueError:        Unsupported file type or oversized content.
            UnicodeDecodeError: File cannot be decoded with the configured encoding.
        """
        if isinstance(source, Path):
            if not source.exists():
                raise FileNotFoundError(f"File not found: {source}")
            if self._strict_extension and source.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(
                    f"Unsupported extension '{source.suffix}'. "
                    f"Expected one of {self.SUPPORTED_EXTENSIONS}"
                )
            if source.stat().st_size > self._max_bytes:
                raise ValueError(f"File {source.name} exceeds size limit.")
            self._raw_content = source.read_bytes()
            self._file_name = file_name or source.name

        elif hasattr(source, "read"):
            data = source.read()
            if isinstance(data, str):
                data = data.encode(self._encoding)
            if len(data) > self._max_bytes:
                raise ValueError("Stream exceeds size limit.")
            self._raw_content = data
            self._file_name = file_name or "upload.md"

        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        self._text = self._raw_content.decode(self._encoding)

    def extract_text(self) -> list[tuple[str, int | None]]:
        """
        Return the entire document as a single (text, None) tuple.

        Markdown has no pages, so page_number is always None.
        The ChunkingService's RecursiveCharacterTextSplitter uses heading
        markers (# / ##) as preferred split boundaries automatically.

        Raises:
            RuntimeError: If called before load().
        """
        self._assert_loaded()
        return [(self._text, None)]

    def chunk(
        self,
        *,
        document_id: str,
        source_file: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        extra_metadata: dict[str, str] | None = None,
    ):
        """
        Override: use MarkdownHeaderTextSplitter before character splitting
        so heading context is preserved in every chunk.
        """
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
        from rag_system.core.interfaces.document_processor import DocumentChunk

        self._assert_loaded()
        fname = source_file or self._file_name

        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,  # keep heading text inside chunk for context
        )
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        md_docs = md_splitter.split_text(self._text)
        char_docs = char_splitter.split_documents(md_docs)

        chunks = []
        for idx, doc in enumerate(char_docs):
            heading_meta = {
                k: v for k, v in doc.metadata.items()
                if k in ("h1", "h2", "h3")
            }
            merged_meta = {**(extra_metadata or {}), **heading_meta}
            chunks.append(
                DocumentChunk.create(
                    document_id=document_id,
                    source_file=fname,
                    page_number=None,
                    chunk_index=idx,
                    text=doc.page_content,
                    extra_metadata={k: str(v) for k, v in merged_meta.items()},
                )
            )
        return chunks

    # ── Convenience Properties ─────────────────────────────────────────────

    @property
    def sections(self) -> list[tuple[str, str]]:
        """
        Return (heading_text, body_text) pairs for every top-level section.
        Useful for structured previews or section-level metadata enrichment.
        Returns empty list if not yet loaded or no headings found.
        """
        if not self._text:
            return []
        parts: list[tuple[str, str]] = []
        positions = [(m.start(), m.group(2).rstrip('\r')) for m in _HEADING_RE.finditer(self._text)]
        for i, (pos, heading) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(self._text)
            body = self._text[pos:end].strip()
            parts.append((heading, body))
        return parts
