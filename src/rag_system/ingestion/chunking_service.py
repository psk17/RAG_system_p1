"""
app/ingestion/chunking_service.py
───────────────────────────────────
Dedicated ChunkingService that selects the right processor for a file,
drives the load → chunk pipeline, and returns enriched DocumentChunks.

Responsibilities
────────────────
• Processor registry: maps file extension → concrete BaseDocumentProcessor.
• Semantic boundary configuration: custom separator hierarchy ensures
  headings, paragraphs, sentences, and words are preferred split points
  (in that order) so context is never broken mid-sentence.
• document_id generation: deterministic SHA-256 hash of (filename + mtime)
  so re-ingesting the same file is idempotent.
• Metadata enrichment: injects collection_id and any caller-supplied tags
  into every chunk's extra_metadata so vector DB filters work (FR-RET-3).

Usage
─────
    service = ChunkingService()
    chunks = service.process_file(
        path=Path("docs/report.pdf"),
        collection_id="hr_policies_2026",
    )

    # Batch a whole directory
    all_chunks = service.process_directory(
        directory=Path("./documents"),
        collection_id="company_kb",
    )
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable

from rag_system.core.config.settings import get_settings
from rag_system.core.interfaces.document_processor import BaseDocumentProcessor, DocumentChunk
from rag_system.ingestion.pdf_processor import PDFProcessor
from rag_system.ingestion.markdown_processor import MarkdownProcessor

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Separator hierarchy — drives RecursiveCharacterTextSplitter
# Priority: Markdown headings → blank lines → sentences → words → chars
# ──────────────────────────────────────────────────────────────────────────────
SEMANTIC_SEPARATORS: list[str] = [
    "\n## ",    # H2 headings
    "\n### ",   # H3 headings
    "\n\n",     # paragraph break
    "\n",       # line break
    ". ",       # sentence boundary
    "! ",
    "? ",
    "; ",
    ", ",
    " ",        # word boundary
    "",         # character fallback
]

# Extension → factory callable
ProcessorFactory = Callable[[], BaseDocumentProcessor]

_DEFAULT_REGISTRY: dict[str, ProcessorFactory] = {
    ".pdf":      lambda: PDFProcessor(),
    ".md":       lambda: MarkdownProcessor(),
    ".markdown": lambda: MarkdownProcessor(),
    ".txt":      lambda: MarkdownProcessor(),
    ".text":     lambda: MarkdownProcessor(),
}


class ChunkingService:
    """
    Orchestrates file loading, processor selection, and chunking.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        registry: dict[str, ProcessorFactory] | None = None,
    ) -> None:
        """
        Args:
            chunk_size:    Characters per chunk; defaults to settings value.
            chunk_overlap: Overlap between chunks; defaults to settings value.
            registry:      Extension → processor factory map; override for
                           custom formats (e.g. DOCX, HTML).
        """
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._registry: dict[str, ProcessorFactory] = registry or dict(_DEFAULT_REGISTRY)

    # ── Public API ────────────────────────────────────────────────────────────

    def process_file(
        self,
        path: Path,
        *,
        collection_id: str = "default",
        extra_tags: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        """
        Load, parse, and chunk a single file.

        Args:
            path:          Absolute or relative path to the document.
            collection_id: Logical namespace stored in chunk metadata for
                           filtered retrieval (FR-RET-3).
            extra_tags:    Any additional key/value pairs to embed in metadata.

        Returns:
            Ordered list of DocumentChunks with full metadata.

        Raises:
            ValueError:    Unsupported file extension.
            FileNotFoundError: File does not exist.
        """
        ext = path.suffix.lower()
        if ext not in self._registry:
            raise ValueError(
                f"No processor registered for extension '{ext}'. "
                f"Supported: {list(self._registry.keys())}"
            )

        processor = self._registry[ext]()
        processor.load(path)

        document_id = self._make_document_id(path)
        metadata: dict[str, str] = {"collection_id": collection_id, **(extra_tags or {})}

        chunks = processor.chunk(
            document_id=document_id,
            source_file=path.name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            extra_metadata=metadata,
        )

        logger.info(
            "Processed '%s' → %d chunks (doc_id=%s, collection=%s)",
            path.name, len(chunks), document_id, collection_id,
        )
        return chunks

    def process_directory(
        self,
        directory: Path,
        *,
        collection_id: str = "default",
        recursive: bool = True,
        extra_tags: dict[str, str] | None = None,
    ) -> list[DocumentChunk]:
        """
        Process all supported files in `directory`.

        Args:
            directory:     Root directory to scan.
            collection_id: Applied to every chunk from this directory.
            recursive:     Whether to descend into subdirectories.
            extra_tags:    Extra metadata tags added to every chunk.

        Returns:
            Flat list of all DocumentChunks from all files.
        """
        if not directory.is_dir():
            raise ValueError(f"'{directory}' is not a directory.")

        glob = "**/*" if recursive else "*"
        files = [
            f for f in directory.glob(glob)
            if f.is_file() and f.suffix.lower() in self._registry
        ]

        if not files:
            logger.warning("No supported files found in '%s'.", directory)
            return []

        all_chunks: list[DocumentChunk] = []
        errors: list[tuple[Path, Exception]] = []

        for file_path in sorted(files):
            try:
                chunks = self.process_file(
                    file_path,
                    collection_id=collection_id,
                    extra_tags=extra_tags,
                )
                all_chunks.extend(chunks)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to process '%s': %s", file_path, exc)
                errors.append((file_path, exc))

        logger.info(
            "Directory scan complete: %d files, %d chunks, %d errors.",
            len(files), len(all_chunks), len(errors),
        )
        return all_chunks

    def register_processor(self, extension: str, factory: ProcessorFactory) -> None:
        """
        Register a custom processor for a file extension at runtime.

        Args:
            extension: Lowercase extension including dot, e.g. ".docx".
            factory:   Zero-argument callable returning a BaseDocumentProcessor.
        """
        if not extension.startswith("."):
            raise ValueError(f"Extension must start with '.', got '{extension}'")
        self._registry[extension.lower()] = factory
        logger.debug("Registered processor for '%s'.", extension)

    # ── Internal Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _make_document_id(path: Path) -> str:
        """
        Generate a stable, deterministic document_id from filename + mtime.
        Re-ingesting an unchanged file produces the same ID → idempotent upserts.
        """
        mtime = str(path.stat().st_mtime) if path.exists() else "0"
        raw = f"{path.name}:{mtime}"
        return "doc_" + hashlib.sha256(raw.encode()).hexdigest()[:16]
