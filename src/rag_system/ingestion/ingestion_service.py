"""
app/ingestion/ingestion_service.py
────────────────────────────────────
Top-level IngestionService — the single entry point called by FastAPI routes
and the CLI script. Combines ChunkingService (parse → chunk) with a
BaseVectorStore (embed → persist).

Responsibilities
────────────────
• Coordinate the full load → chunk → embed → upsert pipeline.
• Return a structured IngestionResult with counts and timing data.
• Handle per-file errors gracefully without aborting the entire batch.

Usage
─────
    from rag_system.ingestion.vector_store_chroma import ChromaAdapter
    from rag_system.ingestion.ingestion_service import IngestionService

    store   = ChromaAdapter.from_settings()
    service = IngestionService(vector_store=store)

    result = await service.ingest_file(
        path=Path("docs/hr_handbook.pdf"),
        collection_id="hr_policies_2026",
    )
    print(result)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from rag_system.core.interfaces.vector_store import BaseVectorStore
from rag_system.ingestion.chunking_service import ChunkingService

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Structured return type for every ingest operation."""

    source: str                    # filename or directory path
    collection_id: str
    chunks_processed: int = 0
    chunks_upserted: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"{status} '{self.source}' → {self.chunks_upserted} chunks upserted "
            f"into '{self.collection_id}' in {self.duration_seconds:.2f}s"
            + (f" | errors: {self.errors}" if self.errors else "")
        )


class IngestionService:
    """
    Orchestrates the full ingestion pipeline for files and directories.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        chunking_service: ChunkingService | None = None,
    ) -> None:
        self._store = vector_store
        self._chunker = chunking_service or ChunkingService()

    # ── Public API ────────────────────────────────────────────────────────────

    async def ingest_file(
        self,
        path: Path,
        *,
        collection_id: str = "default",
        extra_tags: dict[str, str] | None = None,
    ) -> IngestionResult:
        """
        Ingest a single file into the vector store.

        Args:
            path:          Path to the document.
            collection_id: Vector DB collection / namespace.
            extra_tags:    Extra metadata to attach to every chunk.

        Returns:
            IngestionResult with chunk counts, timing, and any errors.
        """
        result = IngestionResult(source=path.name, collection_id=collection_id)
        t0 = time.perf_counter()

        try:
            chunks = self._chunker.process_file(
                path,
                collection_id=collection_id,
                extra_tags=extra_tags,
            )
            result.chunks_processed = len(chunks)

            if chunks:
                result.chunks_upserted = await self._store.upsert(
                    chunks, collection_name=collection_id
                )
        except Exception as exc:  # noqa: BLE001
            msg = f"Ingestion failed for '{path.name}': {exc}"
            logger.error(msg)
            result.errors.append(msg)
        finally:
            result.duration_seconds = time.perf_counter() - t0

        logger.info(str(result))
        return result

    async def ingest_directory(
        self,
        directory: Path,
        *,
        collection_id: str = "default",
        recursive: bool = True,
        extra_tags: dict[str, str] | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest every supported file in `directory`.

        Files are processed one-by-one so a single failure doesn't abort
        the entire batch. Each file gets its own IngestionResult entry.
        """
        if not directory.is_dir():
            raise ValueError(f"'{directory}' is not a directory.")

        supported = {".pdf", ".md", ".markdown", ".txt", ".text"}
        glob = "**/*" if recursive else "*"
        files = sorted(
            f for f in directory.glob(glob)
            if f.is_file() and f.suffix.lower() in supported
        )

        results: list[IngestionResult] = []
        for file_path in files:
            r = await self.ingest_file(
                file_path,
                collection_id=collection_id,
                extra_tags=extra_tags,
            )
            results.append(r)

        total_chunks = sum(r.chunks_upserted for r in results)
        total_errors = sum(len(r.errors) for r in results)
        logger.info(
            "Batch complete: %d files, %d chunks upserted, %d errors.",
            len(results), total_chunks, total_errors,
        )
        return results
