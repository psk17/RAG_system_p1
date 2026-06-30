"""
app/core/interfaces/vector_store.py
─────────────────────────────────────
Abstract contract for every vector database adapter.

Concrete implementations (Phase 2):
  - ChromaAdapter   (chromadb)
  - QdrantAdapter   (qdrant-client)
  - PGVectorAdapter (pgvector + psycopg2)

Design Notes
────────────
• All methods are async-first; sync vector DBs should be wrapped with
  `asyncio.to_thread()` inside the concrete adapter.
• `SearchResult` is a lightweight value object — the calling service never
  touches the raw DB response directly, keeping the adapter truly swappable.
• `upsert()` accepts DocumentChunks (not raw strings) so the adapter owns
  the responsibility of calling the embedding service and structuring the
  payload for whichever DB is configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_system.core.interfaces.document_processor import DocumentChunk


# ──────────────────────────────────────────────────────────────────────────────
# Value Objects
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SearchResult:
    """
    A single ranked result returned from similarity search.

    Attributes
    ──────────
    chunk:      The original DocumentChunk that matched.
    score:      Cosine similarity score in [0.0, 1.0]; higher = more relevant.
                Adapters that return distance instead of similarity must
                normalise before returning (score = 1 - distance).
    """

    chunk: DocumentChunk
    score: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"SearchResult.score must be in [0, 1]; got {self.score}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Abstract Base Class
# ──────────────────────────────────────────────────────────────────────────────

class BaseVectorStore(ABC):
    """
    Interface for all vector database adapters.

    Subclasses MUST override:
        upsert()            — embed chunks and persist them
        similarity_search() — retrieve top-K chunks by query vector
        delete_document()   — remove all chunks for a given document_id
        list_collections()  — list available namespaces / collections

    Subclasses MAY override:
        collection_exists() — default implementation calls list_collections()
    """

    # ── Abstract Methods ──────────────────────────────────────────────────────

    @abstractmethod
    async def upsert(
        self,
        chunks: list[DocumentChunk],
        *,
        collection_name: str,
    ) -> int:
        """
        Embed `chunks` and persist them to `collection_name`.

        The adapter is responsible for:
          1. Calling the embedding service to convert chunk.text → vector.
          2. Structuring the payload (id, vector, metadata) for the target DB.
          3. Performing an upsert (insert-or-update) using chunk.chunk_id as
             the unique key so re-ingestion is idempotent.

        Args:
            chunks:          Ordered list of DocumentChunks to embed and store.
            collection_name: The target collection / namespace.

        Returns:
            Number of chunks successfully upserted.

        Raises:
            VectorStoreError: Wraps any DB-specific exception for uniform
                              error handling in the calling service.
        """
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        *,
        collection_name: str,
        top_k: int = 4,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """
        Embed `query` and retrieve the top-K most similar chunks.

        Args:
            query:           Natural language question from the user.
            collection_name: Collection to search within.
            top_k:           Maximum number of results to return (FR-RET-2).
            metadata_filter: Optional key/value pairs to pre-filter results
                             before vector scoring (FR-RET-3).
                             Example: {"source_file": "HR_Handbook_2026.pdf"}

        Returns:
            List of SearchResult objects ordered by descending similarity score.
            The list may be shorter than top_k if fewer matches exist.

        Raises:
            VectorStoreError: Wraps any DB-specific exception.
        """
        ...

    @abstractmethod
    async def delete_document(
        self,
        document_id: str,
        *,
        collection_name: str,
    ) -> int:
        """
        Delete every chunk belonging to `document_id` from `collection_name`.

        Args:
            document_id:     The parent document whose chunks should be removed.
            collection_name: Collection to target.

        Returns:
            Number of chunks deleted.

        Raises:
            VectorStoreError: If the delete operation fails.
        """
        ...

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """
        Return the names of all collections managed by this adapter.

        Returns:
            List of collection name strings (may be empty).
        """
        ...

    # ── Concrete Methods ──────────────────────────────────────────────────────

    async def collection_exists(self, collection_name: str) -> bool:
        """
        Return True if `collection_name` exists.

        Default implementation calls list_collections(); override if the
        target DB exposes a cheaper existence-check operation.
        """
        return collection_name in await self.list_collections()


# ──────────────────────────────────────────────────────────────────────────────
# Custom Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class VectorStoreError(Exception):
    """
    Raised by any BaseVectorStore implementation when a DB operation fails.
    Wraps the underlying driver exception for uniform error handling.
    """

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (caused by: {self.cause!r})" if self.cause else base
