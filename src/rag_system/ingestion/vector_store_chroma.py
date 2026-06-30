"""
app/ingestion/vector_store_chroma.py
──────────────────────────────────────
ChromaDB adapter — implements BaseVectorStore for local / lightweight dev.

Swap this out for QdrantAdapter or PGVectorAdapter in production by
registering a different factory in the VectorStoreFactory (see bottom).

Embedding Strategy
──────────────────
• "openai"      → langchain_openai.OpenAIEmbeddings  (cloud, high quality)
• "huggingface" → langchain_huggingface.HuggingFaceEmbeddings (local, free)
• "fake"        → DeterministicFakeEmbedding  (tests — no API key needed)

Connection Modes
────────────────
• Persistent (default): data stored to CHROMA_PERSIST_DIR on disk.
• In-memory: pass persist_dir=None — used for integration tests.
• HTTP client: pass host/port to connect to a remote Chroma server.

Usage
─────
    adapter = ChromaAdapter.from_settings()
    await adapter.upsert(chunks, collection_name="hr_policies_2026")
    results = await adapter.similarity_search(
        "remote work policy",
        collection_name="hr_policies_2026",
        top_k=4,
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import warnings
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import Embeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]

from rag_system.core.interfaces.document_processor import DocumentChunk
from rag_system.core.interfaces.vector_store import BaseVectorStore, SearchResult, VectorStoreError

# Disable ChromaDB anonymous telemetry before importing chromadb.
# This prevents posthog's capture() signature mismatch errors from polluting logs.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")
warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Embedding Factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_embeddings(provider: str, **kwargs: Any) -> Embeddings:
    """
    Instantiate the correct LangChain Embeddings class for `provider`.

    Args:
        provider: "openai" | "huggingface" | "ollama" | "fake"
        **kwargs: Forwarded to the underlying constructor (model name, etc.)
    """
    if provider == "openai":
        openai_api_key = kwargs.get("openai_api_key")
        if not openai_api_key or str(openai_api_key).startswith("sk-abcdef") or "mock" in str(openai_api_key).lower() or "your_" in str(openai_api_key).lower():
            logger.warning(
                "OpenAI API key is missing or placeholder; falling back to HuggingFace embeddings."
            )
            provider = "huggingface"
        else:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                model=kwargs.get("model", "text-embedding-3-small"),
                api_key=openai_api_key,
            )
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=kwargs.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            model_kwargs={"device": "cpu"},
        )
    if provider == "ollama":
        try:
            from langchain_community.embeddings import OllamaEmbeddings
        except ImportError:
            from langchain_ollama import OllamaEmbeddings
        base_url = str(kwargs.get("ollama_base_url")) if kwargs.get("ollama_base_url") else "http://localhost:11434"
        return OllamaEmbeddings(
            base_url=base_url,
            model=kwargs.get("ollama_model", "llama3"),
        )
    if provider == "fake":
        # Deterministic fake embeddings — for unit/integration tests only
        from langchain_core.embeddings.fake import DeterministicFakeEmbedding
        return DeterministicFakeEmbedding(size=kwargs.get("size", 384))

    raise ValueError(
        f"Unknown embedding provider '{provider}'. "
        "Choose 'openai', 'huggingface', 'ollama', or 'fake'."
    )


# ──────────────────────────────────────────────────────────────────────────────
# ChromaAdapter
# ──────────────────────────────────────────────────────────────────────────────

class ChromaAdapter(BaseVectorStore):
    """
    BaseVectorStore implementation backed by ChromaDB.

    All public methods are async; the synchronous Chroma client is wrapped
    with asyncio.to_thread() so this adapter is safe inside FastAPI routes.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        persist_dir: Path | None = Path("./data/chroma"),
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """
        Args:
            embeddings:   LangChain Embeddings instance (OpenAI, HF, Fake…).
            persist_dir:  Local directory for on-disk persistence.
                          Pass None for a pure in-memory client (tests).
            host:         Remote Chroma server hostname (optional).
            port:         Remote Chroma server port (optional).
        """
        self._embeddings = embeddings
        self._persist_dir = persist_dir

        # Build the correct chromadb client
        if host and port:
            self._client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        elif persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            # Pure in-memory — for tests
            self._client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False),
            )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> "ChromaAdapter":
        """
        Instantiate directly from the app Settings singleton.
        Uses embedding_provider and chroma_* config values.
        """
        from rag_system.core.config.settings import get_settings
        s = get_settings()
        embeddings = _build_embeddings(
            s.embedding_provider.value,
            model=s.openai_embedding_model,
            model_name=s.huggingface_embedding_model,
            openai_api_key=s.openai_api_key,
            ollama_base_url=s.ollama_base_url,
            ollama_model=s.ollama_model,
        )
        return cls(
            embeddings=embeddings,
            persist_dir=s.chroma_persist_dir,
            host=s.chroma_host if s.chroma_host != "localhost" else None,
            port=s.chroma_port if s.chroma_host != "localhost" else None,
        )

    @classmethod
    def for_testing(cls) -> "ChromaAdapter":
        """
        In-memory adapter with fake embeddings — no API keys, no disk I/O.
        Import this in your tests instead of from_settings().
        """
        return cls(
            embeddings=_build_embeddings("fake"),
            persist_dir=None,
        )

    # ── Abstract Method Implementations ──────────────────────────────────────

    async def upsert(
        self,
        chunks: list[DocumentChunk],
        *,
        collection_name: str,
    ) -> int:
        """
        Embed `chunks` and upsert into `collection_name`.

        Uses chunk.chunk_id as the Chroma document ID for idempotency —
        re-ingesting the same file updates existing records instead of
        creating duplicates.
        """
        if not chunks:
            return 0

        try:
            def _sync_upsert() -> int:
                vectorstore = Chroma(
                    client=self._client,
                    collection_name=collection_name,
                    embedding_function=self._embeddings,
                )
                texts = [c.text for c in chunks]
                metadatas = [
                    {k: v for k, v in c.to_metadata_dict().items() if v is not None}
                    for c in chunks
                ]
                ids = [c.chunk_id for c in chunks]

                vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
                return len(chunks)

            count = await asyncio.to_thread(_sync_upsert)
            logger.info("Upserted %d chunks into collection '%s'.", count, collection_name)
            return count

        except Exception as exc:
            raise VectorStoreError(
                f"Upsert failed for collection '{collection_name}'", cause=exc
            ) from exc

    async def similarity_search(
        self,
        query: str,
        *,
        collection_name: str,
        top_k: int = 4,
        metadata_filter: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        """
        Embed `query` and return the top_k most similar chunks.

        Args:
            metadata_filter: Chroma `where` clause, e.g.
                             {"source_file": "HR_Handbook.pdf"}.
                             Passed directly to Chroma's where= parameter.
        """
        try:
            def _sync_search() -> list[SearchResult]:
                vectorstore = Chroma(
                    client=self._client,
                    collection_name=collection_name,
                    embedding_function=self._embeddings,
                )
                where = (
                    {k: {"$eq": v} for k, v in metadata_filter.items()}
                    if metadata_filter
                    else None
                )
                results = vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=top_k,
                    filter=where,
                )
                search_results = []
                for doc, score in results:
                    meta = doc.metadata
                    chunk = DocumentChunk(
                        chunk_id=meta.get("chunk_id", ""),
                        document_id=meta.get("document_id", ""),
                        source_file=meta.get("source_file", ""),
                        page_number=meta.get("page_number"),
                        chunk_index=int(meta.get("chunk_index", 0)),
                        text=doc.page_content,
                        extra_metadata={
                            k: v for k, v in meta.items()
                            if k not in (
                                "chunk_id", "document_id", "source_file",
                                "page_number", "chunk_index"
                            )
                        },
                    )
                    # Clamp score to [0, 1] — Chroma can return slightly negative scores
                    clamped = max(0.0, min(1.0, float(score)))
                    search_results.append(SearchResult(chunk=chunk, score=clamped))
                return search_results

            return await asyncio.to_thread(_sync_search)

        except Exception as exc:
            raise VectorStoreError(
                f"Similarity search failed in '{collection_name}'", cause=exc
            ) from exc

    async def delete_document(
        self,
        document_id: str,
        *,
        collection_name: str,
    ) -> int:
        """Delete all chunks whose metadata.document_id == document_id."""
        try:
            def _sync_delete() -> int:
                col = self._client.get_or_create_collection(collection_name)
                existing = col.get(where={"document_id": {"$eq": document_id}})
                ids = existing.get("ids", [])
                if ids:
                    col.delete(ids=ids)
                return len(ids)

            deleted = await asyncio.to_thread(_sync_delete)
            logger.info(
                "Deleted %d chunks for document '%s' from '%s'.",
                deleted, document_id, collection_name,
            )
            return deleted

        except Exception as exc:
            raise VectorStoreError(
                f"Delete failed for document '{document_id}'", cause=exc
            ) from exc

    async def list_collections(self) -> list[str]:
        """Return names of all Chroma collections."""
        try:
            cols = await asyncio.to_thread(self._client.list_collections)
            # chromadb ≥0.5 returns Collection objects; earlier versions return strings
            return [c.name if hasattr(c, "name") else str(c) for c in cols]
        except Exception as exc:
            raise VectorStoreError("Failed to list collections", cause=exc) from exc
