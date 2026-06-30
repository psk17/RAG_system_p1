"""
api/dependencies.py
────────────────────
Service singletons initialized once at startup.
Redis is optional for baseline — if not available, memory features are disabled.
"""
import logging
from rag_system.ingestion.vector_store_chroma import ChromaAdapter
from rag_system.ingestion.ingestion_service import IngestionService
from rag_system.rag.chain_manager import RAGChainManager
from rag_system.rag.retriever import RetrieverService
from rag_system.core.config.settings import get_settings

logger = logging.getLogger(__name__)

_vector_store = None
_rag_manager = None
_redis_client = None
_memory_store = None
_ingestion_service = None


async def initialize_services():
    global _vector_store, _rag_manager, _redis_client, _memory_store, _ingestion_service
    settings = get_settings()

    # ── Vector store (Chroma, local) ──────────────────────────────────────────
    _vector_store = ChromaAdapter.from_settings()
    _ingestion_service = IngestionService(_vector_store)
    retriever = RetrieverService(_vector_store)

    # ── Redis / Memory (optional — gracefully skip if unavailable) ────────────
    try:
        from redis.asyncio import from_url as redis_from_url
        from rag_system.memory.redis_memory import RedisMemoryStore
        _redis_client = redis_from_url(settings.redis_url, decode_responses=True)
        # Ping to verify connection
        await _redis_client.ping()
        _memory_store = RedisMemoryStore(_redis_client)
        logger.info("Redis connected at %s — memory/session features enabled.", settings.redis_url)
    except Exception as exc:
        logger.warning(
            "Redis not available (%s). Running without session memory. "
            "Install Redis or set REDIS_URL to enable memory features.",
            exc,
        )
        _redis_client = None
        _memory_store = None

    # ── RAG chain manager ─────────────────────────────────────────────────────
    _rag_manager = RAGChainManager(retriever, _memory_store)
    logger.info("All services initialized successfully.")


def get_vector_store():
    return _vector_store


def get_rag_manager():
    return _rag_manager


def get_redis_client():
    return _redis_client


def get_memory_store():
    return _memory_store


def get_ingestion_service():
    return _ingestion_service
