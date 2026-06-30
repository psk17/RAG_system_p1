import os
import pytest

# Set env variables before any tests run
os.environ["OPENAI_API_KEY"] = "mock-openai-key-for-testing"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"

@pytest.fixture(autouse=True)
def setup_global_fake_services():
    from rag_system.api.app import app
    from rag_system.api.dependencies import get_vector_store, get_ingestion_service, get_rag_manager
    from rag_system.ingestion.vector_store_chroma import ChromaAdapter
    from rag_system.ingestion.ingestion_service import IngestionService
    from rag_system.rag.chain_manager import RAGChainManager
    from rag_system.rag.retriever import RetrieverService
    
    fake_store = ChromaAdapter.for_testing()
    fake_ingestion = IngestionService(fake_store)
    retriever = RetrieverService(fake_store)
    fake_rag = RAGChainManager(retriever)
    
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_ingestion_service] = lambda: fake_ingestion
    app.dependency_overrides[get_rag_manager] = lambda: fake_rag
    yield
    app.dependency_overrides.clear()
