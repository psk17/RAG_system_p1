import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.runnables import RunnableLambda
from rag_system.rag.grounding import ensure_grounded
from rag_system.rag.prompts import RAG_PROMPT, CONVERSATIONAL_RAG_PROMPT
from rag_system.rag.models import RAGResult
from rag_system.rag.retriever import RetrieverService
from rag_system.rag.chain_manager import RAGChainManager
from rag_system.api.schemas.query import QueryRequest, QueryResponse, SourceChunk

@pytest.fixture(autouse=True)
def mock_build_llm():
    with patch("rag_system.rag.chain_manager.build_llm") as mock:
        mock.return_value = RunnableLambda(lambda _: "Employees can carry over up to 40 hours of PTO.")
        yield mock

class TestPhase4Grounding:
    def test_ensure_grounded_valid(self):
        ans = "The carryover limit is 40 hours."
        ctx = ["Employees can carry over up to 40 hours of PTO."]
        assert ensure_grounded(ans, ctx) == ans

    def test_ensure_grounded_empty_context(self):
        ans = "The carryover limit is 40 hours."
        assert ensure_grounded(ans, []) == "I cannot find the answer in the provided documents."

    def test_ensure_grounded_empty_answer(self):
        ctx = ["Context text"]
        assert ensure_grounded("", ctx) == "I cannot find the answer in the provided documents."

    def test_ensure_grounded_whitespace_answer(self):
        ctx = ["Context text"]
        assert ensure_grounded("   \n  ", ctx) == "I cannot find the answer in the provided documents."

    def test_ensure_grounded_none_context(self):
        assert ensure_grounded("Answer", None) == "I cannot find the answer in the provided documents."

class TestPhase4Prompts:
    def test_rag_prompt_messages(self):
        messages = RAG_PROMPT.messages
        assert len(messages) == 2
        assert messages[0].prompt.template is not None
        assert messages[1].prompt.template == "{question}"

    def test_conversational_prompt_messages(self):
        messages = CONVERSATIONAL_RAG_PROMPT.messages
        assert len(messages) == 2
        assert "history" in messages[0].prompt.template
        assert "context" in messages[0].prompt.template

    def test_rag_prompt_placeholders(self):
        input_vars = RAG_PROMPT.input_variables
        assert "context" in input_vars
        assert "question" in input_vars

    def test_conversational_prompt_placeholders(self):
        input_vars = CONVERSATIONAL_RAG_PROMPT.input_variables
        assert "history" in input_vars
        assert "context" in input_vars
        assert "question" in input_vars

    def test_system_prompt_rules_exist(self):
        template = RAG_PROMPT.messages[0].prompt.template
        assert "Answer ONLY from the provided context." in template
        assert "Never use outside knowledge." in template

class TestPhase4Schemas:
    def test_query_request_validation(self):
        req = QueryRequest(question="What is the carryover policy?")
        assert req.question == "What is the carryover policy?"
        assert req.session_id is None

    def test_query_request_with_session(self):
        req = QueryRequest(question="What is the carryover policy?", session_id="session-123")
        assert req.session_id == "session-123"

    def test_source_chunk_fields(self):
        chunk = SourceChunk(chunk_id="chk-1", source="doc.pdf", page_number=2, score=0.95, content="chunk content")
        assert chunk.chunk_id == "chk-1"
        assert chunk.page_number == 2
        assert chunk.score == 0.95

    def test_query_response_structure(self):
        chunk = SourceChunk(chunk_id="chk-1", source="doc.pdf", page_number=1, score=0.9, content="content")
        resp = QueryResponse(answer="Yes.", sources=[chunk])
        assert resp.answer == "Yes."
        assert len(resp.sources) == 1

class TestPhase4ChainManagerOrchestration:
    @pytest.mark.asyncio
    async def test_chain_manager_query_without_memory(self):
        mock_retriever = AsyncMock()
        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "chunk-1"
        mock_chunk.source_file = "test.pdf"
        mock_chunk.page_number = 1
        mock_chunk.text = "carry over up to 40 hours of PTO"
        
        mock_result = MagicMock()
        mock_result.chunk = mock_chunk
        mock_result.score = 0.95
        
        mock_retriever.retrieve.return_value = [mock_result]
        
        manager = RAGChainManager(retriever=mock_retriever)
        manager.llm = RunnableLambda(lambda _: "Employees can carry over up to 40 hours of PTO.")
        
        res = await manager.query("What is PTO limit?")
        assert isinstance(res, RAGResult)
        assert res.answer == "Employees can carry over up to 40 hours of PTO."
        assert len(res.contexts) == 1
        assert res.contexts[0].chunk_id == "chunk-1"
        assert res.contexts[0].score == 0.95

    @pytest.mark.asyncio
    async def test_chain_manager_query_handles_empty_retrieval(self):
        mock_retriever = AsyncMock()
        mock_retriever.retrieve.return_value = []
        
        manager = RAGChainManager(retriever=mock_retriever)
        manager.llm = RunnableLambda(lambda _: "I cannot find the answer in the provided documents.")
        
        res = await manager.query("What is the policy?")
        assert "I cannot find the answer" in res.answer
        assert len(res.contexts) == 0

    @pytest.mark.asyncio
    async def test_retriever_service_retrieve_calls_similarity_search(self):
        mock_store = AsyncMock()
        retriever = RetrieverService(mock_store, k=5, collection_name="custom-col")
        await retriever.retrieve("query text")
        mock_store.similarity_search.assert_called_once_with(
            query="query text",
            collection_name="custom-col",
            top_k=5
        )
