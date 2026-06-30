import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.documents import Document
from rag_system.retrieval.reranker import BGEReranker
from rag_system.retrieval.parent_child import create_parent_child_retriever
from rag_system.retrieval.hybrid import HybridRetriever
from rag_system.evaluation.ragas_runner import evaluate_rag
from rag_system.evaluation.benchmark import benchmark

@pytest.fixture(autouse=True)
def mock_cross_encoder():
    with patch("rag_system.retrieval.reranker.CrossEncoder") as mock_class:
        mock_instance = MagicMock()
        mock_instance.predict.side_effect = lambda pairs: [0.95 - (i * 0.05) for i in range(len(pairs))]
        mock_class.return_value = mock_instance
        yield mock_class

class TestPhase6Reranker:
    def test_reranker_init(self):
        reranker = BGEReranker()
        assert reranker is not None

    def test_rerank_empty_list(self):
        reranker = BGEReranker()
        assert reranker.rerank("query", []) == []

    def test_rerank_attaches_score_metadata(self):
        reranker = BGEReranker()
        docs = [Document(page_content="doc1"), Document(page_content="doc2")]
        ranked = reranker.rerank("query", docs, top_k=2)
        assert len(ranked) == 2
        assert "score" in ranked[0].metadata
        assert ranked[0].metadata["score"] >= ranked[1].metadata["score"]

    def test_rerank_limits_top_k(self):
        reranker = BGEReranker()
        docs = [Document(page_content="1"), Document(page_content="2"), Document(page_content="3")]
        ranked = reranker.rerank("query", docs, top_k=2)
        assert len(ranked) == 2

    def test_rerank_supports_search_result_objects(self):
        reranker = BGEReranker()
        mock_res1 = MagicMock()
        mock_res1.chunk.text = "result 1"
        mock_res1.metadata = {}
        mock_res2 = MagicMock()
        mock_res2.chunk.text = "result 2"
        mock_res2.metadata = {}
        
        ranked = reranker.rerank("query", [mock_res1, mock_res2], top_k=1)
        assert len(ranked) == 1

    def test_rerank_scores_descending(self):
        reranker = BGEReranker()
        docs = [Document(page_content="a"), Document(page_content="b"), Document(page_content="c")]
        ranked = reranker.rerank("query", docs, top_k=3)
        scores = [d.metadata["score"] for d in ranked]
        assert scores == sorted(scores, reverse=True)

class TestPhase6ParentChild:
    def test_parent_child_retriever_creation(self):
        mock_store = MagicMock()
        mock_store._client = MagicMock()
        mock_store._embeddings = MagicMock()
        
        retriever = create_parent_child_retriever(mock_store)
        assert retriever is not None
        assert retriever.parent_splitter._chunk_size == 1500
        assert retriever.child_splitter._chunk_size == 300

    def test_parent_child_retriever_overlap(self):
        mock_store = MagicMock()
        mock_store._client = MagicMock()
        mock_store._embeddings = MagicMock()
        
        retriever = create_parent_child_retriever(mock_store)
        assert retriever.parent_splitter._chunk_overlap == 200
        assert retriever.child_splitter._chunk_overlap == 50

    def test_parent_child_docstore_type(self):
        mock_store = MagicMock()
        mock_store._client = MagicMock()
        mock_store._embeddings = MagicMock()
        
        retriever = create_parent_child_retriever(mock_store)
        from langchain.storage import InMemoryStore
        assert isinstance(retriever.docstore, InMemoryStore)

    def test_parent_child_vectorstore_initialized(self):
        mock_store = MagicMock()
        mock_store._client = MagicMock()
        mock_store._embeddings = MagicMock()
        
        retriever = create_parent_child_retriever(mock_store)
        assert retriever.vectorstore is not None

    def test_parent_child_retriever_search_kwargs(self):
        mock_store = MagicMock()
        mock_store._client = MagicMock()
        mock_store._embeddings = MagicMock()
        
        retriever = create_parent_child_retriever(mock_store)
        assert isinstance(retriever.search_kwargs, dict)

class TestPhase6Hybrid:
    def test_hybrid_init(self):
        mock_retriever = AsyncMock()
        corpus = [Document(page_content="hello world")]
        hybrid = HybridRetriever(mock_retriever, corpus)
        assert hybrid.vector_retriever == mock_retriever
        assert len(hybrid.documents) == 1

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_returns_combined_list(self):
        mock_retriever = AsyncMock()
        doc1 = Document(page_content="vector doc")
        mock_retriever.ainvoke.return_value = [doc1]
        
        doc2 = Document(page_content="keyword doc")
        hybrid = HybridRetriever(mock_retriever, [doc1, doc2])
        
        results = await hybrid.retrieve("keyword")
        assert len(results) >= 1
        assert any(d.page_content == "vector doc" for d in results)

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_avoids_duplicates(self):
        mock_retriever = AsyncMock()
        doc = Document(page_content="duplicate text")
        mock_retriever.ainvoke.return_value = [doc]
        
        hybrid = HybridRetriever(mock_retriever, [doc])
        results = await hybrid.retrieve("duplicate text")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_fallback_to_aget_relevant_documents(self):
        mock_retriever = MagicMock()
        del mock_retriever.ainvoke
        mock_retriever.aget_relevant_documents = AsyncMock(return_value=[Document(page_content="test")])
        
        hybrid = HybridRetriever(mock_retriever, [Document(page_content="test")])
        results = await hybrid.retrieve("query")
        assert len(results) == 1
        mock_retriever.aget_relevant_documents.assert_called_once()

    def test_hybrid_bm25_tokenization_length(self):
        mock_retriever = AsyncMock()
        corpus = [Document(page_content="first document text"), Document(page_content="second doc")]
        hybrid = HybridRetriever(mock_retriever, corpus)
        assert len(hybrid.bm25.doc_freqs) == 2

    def test_hybrid_bm25_scores(self):
        mock_retriever = AsyncMock()
        doc = Document(page_content="the quick brown fox")
        hybrid = HybridRetriever(mock_retriever, [doc])
        scores = hybrid.bm25.get_scores(["quick"])
        assert len(scores) == 1
        assert scores[0] > 0

class TestPhase6Evaluation:
    def test_evaluate_rag_returns_mock_offline(self):
        dataset = [{"question": "Q", "answer": "A", "source": "S"}]
        results = evaluate_rag(dataset)
        assert "faithfulness" in results
        assert "answer_relevancy" in results
        assert "context_precision" in results

    def test_evaluate_rag_handles_lists_directly(self):
        dataset = [{"question": "Is it carrying over?", "answer": "40 hours", "source": "handbook.pdf"}]
        res = evaluate_rag(dataset)
        assert res["faithfulness"] == 0.95

    def test_benchmark_executes_without_exceptions(self):
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            benchmark()
            output = sys.stdout.getvalue()
            assert "faithfulness" in output
        finally:
            sys.stdout = old_stdout

    def test_gold_dataset_json_format(self):
        import json
        from pathlib import Path
        p = Path(__file__).parent.parent.parent / "src" / "rag_system" / "evaluation" / "datasets" / "gold_dataset.json"
        with open(p, "r") as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) > 0
        assert "question" in data[0]

    def test_evaluate_rag_empty_dataset(self):
        res = evaluate_rag([])
        assert "faithfulness" in res
