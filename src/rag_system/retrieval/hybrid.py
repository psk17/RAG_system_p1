try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class _BM25OkapiFallback:
        def __init__(self, corpus: list[list[str]]) -> None:
            self.corpus = corpus
            self.doc_freqs = corpus

        def get_scores(self, query: list[str]) -> list[float]:
            return [1.0] * len(self.corpus)

        def get_top_documents(self, query: list[str], documents: list, n: int = 5) -> list:
            return documents[:n]
    BM25Okapi = _BM25OkapiFallback

from typing import Any, List

class HybridRetriever:
    def __init__(self, vector_retriever: Any, corpus_documents: List[Any]) -> None:
        self.vector_retriever = vector_retriever
        self.documents = corpus_documents

        tokenized = [doc.page_content.split() for doc in corpus_documents]
        self.bm25 = BM25Okapi(tokenized)

    async def retrieve(self, query: str, vector_k: int = 20) -> List[Any]:
        try:
            vector_docs = await self.vector_retriever.ainvoke(query)
        except Exception:
            vector_docs = await self.vector_retriever.aget_relevant_documents(query)

        tokenized_query = query.split()
        bm25_docs = self.bm25.get_top_documents(tokenized_query, self.documents, n=5)

        seen: set[str] = set()
        combined: List[Any] = []
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined
