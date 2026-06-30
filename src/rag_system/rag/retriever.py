from rag_system.core.interfaces.vector_store import BaseVectorStore

class RetrieverService:
    def __init__(
        self,
        vector_store: BaseVectorStore,
        k: int = 4,
        collection_name: str = "default",
    ):
        self.vector_store = vector_store
        self.k = k
        self.collection_name = collection_name

    async def retrieve(
        self,
        query: str,
    ):
        return await self.vector_store.similarity_search(
            query=query,
            collection_name=self.collection_name,
            top_k=self.k,
        )

