from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Any

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore

def create_parent_child_retriever(
    vector_store: Any,
    collection_name: str = "default",
) -> ParentDocumentRetriever:
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

    doc_store = InMemoryStore()

    # Get underlying LangChain vectorstore from ChromaAdapter
    underlying_store = Chroma(
        client=vector_store._client,
        collection_name=collection_name,
        embedding_function=vector_store._embeddings,
    )

    retriever = ParentDocumentRetriever(
        vectorstore=underlying_store,
        docstore=doc_store,
        parent_splitter=parent_splitter,
        child_splitter=child_splitter,
    )

    return retriever
