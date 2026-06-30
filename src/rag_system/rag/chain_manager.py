from datetime import datetime
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from rag_system.rag.models import RetrievedContext, RAGResult
from rag_system.rag.prompts import RAG_PROMPT, CONVERSATIONAL_RAG_PROMPT
from rag_system.rag.factories import build_llm
from rag_system.rag.grounding import ensure_grounded
from rag_system.memory.models import ChatMessage
from rag_system.telemetry.tracing import tracer
from rag_system.telemetry.metrics import QUERY_COUNTER, QUERY_DURATION

class RAGChainManager:
    def __init__(self, retriever, memory_store=None, reranker=None):
        self.retriever = retriever
        self.memory_store = memory_store
        self.reranker = reranker
        self.llm = build_llm()

    async def query(
        self,
        question: str,
        session_id: str | None = None,
    ) -> RAGResult:
        with QUERY_DURATION.time():
            QUERY_COUNTER.inc()
            
            with tracer.start_as_current_span("retrieval") as span:
                docs = await self.retriever.retrieve(question)
                span.set_attribute("docs_found", len(docs))
                
            if self.reranker and docs:
                with tracer.start_as_current_span("reranking"):
                    docs = self.reranker.rerank(question, docs, top_k=4)

            context_texts = []
            contexts = []
            for d in docs:
                if hasattr(d, "chunk"):
                    text = d.chunk.text
                    chunk_id = d.chunk.chunk_id
                    source = d.chunk.source_file
                    page_number = d.chunk.page_number
                    score = d.score
                else:
                    text = d.page_content
                    chunk_id = d.metadata.get("chunk_id", d.metadata.get("doc_id", "unknown"))
                    source = d.metadata.get("source_file", d.metadata.get("source", "unknown"))
                    page_number = d.metadata.get("page_number")
                    score = d.metadata.get("score", 0.0)
                
                context_texts.append(text)
                contexts.append(
                    RetrievedContext(
                        chunk_id=chunk_id,
                        source=source,
                        page_number=page_number,
                        content=text,
                        score=score,
                    )
                )
            context_text = "\n\n".join(context_texts)
            
            if self.memory_store and session_id:
                history = await self.memory_store.get_history(session_id)
                history_text = "\n".join(f"{m.role}: {m.content}" for m in history)
                
                payload = {
                    "question": question,
                    "history": history_text,
                    "context": context_text,
                }
                chain = (
                    RunnableLambda(lambda _: payload)
                    | CONVERSATIONAL_RAG_PROMPT
                    | self.llm
                    | StrOutputParser()
                )
                with tracer.start_as_current_span("generation"):
                    answer = await chain.ainvoke({})
                answer = ensure_grounded(answer, context_texts)
                
                await self.memory_store.append_message(
                    session_id,
                    ChatMessage(role="user", content=question, timestamp=datetime.utcnow())
                )
                await self.memory_store.append_message(
                    session_id,
                    ChatMessage(role="assistant", content=answer, timestamp=datetime.utcnow())
                )
            else:
                payload = {
                    "question": question,
                    "context": context_text,
                }
                chain = (
                    RunnableLambda(lambda _: payload)
                    | RAG_PROMPT
                    | self.llm
                    | StrOutputParser()
                )
                with tracer.start_as_current_span("generation"):
                    answer = await chain.ainvoke({})
                answer = ensure_grounded(answer, context_texts)
                
            return RAGResult(
                answer=answer,
                contexts=contexts,
            )

    async def stream(
        self,
        question: str,
        session_id: str | None = None,
    ):
        with QUERY_DURATION.time():
            QUERY_COUNTER.inc()
            
            with tracer.start_as_current_span("retrieval") as span:
                docs = await self.retriever.retrieve(question)
                span.set_attribute("docs_found", len(docs))
                
            if self.reranker and docs:
                with tracer.start_as_current_span("reranking"):
                    docs = self.reranker.rerank(question, docs, top_k=4)

            context_texts = []
            for d in docs:
                if hasattr(d, "chunk"):
                    context_texts.append(d.chunk.text)
                else:
                    context_texts.append(d.page_content)
            context_text = "\n\n".join(context_texts)
            
            if self.memory_store and session_id:
                history = await self.memory_store.get_history(session_id)
                history_text = "\n".join(f"{m.role}: {m.content}" for m in history)
                
                payload = {
                    "question": question,
                    "history": history_text,
                    "context": context_text,
                }
                prompt_template = CONVERSATIONAL_RAG_PROMPT
            else:
                payload = {
                    "question": question,
                    "context": context_text,
                }
                prompt_template = RAG_PROMPT
                
            chain = (
                RunnableLambda(lambda _: payload)
                | prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            full_answer = []
            with tracer.start_as_current_span("generation"):
                async for token in chain.astream({}):
                    full_answer.append(token)
                    yield token
                
            answer = "".join(full_answer)
            answer = ensure_grounded(answer, context_texts)
            
            if self.memory_store and session_id:
                await self.memory_store.append_message(
                    session_id,
                    ChatMessage(role="user", content=question, timestamp=datetime.utcnow())
                )
                await self.memory_store.append_message(
                    session_id,
                    ChatMessage(role="assistant", content=answer, timestamp=datetime.utcnow())
                )


