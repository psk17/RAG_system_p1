from dataclasses import dataclass

@dataclass(frozen=True)
class RetrievedContext:
    chunk_id: str
    source: str
    page_number: int | None
    content: str
    score: float

@dataclass(frozen=True)
class RAGResult:
    answer: str
    contexts: list[RetrievedContext]
