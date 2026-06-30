from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None

class SourceChunk(BaseModel):
    chunk_id: str
    source: str
    page_number: int | None
    score: float
    content: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
