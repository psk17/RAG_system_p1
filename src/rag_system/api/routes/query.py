from fastapi import APIRouter, Depends
from rag_system.api.schemas.query import QueryRequest, QueryResponse, SourceChunk
from rag_system.api.dependencies import get_rag_manager
from rag_system.api.auth import verify_api_token

router = APIRouter(
    prefix="/v1/query",
    tags=["query"],
)

@router.post(
    "",
    response_model=QueryResponse,
)
async def query_documents(
    request: QueryRequest,
    rag_manager=Depends(get_rag_manager),
    _: bool = Depends(verify_api_token),
):
    result = await rag_manager.query(
        question=request.question,
        session_id=request.session_id,
    )

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceChunk(
                chunk_id=s.chunk_id,
                source=s.source,
                page_number=s.page_number,
                score=s.score,
                content=s.content,
            )
            for s in result.contexts
        ],
    )
