from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from rag_system.api.schemas.query import QueryRequest
from rag_system.api.dependencies import get_rag_manager
from rag_system.api.auth import verify_api_token

router = APIRouter(
    prefix="/v1/query",
    tags=["streaming"],
)

@router.post("/stream")
async def stream_query(
    request: QueryRequest,
    rag_manager = Depends(get_rag_manager),
    _: bool = Depends(verify_api_token),
):
    async def event_generator():
        async for token in rag_manager.stream(
            question=request.question,
            session_id=request.session_id,
        ):
            yield f"data:{token}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )
