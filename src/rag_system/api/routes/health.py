from fastapi import APIRouter
from rag_system.api.schemas.health import HealthResponse

router = APIRouter(
    prefix="/v1/health",
    tags=["health"],
)

@router.get(
    "",
    response_model=HealthResponse,
)
async def health():
    return HealthResponse(
        status="healthy",
    )
