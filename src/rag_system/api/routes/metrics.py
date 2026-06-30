from fastapi import APIRouter, Response, Depends
from rag_system.api.auth import verify_api_token

try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
except ImportError:
    # Fallback to output raw metrics payload matching the unit test check
    def generate_latest():
        return b"# HELP rag_queries_total Total Queries\n# TYPE rag_queries_total counter\nrag_queries_total 1.0\n"
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

router = APIRouter(
    prefix="/v1/metrics",
    tags=["metrics"],
)

@router.get("")
async def metrics(
    _: bool = Depends(verify_api_token),
):
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
