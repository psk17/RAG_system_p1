from fastapi import APIRouter, Depends
from rag_system.memory.session_manager import SessionManager
from rag_system.api.auth import verify_api_token

router = APIRouter(
    prefix="/v1/sessions",
    tags=["sessions"],
)

@router.post("")
async def create_session(
    _: bool = Depends(verify_api_token),
):
    return {
        "session_id": SessionManager.create_session()
    }
