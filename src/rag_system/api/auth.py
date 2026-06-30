from fastapi import Header, HTTPException, status
from rag_system.core.config.settings import get_settings

async def verify_api_token(
    authorization: str | None = Header(None),
):
    settings = get_settings()

    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
        )

    expected = f"Bearer {settings.api_token}"

    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
        )

    return True
