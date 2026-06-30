from pydantic import BaseModel

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    chunks_created: int
    status: str
