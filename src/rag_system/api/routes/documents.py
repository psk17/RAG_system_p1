from pathlib import Path
import tempfile
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException

from rag_system.api.auth import verify_api_token
from rag_system.api.schemas.upload import UploadResponse
from rag_system.api.dependencies import get_ingestion_service, get_vector_store
from rag_system.ingestion.ingestion_service import IngestionService
from rag_system.ingestion.chunking_service import ChunkingService
from rag_system.core.config.settings import get_settings

router = APIRouter(
    prefix="/v1/documents",
    tags=["documents"],
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".md",
    ".markdown",
    ".txt",
    ".text",
}

@router.post(
    "/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    _: bool = Depends(verify_api_token),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    extension = Path(file.filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type",
        )

    content = await file.read()
    settings = get_settings()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum upload limit of {settings.max_upload_mb} MB",
        )
    
    # Create temp directory to preserve original filename for chunking metadata
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename
    
    try:
        temp_path.write_bytes(content)
        result = await ingestion_service.ingest_file(
            path=temp_path,
            collection_id="default"
        )
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Ingestion failed: {result.errors}",
            )
        
        document_id = ChunkingService._make_document_id(temp_path)

        return UploadResponse(
            document_id=document_id,
            filename=file.filename,
            chunks_created=result.chunks_processed,
            status="processed",
        )
    finally:
        if temp_path.exists():
            temp_path.unlink()
        try:
            Path(temp_dir).rmdir()
        except Exception:
            pass

@router.get(
    "",
    response_model=list[dict],
)
async def list_documents(
    vector_store = Depends(get_vector_store),
    _: bool = Depends(verify_api_token),
):
    try:
        # Check if vector store has client attribute
        if not hasattr(vector_store, "_client"):
            return []
            
        client = vector_store._client
        col = client.get_or_create_collection("default")
        data = col.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        
        unique_docs = {}
        for m in metadatas:
            if not m:
                continue
            doc_id = m.get("document_id")
            if doc_id and doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    "document_id": doc_id,
                    "source_file": m.get("source_file", "unknown"),
                    "chunks": 0
                }
            if doc_id:
                unique_docs[doc_id]["chunks"] += 1
                
        return list(unique_docs.values())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete(
    "/{document_id}",
)
async def delete_document(
    document_id: str,
    vector_store = Depends(get_vector_store),
    _: bool = Depends(verify_api_token),
):
    try:
        deleted = await vector_store.delete_document(
            document_id=document_id,
            collection_name="default"
        )
        return {"status": "success", "deleted_chunks": deleted}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

