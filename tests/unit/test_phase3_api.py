import pytest
import io
import fitz
from fastapi.testclient import TestClient
from rag_system.api.app import app
from rag_system.core.config.settings import get_settings
from rag_system.api.auth import verify_api_token

def make_valid_pdf_bytes():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This is mock policy content for testing.", fontsize=12)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

@pytest.fixture(autouse=True)
def setup_fake_services():
    # Override dependencies to use offline in-memory Chroma with fake embeddings
    from rag_system.api.dependencies import get_vector_store, get_ingestion_service
    from rag_system.ingestion.vector_store_chroma import ChromaAdapter
    from rag_system.ingestion.ingestion_service import IngestionService
    
    fake_store = ChromaAdapter.for_testing()
    fake_ingestion = IngestionService(fake_store)
    
    app.dependency_overrides[get_vector_store] = lambda: fake_store
    app.dependency_overrides[get_ingestion_service] = lambda: fake_ingestion
    yield
    app.dependency_overrides.clear()

@pytest.fixture
def client():
    settings = get_settings()
    settings.api_token = "test-token"
    with TestClient(app) as c:
        yield c

class TestPhase3Health:
    def test_health_check_get(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_check_post_method_not_allowed(self, client):
        response = client.post("/v1/health")
        assert response.status_code == 405

    def test_health_check_headers(self, client):
        response = client.get("/v1/health")
        assert "content-type" in response.headers
        assert "application/json" in response.headers["content-type"]

    def test_health_check_response_structure(self, client):
        response = client.get("/v1/health")
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 1
        assert "status" in data

class TestPhase3Auth:
    def test_no_auth_header(self, client):
        response = client.post("/v1/documents/upload")
        assert response.status_code == 401
        assert "Missing authorization header" in response.json()["detail"]

    def test_empty_auth_header(self, client):
        response = client.post("/v1/documents/upload", headers={"Authorization": ""})
        assert response.status_code == 401

    def test_wrong_auth_prefix(self, client):
        response = client.post("/v1/documents/upload", headers={"Authorization": "Basic dGVzdC10b2tlbg=="})
        assert response.status_code == 401
        assert "Invalid API token" in response.json()["detail"]

    def test_wrong_token(self, client):
        response = client.post("/v1/documents/upload", headers={"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401
        assert "Invalid API token" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_valid_token_structure(self, client):
        with pytest.raises(Exception):
            await verify_api_token(None)

    def test_token_casing(self, client):
        response = client.post("/v1/documents/upload", headers={"Authorization": "bearer test-token"})
        assert response.status_code == 401

class TestPhase3Upload:
    def test_upload_pdf(self, client):
        files = {"file": ("test.pdf", make_valid_pdf_bytes(), "application/pdf")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.pdf"
        assert data["status"] == "processed"
        assert "document_id" in data

    def test_upload_md(self, client):
        files = {"file": ("doc.md", b"# Header\nMarkdown content.", "text/markdown")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "doc.md"

    def test_upload_markdown(self, client):
        files = {"file": ("doc.markdown", b"# Header\nMarkdown content.", "text/markdown")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "doc.markdown"

    def test_upload_txt(self, client):
        files = {"file": ("doc.txt", b"Plain text content.", "text/plain")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "doc.txt"

    def test_upload_text(self, client):
        files = {"file": ("doc.text", b"Plain text content.", "text/plain")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "doc.text"

    def test_upload_unsupported_png(self, client):
        files = {"file": ("img.png", b"fake png", "image/png")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_unsupported_html(self, client):
        files = {"file": ("doc.html", b"<html></html>", "text/html")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 400

    def test_upload_no_file(self, client):
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 422

    def test_upload_empty_filename(self, client):
        files = {"file": ("", b"", "text/plain")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code in (400, 422)

    def test_upload_case_insensitivity(self, client):
        files = {"file": ("TEST.PDF", make_valid_pdf_bytes(), "application/pdf")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "TEST.PDF"

    def test_upload_large_file_name(self, client):
        long_name = "a" * 100 + ".txt"
        files = {"file": (long_name, b"Plain text content.", "text/plain")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200
        assert response.json()["filename"] == long_name

    def test_upload_spaces_in_filename(self, client):
        files = {"file": ("spaced file name.txt", b"Plain text content.", "text/plain")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "spaced file name.txt"

    def test_upload_weird_characters_in_filename(self, client):
        files = {"file": ("file-@#$_&-+()~'=%!;.txt", b"Plain text content.", "text/plain")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200

    def test_upload_empty_file_content(self, client):
        files = {"file": ("empty.txt", b"", "text/plain")}
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer test-token"},
            files=files
        )
        assert response.status_code == 200

    def test_upload_no_headers_raises_401(self, client):
        files = {"file": ("test.txt", b"content", "text/plain")}
        response = client.post("/v1/documents/upload", files=files)
        assert response.status_code == 401

    def test_health_check_query_params_ignored(self, client):
        response = client.get("/v1/health?param=value")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_health_check_case_sensitivity(self, client):
        response = client.get("/V1/HEALTH")
        assert response.status_code == 404

    def test_auth_special_chars_token_rejected(self, client):
        response = client.post(
            "/v1/documents/upload",
            headers={"Authorization": "Bearer token-with-special-chars-@#$"}
        )
        assert response.status_code == 401

    def test_upload_size_limit_exceeded(self, client):
        settings = get_settings()
        old_val = settings.max_upload_mb
        settings.max_upload_mb = 1  # 1MB
        try:
            large_content = b"a" * (2 * 1024 * 1024)
            files = {"file": ("large.txt", large_content, "text/plain")}
            response = client.post(
                "/v1/documents/upload",
                headers={"Authorization": "Bearer test-token"},
                files=files
            )
            assert response.status_code == 413
            assert "exceeds maximum upload limit" in response.json()["detail"]
        finally:
            settings.max_upload_mb = old_val

    def test_query_endpoint_auth_enforced(self, client):
        response = client.post("/v1/query", json={"question": "hello"})
        assert response.status_code == 401

    def test_query_stream_endpoint_auth_enforced(self, client):
        response = client.post("/v1/query/stream", json={"question": "hello"})
        assert response.status_code == 401

    def test_sessions_endpoint_auth_enforced(self, client):
        response = client.post("/v1/sessions")
        assert response.status_code == 401

    def test_metrics_endpoint_auth_enforced(self, client):
        response = client.get("/v1/metrics")
        assert response.status_code == 401

    def test_cors_production_no_wildcard(self):
        from rag_system.api.middleware import configure_middleware
        from fastapi import FastAPI
        from rag_system.core.config.settings import get_settings, AppEnv
        
        test_app = FastAPI()
        settings = get_settings()
        
        old_env = settings.app_env
        old_origins = settings.cors_origins
        
        settings.app_env = AppEnv.PRODUCTION
        settings.cors_origins = ["*", "http://trusted.com"]
        
        try:
            configure_middleware(test_app)
            cors_middleware = None
            for m in test_app.user_middleware:
                if m.cls.__name__ == "CORSMiddleware":
                    cors_middleware = m
                    break
            
            assert cors_middleware is not None
            assert "*" not in cors_middleware.kwargs["allow_origins"]
            assert "http://trusted.com" in cors_middleware.kwargs["allow_origins"]
        finally:
            settings.app_env = old_env
            settings.cors_origins = old_origins


