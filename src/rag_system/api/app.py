import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from rag_system.api.dependencies import initialize_services
from rag_system.api.middleware import configure_middleware
from rag_system.api.routes.documents import router as document_router
from rag_system.api.routes.health import router as health_router
from rag_system.api.routes.metrics import router as metrics_router
from rag_system.api.routes.query import router as query_router
from rag_system.api.routes.sessions import router as sessions_router
from rag_system.api.routes.streaming_query import router as streaming_router

logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_services()
    yield

app = FastAPI(
    title="RAG System",
    version="1.0.0",
    lifespan=lifespan,
)

configure_middleware(app)

@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def get_frontend():
    static_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_file):
        with open(static_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>RAG Portal frontend missing.</h1>"

app.include_router(health_router)
app.include_router(document_router)
app.include_router(query_router)
app.include_router(sessions_router)
app.include_router(streaming_router)
app.include_router(metrics_router)


