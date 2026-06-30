from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_system.core.config.settings import get_settings

def configure_middleware(app: FastAPI):
    settings = get_settings()
    origins = settings.cors_origins
    if settings.is_production and "*" in origins:
        origins = [o for o in origins if o != "*"]
        
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
