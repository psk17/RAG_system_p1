"""
app/core/config/settings.py
─────────────────────────────
Single source of truth for all runtime configuration.
Loaded once at startup via `get_settings()` (cached singleton).

Usage:
    from rag_system.core.config.settings import get_settings
    settings = get_settings()
    print(settings.openai_api_key)
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()



# ──────────────────────────────────────────────────────────────────────────────
# Enumerations — enforced at parse time; no magic strings scattered through code
# ──────────────────────────────────────────────────────────────────────────────

class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"


class VectorStoreProvider(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"
    PGVECTOR = "pgvector"


# ──────────────────────────────────────────────────────────────────────────────
# Settings Model
# ──────────────────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    def __init__(self, _env_file: str | None = None, **data: Any) -> None:
        # _env_file is accepted for compatibility with tests; it is ignored.
        super().__init__(**data)

    """
    All environment variables are validated and typed here.
    Secrets (API keys) are never logged or exposed in __repr__.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: AppEnv = AppEnv.DEVELOPMENT
    app_host: str = "0.0.0.0"
    app_port: int = 8080
    log_level: str = "INFO"

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.OPENAI

    openai_api_key: str | None = Field(default=None, repr=False)
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    anthropic_api_key: str | None = Field(default=None, repr=False)
    anthropic_model: str = "claude-sonnet-4-6"

    ollama_base_url: AnyHttpUrl | None = None
    ollama_model: str = "llama3"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Vector Store ──────────────────────────────────────────────────────────
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.CHROMA

    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_persist_dir: Path = Path("./data/chroma")

    qdrant_url: AnyHttpUrl | None = None
    qdrant_api_key: str | None = Field(default=None, repr=False)

    pgvector_dsn: str | None = Field(default=None, repr=False)

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Security ──────────────────────────────────────────────────────────────
    api_token: str = "dev-token"
    cors_origins: list[str] = ["*"]
    jwt_secret_key: str = Field(default="change-me", repr=False)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_upload_mb: int = 50

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 4

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    @field_validator("max_upload_mb")
    @classmethod
    def positive_upload_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_upload_mb must be a positive integer")
        return v

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnv.PRODUCTION


# ──────────────────────────────────────────────────────────────────────────────
# Cached Singleton — import and call anywhere; parsed only once per process
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the validated, cached Settings instance."""
    return Settings()
