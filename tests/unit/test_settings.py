"""
tests/unit/test_settings.py
─────────────────────────────
Validates that Settings loads correctly and that field validators fire.
No I/O or network calls — pure unit tests.
"""

import pytest
from pydantic import ValidationError

from rag_system.core.config.settings import Settings, AppEnv, EmbeddingProvider, VectorStoreProvider


class TestSettingsDefaults:
    def test_defaults_load_without_env_file(self, monkeypatch):
        """Settings with no .env file should fall back to coded defaults."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        s = Settings(_env_file=None)
        assert s.app_env == AppEnv.DEVELOPMENT
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200
        assert s.top_k == 4

    def test_max_upload_bytes_derived_correctly(self):
        s = Settings(_env_file=None)
        assert s.max_upload_bytes == s.max_upload_mb * 1024 * 1024

    def test_is_production_false_by_default(self):
        s = Settings(_env_file=None)
        assert s.is_production is False


class TestSettingsValidators:
    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        with pytest.raises(ValidationError, match="chunk_overlap"):
            Settings(_env_file=None, chunk_size=500, chunk_overlap=500)

    def test_max_upload_mb_must_be_positive(self):
        with pytest.raises(ValidationError, match="max_upload_mb"):
            Settings(_env_file=None, max_upload_mb=0)

    def test_valid_custom_values_accepted(self):
        s = Settings(_env_file=None, chunk_size=2000, chunk_overlap=400, top_k=8)
        assert s.chunk_size == 2000
        assert s.chunk_overlap == 400
        assert s.top_k == 8


class TestSettingsEnvOverride:
    def test_env_variable_overrides_default(self, monkeypatch):
        monkeypatch.setenv("TOP_K", "10")
        monkeypatch.setenv("CHUNK_SIZE", "1500")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        s = Settings(_env_file=None)
        assert s.top_k == 10
        assert s.chunk_size == 1500

    def test_embedding_provider_enum_parsed(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
        s = Settings(_env_file=None)
        assert s.embedding_provider == EmbeddingProvider.HUGGINGFACE

    def test_vector_store_provider_enum_parsed(self, monkeypatch):
        monkeypatch.setenv("VECTOR_STORE_PROVIDER", "qdrant")
        s = Settings(_env_file=None)
        assert s.vector_store_provider == VectorStoreProvider.QDRANT
