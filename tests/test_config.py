"""Smoke tests for src.config — Settings loading and singleton behaviour."""

from __future__ import annotations

import importlib
from unittest.mock import patch


def _reload_config():
    """Reload the config module and reset the singleton so each test is isolated."""
    import src.config as cfg

    cfg._settings = None
    importlib.reload(cfg)
    cfg._settings = None
    return cfg


def test_settings_loads_from_env(tmp_path, monkeypatch):
    """Settings is populated correctly when all env vars are present."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("QDRANT_HOST", "qdrant-server")
    monkeypatch.setenv("QDRANT_PORT", "6334")
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("LLM_MODEL", "llama-3.3-70b-versatile")
    monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.40")
    monkeypatch.setenv("TOP_K_RETRIEVAL", "15")
    monkeypatch.setenv("TOP_K_RERANK", "3")

    import src.config as cfg

    cfg._settings = None

    settings = cfg.get_settings()

    assert settings.groq_api_key == "gsk-test-key"
    assert settings.llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.qdrant_host == "qdrant-server"
    assert settings.qdrant_port == 6334
    assert settings.collection_name == "test_collection"
    assert settings.similarity_threshold == 0.40
    assert settings.top_k_retrieval == 15
    assert settings.top_k_rerank == 3

    cfg._settings = None  # cleanup


def test_settings_defaults(monkeypatch):
    """Optional env vars fall back to sensible defaults."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    for var in [
        "LLM_BASE_URL", "QDRANT_HOST", "QDRANT_PORT", "COLLECTION_NAME",
        "EMBEDDING_MODEL", "LLM_MODEL", "RERANKER_MODEL",
        "SIMILARITY_THRESHOLD", "TOP_K_RETRIEVAL", "TOP_K_RERANK",
    ]:
        monkeypatch.delenv(var, raising=False)

    import src.config as cfg

    cfg._settings = None

    settings = cfg.get_settings()

    assert settings.llm_base_url == "https://api.groq.com/openai/v1"
    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.collection_name == "medical_docs"
    assert settings.similarity_threshold == 0.35
    assert settings.top_k_retrieval == 20
    assert settings.top_k_rerank == 5

    cfg._settings = None  # cleanup


def test_get_settings_is_singleton(monkeypatch):
    """get_settings() returns the same object on repeated calls."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")

    import src.config as cfg

    cfg._settings = None

    s1 = cfg.get_settings()
    s2 = cfg.get_settings()

    assert s1 is s2

    cfg._settings = None  # cleanup


def test_missing_api_key_raises(monkeypatch):
    """get_settings() raises ValueError when GROQ_API_KEY is absent."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    import src.config as cfg

    cfg._settings = None

    # Patch load_dotenv so it doesn't reload the key from .env on disk
    with patch("src.config.load_dotenv"):
        try:
            cfg.get_settings()
            assert False, "Expected ValueError"
        except ValueError as exc:
            assert "GROQ_API_KEY" in str(exc)
        finally:
            cfg._settings = None  # cleanup


def test_setup_logging_runs_without_error():
    """setup_logging() can be called without raising exceptions."""
    from src.config import setup_logging

    setup_logging()  # idempotent — calling twice is fine
    setup_logging()


# ---------------------------------------------------------------------------
# Phase 2 — Multimodal config tests
# ---------------------------------------------------------------------------


def test_config_has_clip_collection_name(monkeypatch):
    """Settings includes clip_collection_name defaulting to 'multimodal_clip'."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    import src.config as cfg

    cfg._settings = None
    settings = cfg.get_settings()
    assert settings.clip_collection_name == "multimodal_clip"
    cfg._settings = None


def test_config_has_caption_model(monkeypatch):
    """Settings includes caption_model containing 'blip'."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    import src.config as cfg

    cfg._settings = None
    settings = cfg.get_settings()
    assert "blip" in settings.caption_model.lower()
    cfg._settings = None


def test_config_has_device_field(monkeypatch):
    """Settings includes device field as a string."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test-key")
    import src.config as cfg

    cfg._settings = None
    settings = cfg.get_settings()
    assert isinstance(settings.device, str)
    assert settings.device in ("cpu", "cuda")
    cfg._settings = None
