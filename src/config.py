"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Typed settings for the Medical RAG Q&A system.

    All values are read from environment variables (via .env).
    Use :func:`get_settings` to obtain the singleton instance.
    """

    groq_api_key: str
    llm_base_url: str = "https://api.groq.com/openai/v1"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "medical_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "llama-3.3-70b-versatile"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    similarity_threshold: float = 0.35
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    # Phase 2 — Multimodal
    clip_collection_name: str = "multimodal_clip"
    clip_model: str = "openai/clip-vit-base-patch32"
    caption_model: str = "Salesforce/blip-image-captioning-base"
    vision_llm_model: str = "llama-3.2-11b-vision-preview"
    rrf_k: int = 60
    device: str = "cpu"
    extracted_images_dir: str = "data/extracted_images"


def setup_logging() -> None:
    """Configure the root logger at INFO level with a standard timestamp format."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton :class:`Settings` instance.

    On the first call, loads ``.env`` from the current working directory,
    configures logging, and constructs the :class:`Settings` dataclass from
    environment variables.  Subsequent calls return the cached instance.

    Raises:
        ValueError: If ``GROQ_API_KEY`` is not set.
    """
    global _settings
    if _settings is None:
        load_dotenv()
        setup_logging()
        logger.info("Loading application settings from environment")

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Copy .env.example to .env and fill it in.")

        # Phase 2 — DEVICE auto-detection
        _device_env = os.getenv("DEVICE", "cpu")
        if _device_env != "cpu":
            try:
                import torch  # noqa: PLC0415

                _device_env = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                _device_env = "cpu"

        _settings = Settings(
            groq_api_key=api_key,
            llm_base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            collection_name=os.getenv("COLLECTION_NAME", "medical_docs"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            reranker_model=os.getenv(
                "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.35")),
            top_k_retrieval=int(os.getenv("TOP_K_RETRIEVAL", "20")),
            top_k_rerank=int(os.getenv("TOP_K_RERANK", "5")),
            # Phase 2 — Multimodal
            clip_collection_name=os.getenv("CLIP_COLLECTION_NAME", "multimodal_clip"),
            clip_model=os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32"),
            caption_model=os.getenv(
                "CAPTION_MODEL", "Salesforce/blip-image-captioning-base"
            ),
            vision_llm_model=os.getenv("VISION_LLM_MODEL", "llama-3.2-11b-vision-preview"),
            rrf_k=int(os.getenv("RRF_K", "60")),
            device=_device_env,
            extracted_images_dir=os.getenv("EXTRACTED_IMAGES_DIR", "data/extracted_images"),
        )
        logger.info(
            "Settings loaded: llm=%s, embedding=%s", _settings.llm_model, _settings.embedding_model
        )
    return _settings
