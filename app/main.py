"""FastAPI application for the Medical RAG Q&A system.

Exposes two endpoints:

- ``POST /ask``    — retrieve relevant chunks, generate a cited answer, and
  return it with source citations and a confidence score.
- ``GET  /health`` — return service status and Qdrant collection statistics.

All ML dependencies (retrieval pipeline, LLM client, vector store) are
injected via FastAPI's :func:`fastapi.Depends` system so that tests can
substitute lightweight mocks without loading any real models or requiring
running services.

Typical usage
-------------
::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import Depends, FastAPI, HTTPException

from app.schemas import (
    AskRequest,
    AskResponse,
    CitationSource,
    HealthResponse,
    ImageResult,
    MultiModalAskRequest,
    MultiModalAskResponse,
)
from src.config import get_settings
from src.generation.citations import extract_citations, extract_multimodal_citations
from src.generation.prompt_builder import build_multimodal_prompt, build_prompt
from src.generation.refusal import should_refuse

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.retrieval.multimodal_pipeline import MultiModalRetrievalPipeline
    from src.retrieval.pipeline import RetrievalPipeline
    from src.retrieval.vector_store import MultiModalVectorStore, QdrantStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical RAG Q&A",
    description="Retrieval-augmented generation over clinical PDF documents.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Dependency provider functions
# ---------------------------------------------------------------------------
# Thin wrappers around module-level singletons.  Defined at module scope so
# tests can override them with app.dependency_overrides[_get_pipeline] = ...


def _get_pipeline() -> RetrievalPipeline:
    """Return the process-level RetrievalPipeline singleton."""
    from src.retrieval.pipeline import get_pipeline  # noqa: PLC0415

    return get_pipeline()


def _get_llm_client() -> LLMClient:
    """Return the process-level LLMClient singleton."""
    from src.generation.llm_client import get_llm_client  # noqa: PLC0415

    return get_llm_client()


def _get_store() -> QdrantStore:
    """Return the process-level QdrantStore singleton."""
    from src.retrieval.vector_store import get_store  # noqa: PLC0415

    return get_store()


def _get_multimodal_pipeline() -> MultiModalRetrievalPipeline:
    """Return the process-level MultiModalRetrievalPipeline singleton."""
    from src.retrieval.multimodal_pipeline import get_multimodal_pipeline  # noqa: PLC0415

    return get_multimodal_pipeline()


def _get_multimodal_store() -> MultiModalVectorStore:
    """Return the process-level MultiModalVectorStore singleton."""
    from src.retrieval.vector_store import get_multimodal_store  # noqa: PLC0415

    return get_multimodal_store()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    pipeline: RetrievalPipeline = Depends(_get_pipeline),
    llm_client: LLMClient = Depends(_get_llm_client),
) -> AskResponse:
    """Answer a medical question using retrieved document context.

    Orchestrates the full RAG pipeline:

    1. Validate that the question is non-empty.
    2. Retrieve the top-K most relevant chunks from Qdrant.
    3. If retrieval quality is too low (:func:`src.generation.refusal.should_refuse`),
       return a structured refusal without calling the LLM.
    4. Build a prompt, generate an answer, and extract citation markers.
    5. Return the answer, sources, confidence score, and model version.

    Args:
        request: Validated :class:`app.schemas.AskRequest` body.
        pipeline: Injected :class:`src.retrieval.pipeline.RetrievalPipeline`.
        llm_client: Injected :class:`src.generation.llm_client.LLMClient`.

    Returns:
        :class:`app.schemas.AskResponse` with ``answer``, ``sources``,
        ``confidence``, and ``model_version``.

    Raises:
        HTTPException 400: When the question is empty or whitespace-only.
        HTTPException 503: When Qdrant is unreachable.
        HTTPException 500: On any other unexpected internal error.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    logger.info(
        "POST /ask  question_len=%d  filters=%s", len(question), request.filters
    )

    s = get_settings()

    try:
        chunks = pipeline.retrieve(question, top_k=s.top_k_rerank, filters=request.filters)

        if should_refuse(chunks):
            confidence = max((c["score"] for c in chunks), default=0.0)
            logger.info(
                "POST /ask  refusal  confidence=%.4f  question_len=%d",
                confidence,
                len(question),
            )
            return AskResponse(
                answer="I cannot answer from the provided documents.",
                sources=[],
                confidence=confidence,
                model_version=s.llm_model,
            )

        prompt = build_prompt(question, chunks)
        answer = llm_client.generate(prompt)
        raw_citations = extract_citations(answer, chunks)
        sources = [CitationSource(**c) for c in raw_citations]
        confidence = max(c["score"] for c in chunks)

    except ConnectionError as exc:
        logger.error("POST /ask  ConnectionError: %s", exc)
        raise HTTPException(status_code=503, detail="Vector store unavailable.") from exc
    except Exception as exc:
        logger.exception("POST /ask  unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    logger.info(
        "POST /ask  answer_len=%d  sources=%d  confidence=%.4f",
        len(answer),
        len(sources),
        confidence,
    )
    return AskResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        model_version=s.llm_model,
    )


@app.get("/health", response_model=HealthResponse)
def health(store: QdrantStore = Depends(_get_store)) -> HealthResponse:
    """Return service health and Qdrant collection statistics.

    Always returns HTTP 200.  When Qdrant is unreachable the ``status`` field
    is ``"degraded"`` and ``collection_info`` contains an ``"error"`` key so
    callers and load balancers can detect the degraded state without relying on
    the HTTP status code.

    Args:
        store: Injected :class:`src.retrieval.vector_store.QdrantStore`.

    Returns:
        :class:`app.schemas.HealthResponse` with ``status``, ``model_version``,
        and ``collection_info``.
    """
    s = get_settings()

    try:
        collection_info = store.get_collection_info()
        status = "ok"
    except ConnectionError as exc:
        logger.warning("GET /health  Qdrant unavailable: %s", exc)
        collection_info = {"error": "Qdrant unavailable"}
        status = "degraded"

    logger.info("GET /health  status=%s", status)
    return HealthResponse(
        status=status,
        model_version=s.llm_model,
        collection_info=collection_info,
    )


# ---------------------------------------------------------------------------
# Phase 2 endpoints
# ---------------------------------------------------------------------------


@app.post("/ask-multimodal", response_model=MultiModalAskResponse)
def ask_multimodal(
    request: MultiModalAskRequest,
    mm_pipeline: MultiModalRetrievalPipeline = Depends(_get_multimodal_pipeline),
    llm_client: LLMClient = Depends(_get_llm_client),
) -> MultiModalAskResponse:
    """Answer a medical question using multimodal (text + image) retrieval.

    Retrieves text chunks and images via the dual-encoder pipeline, builds a
    multimodal prompt, and generates an answer.  When ``use_vision`` is
    ``True`` and images were retrieved, the vision LLM is called with the
    image files attached; otherwise the text-only ``generate()`` is used.

    Args:
        request: Validated :class:`app.schemas.MultiModalAskRequest` body.
        mm_pipeline: Injected :class:`~src.retrieval.multimodal_pipeline.MultiModalRetrievalPipeline`.
        llm_client: Injected :class:`~src.generation.llm_client.LLMClient`.

    Returns:
        :class:`app.schemas.MultiModalAskResponse`.

    Raises:
        HTTPException 400: When the question is empty or whitespace-only.
        HTTPException 503: When Qdrant is unreachable.
        HTTPException 500: On any other unexpected internal error.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    logger.info("POST /ask-multimodal  question_len=%d  use_vision=%s", len(question), request.use_vision)
    s = get_settings()

    try:
        result = mm_pipeline.retrieve(question, top_k=request.top_k)

        text_chunks = result.text_chunks if not should_refuse(result.text_chunks) else []
        images = result.images if request.include_images else []

        prompt = build_multimodal_prompt(question, text_chunks, images)

        used_vision = False
        if request.use_vision and images:
            image_paths = [img.get("image_path", "") for img in images if img.get("image_path")]
            answer = llm_client.generate_with_vision(prompt, image_paths)
            used_vision = True
        else:
            answer = llm_client.generate(prompt)

        raw_citations = extract_multimodal_citations(answer, text_chunks, images)

        image_sources: list[ImageResult] = []
        for img in images:
            image_sources.append(
                ImageResult(
                    image_id=img.get("image_id", ""),
                    image_path=img.get("image_path", ""),
                    caption=img.get("caption", img.get("text", "")),
                    source_pdf=img.get("source_pdf", img.get("source_document", "")),
                    page_number=img.get("page_number", 0),
                    relevance_score=img.get("rrf_score", img.get("score", 0.0)),
                )
            )

    except ConnectionError as exc:
        logger.error("POST /ask-multimodal  ConnectionError: %s", exc)
        raise HTTPException(status_code=503, detail="Vector store unavailable.") from exc
    except Exception as exc:
        logger.exception("POST /ask-multimodal  unexpected error: %s", exc)
        raise HTTPException(status_code=500, detail="Internal server error.") from exc

    logger.info(
        "POST /ask-multimodal  answer_len=%d  text_sources=%d  images=%d",
        len(answer),
        len(raw_citations),
        len(image_sources),
    )
    return MultiModalAskResponse(
        answer=answer,
        text_sources=raw_citations,
        image_sources=image_sources,
        used_vision_model=used_vision,
        retrieval_time_ms=result.retrieval_time_ms,
        model_version=s.llm_model,
    )


@app.get("/images/{image_id}")
def get_image(image_id: str) -> object:
    """Serve an extracted image file by its image_id.

    Looks up ``{image_id}.png`` inside the configured
    ``Settings.extracted_images_dir`` directory.

    Args:
        image_id: Unique image identifier (``{pdf_stem}_p{page}_x{xref}``).

    Returns:
        :class:`starlette.responses.FileResponse` with content-type
        ``image/png``.

    Raises:
        HTTPException 404: When the image file does not exist.
    """
    import os  # noqa: PLC0415

    from starlette.responses import FileResponse  # noqa: PLC0415

    images_dir = get_settings().extracted_images_dir
    img_path = os.path.join(images_dir, f"{image_id}.png")

    if not os.path.isfile(img_path):
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found.")

    return FileResponse(img_path, media_type="image/png")
