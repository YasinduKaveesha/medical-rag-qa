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

from app.schemas import AskRequest, AskResponse, CitationSource, HealthResponse
from src.config import get_settings
from src.generation.citations import extract_citations
from src.generation.prompt_builder import build_prompt
from src.generation.refusal import should_refuse

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.retrieval.pipeline import RetrievalPipeline
    from src.retrieval.vector_store import QdrantStore

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
