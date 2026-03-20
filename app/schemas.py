"""Pydantic request/response models for the Medical RAG Q&A FastAPI app.

All models are used directly as FastAPI endpoint parameter and return types,
which gives automatic JSON validation, serialization, and OpenAPI schema
generation for free.

Models
------
- :class:`AskRequest`     — POST /ask request body
- :class:`CitationSource` — one cited source within an :class:`AskResponse`
- :class:`AskResponse`    — POST /ask response body
- :class:`HealthResponse` — GET /health response body
"""

from __future__ import annotations

from pydantic import BaseModel


class AskRequest(BaseModel):
    """Request body for the POST /ask endpoint.

    Attributes:
        question: Natural-language medical question from the user.
            Must not be empty or whitespace-only (validated in the endpoint).
        filters: Optional metadata pre-filter dict forwarded to the retrieval
            pipeline.  Supported keys: ``"document_type"``, ``"section_title"``.
            Pass ``null`` / omit to search the full collection.
    """

    question: str
    filters: dict | None = None


class CitationSource(BaseModel):
    """One source citation extracted from the LLM answer.

    Each instance corresponds to a single ``[N]`` marker in the answer text
    resolved back to its originating chunk by
    :func:`src.generation.citations.extract_citations`.

    Attributes:
        claim:           Sentence from the answer that contains the citation.
        source_chunk:    Body text of the referenced chunk.
        page_number:     Page number in the source document (may be ``None``
                         if the ingested chunk lacked page metadata).
        source_document: Filename of the source PDF.
    """

    claim: str
    source_chunk: str
    page_number: int | None
    source_document: str


class AskResponse(BaseModel):
    """Response body for the POST /ask endpoint.

    Attributes:
        answer:        LLM-generated answer string.  Will be
                       ``"I cannot answer from the provided documents."`` when
                       retrieval quality is below the similarity threshold.
        sources:       List of citation dicts mapping answer sentences to their
                       source chunks.  Empty when the answer is a refusal.
        confidence:    Maximum cosine similarity score across all retrieved
                       chunks.  ``0.0`` when no chunks were retrieved.
        model_version: LLM model identifier used for generation (from
                       ``Settings.llm_model``).
    """

    answer: str
    sources: list[CitationSource]
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    """Response body for the GET /health endpoint.

    Attributes:
        status:          ``"ok"`` when all dependencies are reachable;
                         ``"degraded"`` when Qdrant is unavailable.
        model_version:   LLM model identifier (from ``Settings.llm_model``).
        collection_info: Dict with Qdrant collection stats:
                         ``vectors_count``, ``points_count``, ``status``.
                         Contains ``{"error": "Qdrant unavailable"}`` when the
                         store cannot be reached.
    """

    status: str
    model_version: str
    collection_info: dict
