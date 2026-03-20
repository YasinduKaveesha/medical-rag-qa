"""End-to-end retrieval pipeline for the Medical RAG Q&A system.

Orchestrates three steps in sequence:

1. **Embed** — encode the user query with :class:`src.embeddings.encoder.EmbeddingEncoder`.
2. **Search** — retrieve the top-K candidates from Qdrant via
   :class:`src.retrieval.vector_store.QdrantStore`.
3. **Rerank** — re-score and reorder with
   :class:`src.retrieval.reranker.CrossEncoderReranker`.

Typical usage
-------------
::

    from src.retrieval.pipeline import get_pipeline

    pipeline = get_pipeline()
    results = pipeline.retrieve("What is the dose of amitriptyline?", top_k=5)
    # results[i]["chunk_text"]     — the chunk body text
    # results[i]["metadata"]       — source_document, section_title, etc.
    # results[i]["score"]          — cosine similarity from Qdrant
    # results[i]["reranker_score"] — cross-encoder logit
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from src.config import get_settings

if TYPE_CHECKING:
    import numpy as np

    from src.embeddings.encoder import EmbeddingEncoder
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)


class RetrievalPipeline:
    """Orchestrates query embedding → vector search → reranking.

    All three ML components are injected via constructor so that tests can
    substitute lightweight mocks without loading any real models or requiring
    a running Qdrant instance.

    Args:
        encoder: :class:`src.embeddings.encoder.EmbeddingEncoder` instance.
            When ``None``, the process-level singleton from
            :func:`src.embeddings.encoder.get_encoder` is used.
        store: :class:`src.retrieval.vector_store.QdrantStore` instance.
            When ``None``, the process-level singleton from
            :func:`src.retrieval.vector_store.get_store` is used.
        reranker: :class:`src.retrieval.reranker.CrossEncoderReranker` instance.
            When ``None``, the process-level singleton from
            :func:`src.retrieval.reranker.get_reranker` is used.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        store: QdrantStore | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if encoder is not None:
            self._encoder = encoder
        else:
            from src.embeddings.encoder import get_encoder  # noqa: PLC0415

            self._encoder = get_encoder()

        if store is not None:
            self._store = store
        else:
            from src.retrieval.vector_store import get_store  # noqa: PLC0415

            self._store = get_store()

        if reranker is not None:
            self._reranker = reranker
        else:
            from src.retrieval.reranker import get_reranker  # noqa: PLC0415

            self._reranker = get_reranker()

        logger.debug("RetrievalPipeline initialised with injected components")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for *query*.

        Executes three steps in sequence — embed, search, rerank — and
        reshapes the internal representation into a clean public format.
        Retrieval latency for each step and the total wall-clock time are
        logged at INFO level.

        Args:
            query: Raw natural-language question string.
            top_k: Number of chunks to return after reranking.  Defaults to
                ``5``.  The vector search always fetches
                ``Settings.top_k_retrieval`` (default 20) candidates so the
                reranker has a wide pool to work with.
            filters: Optional metadata pre-filter dict forwarded to
                :meth:`src.retrieval.vector_store.QdrantStore.search`.
                Supported keys: ``"document_type"``, ``"section_title"``.
                Pass ``None`` to search the full collection.

        Returns:
            List of dicts, at most *top_k* entries, sorted by
            ``"reranker_score"`` descending.  Each dict contains:

            - ``"chunk_text"`` — body text of the retrieved chunk.
            - ``"metadata"`` — dict of all metadata fields
              (``source_document``, ``document_type``, ``section_title``,
              ``page_number``, ``chunk_id``, ``chunk_index``,
              ``chunking_strategy``).  Does **not** contain ``"text"``.
            - ``"score"`` — cosine similarity from Qdrant (``float``).
            - ``"reranker_score"`` — raw cross-encoder logit (``float``).

            Returns ``[]`` when the vector search finds no candidates.
        """
        top_k_retrieval: int = get_settings().top_k_retrieval

        logger.info(
            "retrieve() query=%r  top_k=%d  top_k_retrieval=%d  filters=%s",
            query[:80],
            top_k,
            top_k_retrieval,
            filters,
        )

        t0 = time.perf_counter()

        # Step 1 — embed query
        query_vector: np.ndarray = self._encoder.encode(query)
        t1 = time.perf_counter()

        # Step 2 — vector search
        candidates = self._store.search(
            query_vector, top_k=top_k_retrieval, filters=filters
        )
        t2 = time.perf_counter()

        if not candidates:
            logger.info(
                "retrieve() — no candidates returned by vector search (%.3fs total)",
                t2 - t0,
            )
            return []

        # Step 3 — rerank
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)
        t3 = time.perf_counter()

        logger.info(
            "retrieve() done — total=%.3fs  encode=%.3fs  search=%.3fs  rerank=%.3fs  "
            "candidates=%d  returned=%d",
            t3 - t0,
            t1 - t0,
            t2 - t1,
            t3 - t2,
            len(candidates),
            len(reranked),
        )

        # Step 4 — reshape internal format → public API format
        results = self._reshape(reranked)

        for i, result in enumerate(results):
            logger.debug(
                "  [%d] score=%.4f  reranker=%.4f  chunk_id=%s",
                i,
                result["score"],
                result["reranker_score"],
                result["metadata"].get("chunk_id", "?"),
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape(reranked: list[dict]) -> list[dict]:
        """Convert reranker output to the clean public API format.

        Splits each reranked dict's ``"chunk"`` payload into ``"chunk_text"``
        (the body text) and ``"metadata"`` (all other payload keys).

        Args:
            reranked: Output of
                :meth:`src.retrieval.reranker.CrossEncoderReranker.rerank`.

        Returns:
            List of dicts with keys ``"chunk_text"``, ``"metadata"``,
            ``"score"``, and ``"reranker_score"``.
        """
        results: list[dict] = []
        for item in reranked:
            chunk = item["chunk"]
            chunk_text = chunk.get("text", "")
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            results.append(
                {
                    "chunk_text": chunk_text,
                    "metadata": metadata,
                    "score": item["score"],
                    "reranker_score": item["reranker_score"],
                }
            )
        return results


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_pipeline: RetrievalPipeline | None = None


def get_pipeline() -> RetrievalPipeline:
    """Return the process-level singleton :class:`RetrievalPipeline`.

    On the first call, constructs the pipeline using the singleton encoder,
    store, and reranker.  Subsequent calls return the cached instance.

    Returns:
        The singleton :class:`RetrievalPipeline` instance.
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising singleton RetrievalPipeline")
        _pipeline = RetrievalPipeline()
    return _pipeline
