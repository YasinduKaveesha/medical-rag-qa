"""Cross-encoder reranker for the Medical RAG Q&A pipeline.

Provides a singleton :class:`CrossEncoderReranker` that wraps
``cross-encoder/ms-marco-MiniLM-L-6-v2`` and re-scores the top-K candidates
returned by :class:`src.retrieval.vector_store.QdrantStore`.

Typical usage
-------------
::

    from src.retrieval.reranker import get_reranker

    reranker = get_reranker()
    results = reranker.rerank(query, candidates, top_k=5)
    # results[i]["score"]          — original cosine similarity
    # results[i]["reranker_score"] — cross-encoder logit (higher = more relevant)
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Wrapper around a ``CrossEncoder`` model for candidate reranking.

    The model is loaded once in ``__init__`` and reused for every subsequent
    call to :meth:`rerank`.  Use :func:`get_reranker` to obtain the
    process-level singleton instead of constructing this class directly.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
        _model: Optional pre-loaded model object.  When provided it is used
            directly, bypassing ``CrossEncoder`` loading.  Intended for
            testing only — pass a :class:`MockCrossEncoder` to avoid
            downloading the real weights.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        _model: Any | None = None,
    ) -> None:
        self._model_name = model_name

        if _model is not None:
            self._model = _model
            logger.debug("CrossEncoderReranker using injected model (test mode)")
        else:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info("Loading CrossEncoder: %s", model_name)
            self._model = CrossEncoder(model_name)
            logger.info("Reranker model loaded")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """HuggingFace model identifier used by this reranker."""
        return self._model_name

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Re-score and re-rank *candidates* using the cross-encoder.

        Consumes the output of :meth:`src.retrieval.vector_store.QdrantStore.search`
        directly — a list of ``{"chunk": dict, "score": float}`` dicts where
        ``score`` is the cosine similarity from Qdrant.

        Args:
            query: Raw question string used to score each candidate.
            candidates: List of candidate dicts from ``QdrantStore.search()``.
                Each dict must have keys ``"chunk"`` (with a ``"text"`` sub-key)
                and ``"score"`` (cosine similarity float).
            top_k: Number of top-ranked results to return.  Clamped to
                ``len(candidates)`` so the call never raises on small inputs.
                Defaults to ``5``.

        Returns:
            List of dicts sorted by ``"reranker_score"`` descending, length
            ``min(top_k, len(candidates))``.  Each dict contains:

            - ``"chunk"`` — unchanged from the input candidate.
            - ``"score"`` — original cosine similarity (preserved).
            - ``"reranker_score"`` — raw cross-encoder logit (higher = more
              relevant).  Not normalised so the full signal is available for
              RAGAS evaluation in Module 9.

            Returns ``[]`` immediately for empty *candidates* input.
        """
        if not candidates:
            return []

        top_k = min(top_k, len(candidates))

        pairs = [[query, c["chunk"]["text"]] for c in candidates]

        logger.info(
            "Reranking %d candidates (top_k=%d) with %s",
            len(candidates),
            top_k,
            self._model_name,
        )

        scores = self._model.predict(pairs)

        scored: list[dict] = []
        for candidate, reranker_score in zip(candidates, scores):
            scored.append(
                {
                    "chunk": candidate["chunk"],
                    "score": candidate["score"],
                    "reranker_score": float(reranker_score),
                }
            )

        scored.sort(key=lambda x: x["reranker_score"], reverse=True)

        logger.debug(
            "Reranking complete — top score: %.4f, bottom score: %.4f",
            scored[0]["reranker_score"],
            scored[top_k - 1]["reranker_score"],
        )

        return scored[:top_k]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    """Return the process-level singleton :class:`CrossEncoderReranker`.

    On the first call, reads ``Settings.reranker_model`` (from ``.env``) and
    loads the model.  Subsequent calls return the cached instance without
    reloading.

    Returns:
        The singleton :class:`CrossEncoderReranker` instance.
    """
    global _reranker
    if _reranker is None:
        model_name = get_settings().reranker_model
        logger.info("Initialising singleton CrossEncoderReranker: %s", model_name)
        _reranker = CrossEncoderReranker(model_name=model_name)
    return _reranker
