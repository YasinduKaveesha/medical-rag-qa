"""Refusal gate for the Medical RAG Q&A generation step.

Determines whether the retrieved chunks are relevant enough to attempt an
answer.  When the maximum cosine similarity across all retrieved chunks falls
below the configured threshold the system should decline to answer rather
than hallucinate from low-quality context.

Typical usage
-------------
::

    from src.generation.refusal import should_refuse

    if should_refuse(chunks):
        return "I cannot answer from the provided documents."
    answer = client.generate(build_prompt(query, chunks))
"""

from __future__ import annotations

import logging

from src.config import get_settings

logger = logging.getLogger(__name__)


def should_refuse(chunks: list[dict]) -> bool:
    """Return ``True`` if retrieval quality is too low to answer safely.

    Compares the **maximum** cosine similarity score across all retrieved
    chunks against ``Settings.similarity_threshold`` (default ``0.35``).
    Cosine similarity is used rather than the cross-encoder ``reranker_score``
    because it is bounded in ``[0, 1]`` and directly calibrated against the
    threshold value in Settings.

    Args:
        chunks: List of retrieval result dicts as returned by
            :meth:`src.retrieval.pipeline.RetrievalPipeline.retrieve`.
            Each dict must have a ``"score"`` key (cosine similarity float).

    Returns:
        ``True`` if *chunks* is empty or if the maximum ``"score"`` across
        all chunks is strictly less than ``Settings.similarity_threshold``.
        ``False`` otherwise (including when max score equals the threshold).
    """
    if not chunks:
        logger.info("should_refuse: no chunks returned — refusing")
        return True

    max_score = max(c["score"] for c in chunks)
    threshold = get_settings().similarity_threshold

    if max_score < threshold:
        logger.info(
            "should_refuse: max cosine score %.4f < threshold %.4f — refusing",
            max_score,
            threshold,
        )
        return True

    logger.debug(
        "should_refuse: max cosine score %.4f >= threshold %.4f — proceeding",
        max_score,
        threshold,
    )
    return False
