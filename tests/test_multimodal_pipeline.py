"""Tests for src.retrieval.multimodal_pipeline — Module 5b.

All dependencies (encoders, store, reranker, config) are mocked so no model
downloads or Qdrant connection are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.retrieval.multimodal_pipeline import (
    MultiModalRetrievalPipeline,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM_TEXT = 384
_DIM_CLIP = 512


def _make_config(rrf_k: int = 60, clip_collection: str = "multimodal_clip") -> MagicMock:
    cfg = MagicMock()
    cfg.rrf_k = rrf_k
    cfg.clip_collection_name = clip_collection
    return cfg


def _make_text_encoder(vec: np.ndarray | None = None) -> MagicMock:
    enc = MagicMock()
    enc.encode.return_value = vec if vec is not None else np.zeros(_DIM_TEXT, dtype=np.float32)
    return enc


def _make_clip_encoder(vec: np.ndarray | None = None) -> MagicMock:
    enc = MagicMock()
    enc.encode_text.return_value = vec if vec is not None else np.zeros(_DIM_CLIP, dtype=np.float32)
    return enc


def _make_store(
    text_hits: list | None = None,
    clip_hits: list | None = None,
) -> MagicMock:
    store = MagicMock()
    store.search.return_value = text_hits if text_hits is not None else []
    store.search_clip.return_value = clip_hits if clip_hits is not None else []
    return store


def _make_reranker(reranked: list | None = None) -> MagicMock:
    rr = MagicMock()
    rr.rerank.return_value = reranked if reranked is not None else []
    return rr


def _text_qdrant_hit(chunk_id: str, text: str = "some text") -> dict:
    """Simulates a QdrantStore.search() hit."""
    return {
        "chunk": {
            "chunk_id": chunk_id,
            "text": text,
            "type": "text",
            "source_document": "test.pdf",
            "page_number": 1,
        },
        "score": 0.85,
    }


def _reranked_hit(chunk_id: str, text: str = "some text") -> dict:
    return {**_text_qdrant_hit(chunk_id, text), "reranker_score": 0.9}


def _clip_hit(image_id: str) -> dict:
    return {
        "id": f"uuid_{image_id}",
        "image_id": image_id,
        "type": "image",
        "caption": f"diagram {image_id}",
        "image_path": f"data/{image_id}.png",
        "score": 0.75,
    }


def _build_pipeline(
    text_hits: list | None = None,
    clip_hits: list | None = None,
    reranked: list | None = None,
) -> MultiModalRetrievalPipeline:
    return MultiModalRetrievalPipeline(
        text_encoder=_make_text_encoder(),
        clip_encoder=_make_clip_encoder(),
        store=_make_store(text_hits=text_hits, clip_hits=clip_hits),
        reranker=_make_reranker(reranked=reranked),
        config=_make_config(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retrieve_returns_retrieval_result():
    """`retrieve()` returns a `RetrievalResult` instance."""
    pipeline = _build_pipeline()
    result = pipeline.retrieve("chest X-ray findings")
    assert isinstance(result, RetrievalResult)


def test_retrieve_calls_both_encoders():
    """`retrieve()` calls both the text encoder and the CLIP encoder."""
    text_enc = _make_text_encoder()
    clip_enc = _make_clip_encoder()
    pipeline = MultiModalRetrievalPipeline(
        text_encoder=text_enc,
        clip_encoder=clip_enc,
        store=_make_store(),
        reranker=_make_reranker(),
        config=_make_config(),
    )
    pipeline.retrieve("bilateral pneumonia")

    text_enc.encode.assert_called_once_with("bilateral pneumonia")
    clip_enc.encode_text.assert_called_once_with("bilateral pneumonia")


def test_retrieve_calls_text_and_clip_search():
    """`retrieve()` calls both `store.search` and `store.search_clip`."""
    store = _make_store()
    pipeline = MultiModalRetrievalPipeline(
        text_encoder=_make_text_encoder(),
        clip_encoder=_make_clip_encoder(),
        store=store,
        reranker=_make_reranker(),
        config=_make_config(),
    )
    pipeline.retrieve("treatment for hypertension")

    store.search.assert_called_once()
    store.search_clip.assert_called_once()


def test_retrieve_applies_reranker_to_text():
    """`_retrieve_text` passes text candidates to the reranker."""
    candidates = [_text_qdrant_hit("c1"), _text_qdrant_hit("c2")]
    reranker = _make_reranker(reranked=[_reranked_hit("c1")])
    store = _make_store(text_hits=candidates)
    pipeline = MultiModalRetrievalPipeline(
        text_encoder=_make_text_encoder(),
        clip_encoder=_make_clip_encoder(),
        store=store,
        reranker=reranker,
        config=_make_config(),
    )
    pipeline.retrieve("amoxicillin dose")

    reranker.rerank.assert_called_once()
    call_args = reranker.rerank.call_args
    assert call_args[0][0] == "amoxicillin dose"          # query
    assert call_args[0][1] == candidates                   # candidates


def test_retrieve_does_not_rerank_images():
    """`_retrieve_images` does NOT call the reranker."""
    reranker = _make_reranker()
    store = _make_store(
        text_hits=[],
        clip_hits=[_clip_hit("img_001")],
    )
    pipeline = MultiModalRetrievalPipeline(
        text_encoder=_make_text_encoder(),
        clip_encoder=_make_clip_encoder(),
        store=store,
        reranker=reranker,
        config=_make_config(),
    )
    pipeline.retrieve("lung diagram")

    reranker.rerank.assert_not_called()


def test_retrieve_calls_fusion():
    """`retrieve()` calls `reciprocal_rank_fusion` with both result lists."""
    with patch(
        "src.retrieval.multimodal_pipeline.reciprocal_rank_fusion",
        return_value=[],
    ) as mock_rrf:
        pipeline = _build_pipeline(
            reranked=[_reranked_hit("c1")],
            clip_hits=[_clip_hit("img_001")],
        )
        pipeline.retrieve("test query")

        mock_rrf.assert_called_once()
        call_args = mock_rrf.call_args
        result_lists = call_args[0][0]
        assert len(result_lists) == 2
        assert call_args[1]["k"] == 60


def test_classify_separates_types():
    """`_classify_results` routes results correctly by `type` field."""
    fused = [
        {"type": "text", "chunk_id": "c1", "rrf_score": 0.03},
        {"type": "image", "image_id": "img1", "rrf_score": 0.02},
        {"type": "image_caption", "image_id": "img2", "rrf_score": 0.015},
        {"chunk_id": "c2", "rrf_score": 0.01},  # no type → text
    ]
    text_chunks, images = MultiModalRetrievalPipeline._classify_results(fused)

    assert len(text_chunks) == 2
    assert len(images) == 2
    assert all(r.get("type", "text") not in ("image", "image_caption") for r in text_chunks)
    assert all(r["type"] in ("image", "image_caption") for r in images)


def test_retrieve_measures_latency():
    """`retrieve()` returns a positive `retrieval_time_ms`."""
    pipeline = _build_pipeline()
    result = pipeline.retrieve("blood pressure treatment")
    assert result.retrieval_time_ms >= 0.0
    assert isinstance(result.retrieval_time_ms, float)
