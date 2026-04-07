"""Tests for src.retrieval.fusion — Module 5a: RRF Fusion."""

from __future__ import annotations

import pytest

from src.retrieval.fusion import reciprocal_rank_fusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_result(chunk_id: str, score: float = 0.9, text: str = "some text") -> dict:
    return {"chunk_id": chunk_id, "text": text, "score": score, "type": "text"}


def _image_result(image_id: str, score: float = 0.8, caption: str = "a diagram") -> dict:
    return {
        "image_id": image_id,
        "id": image_id,
        "type": "image",
        "caption": caption,
        "score": score,
        "image_path": f"data/{image_id}.png",
    }


def _caption_result(image_id: str, score: float = 0.7) -> dict:
    return {
        "image_id": image_id,
        "id": image_id,
        "type": "image_caption",
        "text": f"caption for {image_id}",
        "score": score,
        "source_document": "test.pdf",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rrf_single_list():
    """A single result list with 3 items returns them all, ranked by RRF score."""
    items = [_text_result(f"c{i}") for i in range(3)]
    result = reciprocal_rank_fusion([items], k=60, top_k=10)
    assert len(result) == 3
    assert all("rrf_score" in r for r in result)


def test_rrf_two_lists_merge():
    """Two disjoint lists are merged; total = sum of unique items (capped by top_k)."""
    list_a = [_text_result("c1"), _text_result("c2")]
    list_b = [_image_result("img1"), _image_result("img2")]
    result = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=10)
    assert len(result) == 4


def test_rrf_duplicate_boosted():
    """An item appearing in both lists scores higher than items in only one list."""
    shared = _text_result("shared_chunk")
    unique_a = _text_result("only_in_a")
    unique_b = _text_result("only_in_b")

    list_a = [shared, unique_a]
    list_b = [shared, unique_b]

    result = reciprocal_rank_fusion([list_a, list_b], k=60, top_k=10)
    scores = {r["chunk_id"]: r["rrf_score"] for r in result if "chunk_id" in r}

    assert scores["shared_chunk"] > scores["only_in_a"]
    assert scores["shared_chunk"] > scores["only_in_b"]


def test_rrf_respects_top_k():
    """Result is truncated to top_k items."""
    items = [_text_result(f"c{i}") for i in range(20)]
    result = reciprocal_rank_fusion([items], k=60, top_k=5)
    assert len(result) == 5


def test_rrf_empty_list():
    """Empty input lists return an empty result."""
    assert reciprocal_rank_fusion([], k=60, top_k=10) == []
    assert reciprocal_rank_fusion([[]], k=60, top_k=10) == []
    assert reciprocal_rank_fusion([[], []], k=60, top_k=10) == []


def test_rrf_preserves_metadata():
    """All metadata fields from original results are present in the output."""
    item = {
        "chunk_id": "c1",
        "text": "amoxicillin dosage",
        "source_document": "who.pdf",
        "page_number": 3,
        "score": 0.9,
        "type": "text",
    }
    result = reciprocal_rank_fusion([[item]], k=60, top_k=10)
    assert len(result) == 1
    for field in ("chunk_id", "text", "source_document", "page_number", "type"):
        assert field in result[0], f"Missing field: {field}"


def test_rrf_score_decreasing():
    """Results are sorted by rrf_score in descending order."""
    # Items ranked 1,2,3 in list — rank 1 should get highest RRF score
    items = [_text_result(f"c{i}") for i in range(5)]
    result = reciprocal_rank_fusion([items], k=60, top_k=10)
    scores = [r["rrf_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_deduplicates_by_image_id():
    """The same image_id appearing in both CLIP and caption lists yields ONE entry."""
    image_hit = _image_result("img_001", score=0.95)
    caption_hit = _caption_result("img_001", score=0.80)

    clip_list = [image_hit]
    text_list = [caption_hit]

    result = reciprocal_rank_fusion([clip_list, text_list], k=60, top_k=10)

    # All entries with image_id == "img_001" should be collapsed to one
    image_entries = [r for r in result if r.get("image_id") == "img_001"]
    assert len(image_entries) == 1, (
        f"Expected 1 entry for img_001, got {len(image_entries)}"
    )


def test_rrf_dedup_merges_metadata():
    """Surviving deduplicated entry has fields from both hits and retrieval_sources."""
    image_hit = _image_result("img_002", score=0.95)
    image_hit["width"] = 800
    image_hit["height"] = 600

    caption_hit = _caption_result("img_002", score=0.80)
    caption_hit["source_document"] = "clinical_guidelines.pdf"

    result = reciprocal_rank_fusion([[image_hit], [caption_hit]], k=60, top_k=10)

    entries = [r for r in result if r.get("image_id") == "img_002"]
    assert len(entries) == 1
    merged = entries[0]

    # Fields from image hit
    assert merged.get("image_path") is not None
    # Fields from caption hit
    assert merged.get("source_document") == "clinical_guidelines.pdf"
    # retrieval_sources must list both types
    sources = merged.get("retrieval_sources", [])
    assert "image" in sources, f"retrieval_sources={sources}"
    assert "image_caption" in sources, f"retrieval_sources={sources}"
