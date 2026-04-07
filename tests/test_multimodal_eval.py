"""Tests for Module 8 — MultiModalEvaluator (all mocked, no real models)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.multimodal_eval import MultiModalEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(text_chunks=None, images=None):
    """Build a minimal RetrievalResult-like object."""
    return SimpleNamespace(
        text_chunks=text_chunks if text_chunks is not None else [],
        images=images if images is not None else [],
        fusion_scores={},
        retrieval_time_ms=10.0,
    )


def _text_chunk(text: str = "amoxicillin dosage 500mg pneumonia antibiotic") -> dict:
    return {
        "text": text,
        "source_document": "who_guidelines.pdf",
        "page_number": 5,
        "score": 0.9,
    }


def _image(caption: str = "chest X-ray showing bilateral infiltrates consolidation opacity") -> dict:
    return {
        "caption": caption,
        "source_pdf": "radiology_atlas.pdf",
        "page_number": 3,
        "image_id": "atlas_p3_x1",
        "type": "image",
        "score": 0.8,
    }


# ---------------------------------------------------------------------------
# Sample query fixtures
# ---------------------------------------------------------------------------


_IMAGE_QUERY = {
    "query_id": "img_001",
    "question": "Show me the anatomical diagram",
    "expected_type": "image",
    "expected_keywords": ["infiltrate", "bilateral", "chest"],
}

_TEXT_QUERY = {
    "query_id": "txt_001",
    "question": "What is the recommended dosage of amoxicillin?",
    "expected_type": "text",
    "expected_keywords": ["amoxicillin", "dosage", "500mg"],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evaluate_retrieval_returns_summary_keys():
    """evaluate_retrieval returns a dict with all expected top-level keys."""
    mm = MagicMock()
    mm.retrieve.return_value = _make_result(
        text_chunks=[_text_chunk()],
        images=[_image()],
    )
    evaluator = MultiModalEvaluator(mm, queries=[_IMAGE_QUERY, _TEXT_QUERY])
    results = evaluator.evaluate_retrieval(top_k=5)

    for key in ("total_queries", "image_queries", "text_queries",
                "image_precision", "text_precision", "modality_accuracy", "per_query"):
        assert key in results, f"Missing key: {key}"


def test_evaluate_retrieval_counts_queries():
    """Query counts match the provided query list."""
    mm = MagicMock()
    mm.retrieve.return_value = _make_result(text_chunks=[_text_chunk()], images=[_image()])
    queries = [_IMAGE_QUERY, _IMAGE_QUERY, _TEXT_QUERY]
    evaluator = MultiModalEvaluator(mm, queries=queries)
    results = evaluator.evaluate_retrieval(top_k=5)

    assert results["total_queries"] == 3
    assert results["image_queries"] == 2
    assert results["text_queries"] == 1


def test_image_retrieval_precision_full_match():
    """image_retrieval_precision returns 1.0 when all keywords appear in captions."""
    mm = MagicMock()
    evaluator = MultiModalEvaluator(mm, queries=[])
    query = {"expected_keywords": ["bilateral", "infiltrate", "chest"]}
    result = _make_result(images=[_image("bilateral infiltrate chest X-ray")])
    score = evaluator.image_retrieval_precision(query, result)
    assert score == pytest.approx(1.0)


def test_image_retrieval_precision_no_images():
    """image_retrieval_precision returns 0.0 when no images were retrieved."""
    mm = MagicMock()
    evaluator = MultiModalEvaluator(mm, queries=[])
    query = {"expected_keywords": ["bilateral", "infiltrate"]}
    result = _make_result(images=[])
    score = evaluator.image_retrieval_precision(query, result)
    assert score == pytest.approx(0.0)


def test_text_retrieval_precision_partial_match():
    """text_retrieval_precision correctly computes partial keyword overlap."""
    mm = MagicMock()
    evaluator = MultiModalEvaluator(mm, queries=[])
    query = {"expected_keywords": ["amoxicillin", "dosage", "glucose"]}
    result = _make_result(text_chunks=[_text_chunk("amoxicillin dosage 500mg")])
    # "amoxicillin" and "dosage" match; "glucose" does not → 2/3
    score = evaluator.text_retrieval_precision(query, result)
    assert score == pytest.approx(2 / 3)


def test_modality_accuracy_image_query():
    """modality_accuracy is high when images dominate for an image query."""
    mm = MagicMock()
    evaluator = MultiModalEvaluator(mm, queries=[])
    query = {"expected_type": "image"}
    result = _make_result(text_chunks=[_text_chunk()], images=[_image(), _image()])
    # 2 images / 3 total ≈ 0.667
    score = evaluator.modality_accuracy(query, result)
    assert score == pytest.approx(2 / 3)


def test_modality_accuracy_empty_result():
    """modality_accuracy returns 0.0 when nothing was retrieved."""
    mm = MagicMock()
    evaluator = MultiModalEvaluator(mm, queries=[])
    query = {"expected_type": "text"}
    result = _make_result()
    assert evaluator.modality_accuracy(query, result) == pytest.approx(0.0)


def test_generate_report_contains_header():
    """generate_report returns a string containing Markdown table header."""
    mm = MagicMock()
    mm.retrieve.return_value = _make_result(text_chunks=[_text_chunk()], images=[_image()])
    evaluator = MultiModalEvaluator(mm, queries=[_TEXT_QUERY])
    results = evaluator.evaluate_retrieval(top_k=5)
    report = evaluator.generate_report(results)

    assert isinstance(report, str)
    assert "# Multimodal RAG Evaluation Report" in report
    assert "txt_001" in report


def test_compare_raises_without_text_pipeline():
    """compare_text_only_vs_multimodal raises RuntimeError when no text_pipeline given."""
    mm = MagicMock()
    evaluator = MultiModalEvaluator(mm, queries=[_TEXT_QUERY])
    with pytest.raises(RuntimeError, match="text_pipeline"):
        evaluator.compare_text_only_vs_multimodal()
