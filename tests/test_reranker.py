"""Tests for src.retrieval.reranker — CrossEncoderReranker and get_reranker()."""

from __future__ import annotations

import pytest

import src.retrieval.reranker as reranker_module
from src.retrieval.reranker import CrossEncoderReranker, get_reranker

# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockCrossEncoder:
    """Returns fixed scores without loading real weights."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.predict_calls: list[list] = []

    def predict(self, pairs: list) -> list[float]:
        self.predict_calls.append(pairs)
        return self._scores[: len(pairs)]


# ---------------------------------------------------------------------------
# Shared fixtures and sample data
# ---------------------------------------------------------------------------

QUERY = "What is the recommended dose of amitriptyline?"

_CHUNK_TEMPLATE = {
    "source_document": "WHO-MHP-HPS-EML-2023.02-eng.pdf",
    "document_type": "essential_medicines_list",
    "section_title": "2.3 Medicines for palliative care",
    "page_number": 3,
    "chunk_index": 0,
    "chunking_strategy": "fixed_size",
}


def _make_candidates(n: int, cosine_scores: list[float] | None = None) -> list[dict]:
    """Build n fake search-result dicts as returned by QdrantStore.search()."""
    if cosine_scores is None:
        cosine_scores = [0.9 - i * 0.05 for i in range(n)]
    candidates = []
    for i in range(n):
        meta = {**_CHUNK_TEMPLATE, "chunk_id": f"chunk_{i:04d}", "chunk_index": i}
        candidates.append(
            {
                "chunk": {"text": f"chunk text {i}", **meta},
                "score": cosine_scores[i],
            }
        )
    return candidates


@pytest.fixture
def mock_model_3() -> MockCrossEncoder:
    """Mock that returns 3 distinct scores."""
    return MockCrossEncoder([2.5, 8.0, -1.3])


@pytest.fixture
def reranker_3(mock_model_3: MockCrossEncoder) -> CrossEncoderReranker:
    """Reranker backed by a 3-score mock."""
    return CrossEncoderReranker(_model=mock_model_3)


# ---------------------------------------------------------------------------
# rerank() — output structure
# ---------------------------------------------------------------------------


def test_rerank_returns_list_of_dicts(reranker_3: CrossEncoderReranker) -> None:
    result = reranker_3.rerank(QUERY, _make_candidates(3))
    assert isinstance(result, list)
    for item in result:
        assert "chunk" in item
        assert "score" in item
        assert "reranker_score" in item


def test_rerank_top_k_limits_output(reranker_3: CrossEncoderReranker) -> None:
    result = reranker_3.rerank(QUERY, _make_candidates(3), top_k=2)
    assert len(result) == 2


def test_rerank_top_k_clamped_to_candidates() -> None:
    """top_k larger than candidate count must not raise."""
    mock = MockCrossEncoder([1.0, 2.0, 3.0])
    reranker = CrossEncoderReranker(_model=mock)
    result = reranker.rerank(QUERY, _make_candidates(3), top_k=100)
    assert len(result) == 3


def test_rerank_empty_candidates(reranker_3: CrossEncoderReranker) -> None:
    result = reranker_3.rerank(QUERY, [])
    assert result == []


def test_rerank_empty_does_not_call_model(mock_model_3: MockCrossEncoder) -> None:
    reranker = CrossEncoderReranker(_model=mock_model_3)
    reranker.rerank(QUERY, [])
    assert mock_model_3.predict_calls == []


# ---------------------------------------------------------------------------
# rerank() — ordering and scores
# ---------------------------------------------------------------------------


def test_rerank_sorted_by_reranker_score() -> None:
    """Results must be ordered by reranker_score descending."""
    mock = MockCrossEncoder([2.5, 8.0, -1.3])
    reranker = CrossEncoderReranker(_model=mock)
    result = reranker.rerank(QUERY, _make_candidates(3), top_k=3)
    scores = [r["reranker_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_rerank_preserves_cosine_score() -> None:
    """Original cosine 'score' values must be unchanged."""
    cosine_scores = [0.85, 0.72, 0.60]
    mock = MockCrossEncoder([1.0, 2.0, 3.0])
    reranker = CrossEncoderReranker(_model=mock)
    result = reranker.rerank(QUERY, _make_candidates(3, cosine_scores), top_k=3)

    # Collect cosine scores from result; order may have changed due to reranking
    result_cosines = sorted(r["score"] for r in result)
    assert result_cosines == sorted(cosine_scores)


def test_rerank_reranker_score_matches_mock() -> None:
    """reranker_score values in the output must match what the mock returned."""
    raw_scores = [5.0, 1.0, 9.0]
    mock = MockCrossEncoder(raw_scores)
    reranker = CrossEncoderReranker(_model=mock)
    result = reranker.rerank(QUERY, _make_candidates(3), top_k=3)

    result_reranker_scores = sorted(r["reranker_score"] for r in result)
    assert result_reranker_scores == sorted(raw_scores)


def test_rerank_top_result_has_highest_score() -> None:
    """First result must have the highest reranker_score."""
    mock = MockCrossEncoder([2.0, 9.5, -0.5])
    reranker = CrossEncoderReranker(_model=mock)
    result = reranker.rerank(QUERY, _make_candidates(3), top_k=3)
    assert result[0]["reranker_score"] == max(r["reranker_score"] for r in result)


# ---------------------------------------------------------------------------
# rerank() — model call behaviour
# ---------------------------------------------------------------------------


def test_rerank_calls_predict_once(mock_model_3: MockCrossEncoder) -> None:
    reranker = CrossEncoderReranker(_model=mock_model_3)
    reranker.rerank(QUERY, _make_candidates(3))
    assert len(mock_model_3.predict_calls) == 1


def test_rerank_predict_pairs_contain_query(mock_model_3: MockCrossEncoder) -> None:
    reranker = CrossEncoderReranker(_model=mock_model_3)
    candidates = _make_candidates(3)
    reranker.rerank(QUERY, candidates)
    pairs = mock_model_3.predict_calls[0]
    for pair in pairs:
        assert pair[0] == QUERY


def test_rerank_predict_pairs_contain_chunk_text(mock_model_3: MockCrossEncoder) -> None:
    reranker = CrossEncoderReranker(_model=mock_model_3)
    candidates = _make_candidates(3)
    reranker.rerank(QUERY, candidates)
    pairs = mock_model_3.predict_calls[0]
    expected_texts = [c["chunk"]["text"] for c in candidates]
    actual_texts = [p[1] for p in pairs]
    assert actual_texts == expected_texts


def test_rerank_predict_receives_all_candidates(mock_model_3: MockCrossEncoder) -> None:
    """predict() must be called with ALL candidates, not just top_k."""
    reranker = CrossEncoderReranker(_model=mock_model_3)
    reranker.rerank(QUERY, _make_candidates(3), top_k=1)
    assert len(mock_model_3.predict_calls[0]) == 3


# ---------------------------------------------------------------------------
# rerank() — edge cases
# ---------------------------------------------------------------------------


def test_rerank_single_candidate() -> None:
    mock = MockCrossEncoder([7.3])
    reranker = CrossEncoderReranker(_model=mock)
    result = reranker.rerank(QUERY, _make_candidates(1), top_k=5)
    assert len(result) == 1
    assert abs(result[0]["reranker_score"] - 7.3) < 1e-6


def test_rerank_chunk_unchanged() -> None:
    """The 'chunk' dict in the output must be identical to the input."""
    mock = MockCrossEncoder([1.0])
    reranker = CrossEncoderReranker(_model=mock)
    candidates = _make_candidates(1)
    result = reranker.rerank(QUERY, candidates, top_k=1)
    assert result[0]["chunk"] == candidates[0]["chunk"]


# ---------------------------------------------------------------------------
# model_name property
# ---------------------------------------------------------------------------


def test_model_name_property() -> None:
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        _model=MockCrossEncoder([]),
    )
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# get_reranker() singleton tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset both singletons before and after every test in this module."""
    import src.config as cfg

    cfg._settings = None
    reranker_module._reranker = None
    yield
    cfg._settings = None
    reranker_module._reranker = None


def _patch_reranker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch CrossEncoderReranker so get_reranker() never loads real weights."""
    monkeypatch.setattr(
        reranker_module,
        "CrossEncoderReranker",
        lambda model_name: CrossEncoderReranker(
            model_name=model_name,
            _model=MockCrossEncoder([]),
        ),
    )


def test_get_reranker_returns_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_reranker(monkeypatch)
    result = get_reranker()
    assert isinstance(result, CrossEncoderReranker)


def test_get_reranker_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_reranker(monkeypatch)
    r1 = get_reranker()
    r2 = get_reranker()
    assert r1 is r2


def test_get_reranker_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_reranker(monkeypatch)
    r1 = get_reranker()
    reranker_module._reranker = None
    r2 = get_reranker()
    assert r1 is not r2


def test_get_reranker_uses_settings_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    captured: list[str] = []

    def fake_reranker(model_name: str) -> CrossEncoderReranker:
        captured.append(model_name)
        return CrossEncoderReranker(model_name=model_name, _model=MockCrossEncoder([]))

    monkeypatch.setattr(reranker_module, "CrossEncoderReranker", fake_reranker)
    get_reranker()
    assert captured[0] == "cross-encoder/ms-marco-MiniLM-L-6-v2"
