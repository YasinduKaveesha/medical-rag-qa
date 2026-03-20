"""Tests for src.retrieval.pipeline — RetrievalPipeline and get_pipeline()."""

from __future__ import annotations

import numpy as np
import pytest

import src.retrieval.pipeline as pipeline_module
from src.retrieval.pipeline import RetrievalPipeline, get_pipeline

# ---------------------------------------------------------------------------
# Mock components
# ---------------------------------------------------------------------------


class MockEncoder:
    """Returns a fixed zero vector for any query."""

    def __init__(self) -> None:
        self.encode_calls: list[str] = []

    def encode(self, text: str) -> np.ndarray:
        self.encode_calls.append(text)
        return np.zeros(384, dtype=np.float32)


class MockStore:
    """Returns pre-set search results."""

    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.search_calls: list[tuple] = []

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        self.search_calls.append((query_vector, top_k, filters))
        return self._results


class MockReranker:
    """Returns pre-set reranked results."""

    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.rerank_calls: list[tuple] = []

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        self.rerank_calls.append((query, candidates, top_k))
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

QUERY = "What is the recommended dose of amitriptyline?"

_CHUNK_META = {
    "source_document": "WHO-MHP-HPS-EML-2023.02-eng.pdf",
    "document_type": "essential_medicines_list",
    "section_title": "2.3 Medicines for palliative care",
    "page_number": 3,
    "chunk_id": "abc123",
    "chunk_index": 0,
    "chunking_strategy": "fixed_size",
}


def _make_store_results(n: int = 3) -> list[dict]:
    """Fake QdrantStore.search() output."""
    results = []
    for i in range(n):
        chunk = {"text": f"chunk body {i}", **_CHUNK_META, "chunk_id": f"cid_{i}"}
        results.append({"chunk": chunk, "score": 0.9 - i * 0.05})
    return results


def _make_reranked(n: int = 3) -> list[dict]:
    """Fake CrossEncoderReranker.rerank() output."""
    results = []
    for i in range(n):
        chunk = {"text": f"chunk body {i}", **_CHUNK_META, "chunk_id": f"cid_{i}"}
        results.append(
            {"chunk": chunk, "score": 0.9 - i * 0.05, "reranker_score": 5.0 - i}
        )
    return results


def _make_pipeline(
    n_store: int = 3,
    n_reranked: int = 3,
    store_results: list[dict] | None = None,
    reranked_results: list[dict] | None = None,
) -> tuple[RetrievalPipeline, MockEncoder, MockStore, MockReranker]:
    encoder = MockEncoder()
    store = MockStore(store_results if store_results is not None else _make_store_results(n_store))
    reranker = MockReranker(
        reranked_results if reranked_results is not None else _make_reranked(n_reranked)
    )
    pipeline = RetrievalPipeline(encoder=encoder, store=store, reranker=reranker)
    return pipeline, encoder, store, reranker


# ---------------------------------------------------------------------------
# retrieve() — output structure
# ---------------------------------------------------------------------------


def test_retrieve_returns_list_of_dicts() -> None:
    pipeline, *_ = _make_pipeline()
    result = pipeline.retrieve(QUERY)
    assert isinstance(result, list)
    for item in result:
        assert "chunk_text" in item
        assert "metadata" in item
        assert "score" in item
        assert "reranker_score" in item


def test_retrieve_chunk_text_key() -> None:
    pipeline, *_ = _make_pipeline()
    result = pipeline.retrieve(QUERY, top_k=3)
    for i, item in enumerate(result):
        assert item["chunk_text"] == f"chunk body {i}"


def test_retrieve_metadata_key() -> None:
    pipeline, *_ = _make_pipeline()
    result = pipeline.retrieve(QUERY, top_k=1)
    meta = result[0]["metadata"]
    assert isinstance(meta, dict)
    assert "source_document" in meta
    assert "chunk_id" in meta


def test_retrieve_metadata_excludes_text() -> None:
    """'text' must not appear inside the metadata dict."""
    pipeline, *_ = _make_pipeline()
    result = pipeline.retrieve(QUERY, top_k=1)
    assert "text" not in result[0]["metadata"]


def test_retrieve_score_preserved() -> None:
    reranked = _make_reranked(2)
    pipeline, *_ = _make_pipeline(reranked_results=reranked)
    result = pipeline.retrieve(QUERY, top_k=2)
    assert abs(result[0]["score"] - reranked[0]["score"]) < 1e-6
    assert abs(result[1]["score"] - reranked[1]["score"]) < 1e-6


def test_retrieve_reranker_score_present() -> None:
    pipeline, *_ = _make_pipeline()
    result = pipeline.retrieve(QUERY, top_k=1)
    assert isinstance(result[0]["reranker_score"], float)


def test_retrieve_output_length() -> None:
    pipeline, *_ = _make_pipeline(n_reranked=3)
    result = pipeline.retrieve(QUERY, top_k=3)
    assert len(result) == 3


def test_retrieve_top_k_respected() -> None:
    """When top_k=2, at most 2 results are returned."""
    pipeline, *_ = _make_pipeline(n_reranked=3)
    result = pipeline.retrieve(QUERY, top_k=2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# retrieve() — component call behaviour
# ---------------------------------------------------------------------------


def test_retrieve_calls_encoder() -> None:
    pipeline, encoder, *_ = _make_pipeline()
    pipeline.retrieve(QUERY)
    assert encoder.encode_calls == [QUERY]


def test_retrieve_calls_store_search() -> None:
    pipeline, _, store, _ = _make_pipeline()
    pipeline.retrieve(QUERY)
    assert len(store.search_calls) == 1


def test_retrieve_passes_top_k_retrieval_to_store() -> None:
    """Store must receive top_k_retrieval (from settings), NOT the user's top_k."""
    import src.config as cfg

    cfg._settings = None
    pipeline, _, store, _ = _make_pipeline()
    pipeline.retrieve(QUERY, top_k=2)
    _, store_top_k, _ = store.search_calls[0]
    from src.config import get_settings

    assert store_top_k == get_settings().top_k_retrieval


def test_retrieve_passes_filters_to_store() -> None:
    pipeline, _, store, _ = _make_pipeline()
    filt = {"document_type": "essential_medicines_list"}
    pipeline.retrieve(QUERY, filters=filt)
    _, _, passed_filters = store.search_calls[0]
    assert passed_filters == filt


def test_retrieve_no_filter_passes_none_to_store() -> None:
    pipeline, _, store, _ = _make_pipeline()
    pipeline.retrieve(QUERY, filters=None)
    _, _, passed_filters = store.search_calls[0]
    assert passed_filters is None


def test_retrieve_calls_reranker() -> None:
    pipeline, _, _, reranker = _make_pipeline()
    pipeline.retrieve(QUERY)
    assert len(reranker.rerank_calls) == 1


def test_retrieve_passes_query_to_reranker() -> None:
    pipeline, _, _, reranker = _make_pipeline()
    pipeline.retrieve(QUERY)
    passed_query, _, _ = reranker.rerank_calls[0]
    assert passed_query == QUERY


def test_retrieve_passes_top_k_to_reranker() -> None:
    pipeline, _, _, reranker = _make_pipeline()
    pipeline.retrieve(QUERY, top_k=3)
    _, _, passed_top_k = reranker.rerank_calls[0]
    assert passed_top_k == 3


def test_retrieve_passes_store_results_to_reranker() -> None:
    store_results = _make_store_results(4)
    pipeline, _, _, reranker = _make_pipeline(store_results=store_results)
    pipeline.retrieve(QUERY)
    _, passed_candidates, _ = reranker.rerank_calls[0]
    assert passed_candidates == store_results


# ---------------------------------------------------------------------------
# retrieve() — edge cases
# ---------------------------------------------------------------------------


def test_retrieve_empty_candidates() -> None:
    """When store returns no results, pipeline returns [] without calling reranker."""
    pipeline, _, _, reranker = _make_pipeline(store_results=[])
    result = pipeline.retrieve(QUERY)
    assert result == []
    assert reranker.rerank_calls == []


def test_retrieve_encoder_vector_forwarded_to_store() -> None:
    """The exact vector returned by encoder must be passed to store.search."""
    pipeline, encoder, store, _ = _make_pipeline()
    pipeline.retrieve(QUERY)
    passed_vector, _, _ = store.search_calls[0]
    assert np.array_equal(passed_vector, np.zeros(384, dtype=np.float32))


# ---------------------------------------------------------------------------
# get_pipeline() singleton tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset all singletons before and after every test."""
    import src.config as cfg

    cfg._settings = None
    pipeline_module._pipeline = None
    yield
    cfg._settings = None
    pipeline_module._pipeline = None


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch RetrievalPipeline so get_pipeline() never loads real deps."""
    monkeypatch.setattr(
        pipeline_module,
        "RetrievalPipeline",
        lambda: RetrievalPipeline(
            encoder=MockEncoder(),
            store=MockStore(_make_store_results()),
            reranker=MockReranker(_make_reranked()),
        ),
    )


def test_get_pipeline_returns_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_pipeline(monkeypatch)
    result = get_pipeline()
    assert isinstance(result, RetrievalPipeline)


def test_get_pipeline_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_pipeline(monkeypatch)
    p1 = get_pipeline()
    p2 = get_pipeline()
    assert p1 is p2


def test_get_pipeline_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_pipeline(monkeypatch)
    p1 = get_pipeline()
    pipeline_module._pipeline = None
    p2 = get_pipeline()
    assert p1 is not p2
