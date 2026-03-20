"""Tests for src.retrieval.vector_store — QdrantStore and get_store()."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import src.retrieval.vector_store as vs_module
from src.retrieval.vector_store import QdrantStore, get_store

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    """A MagicMock standing in for QdrantClient."""
    client = MagicMock()
    # get_collections() returns an object with a .collections list attribute
    client.get_collections.return_value.collections = []
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> QdrantStore:
    """QdrantStore backed by the mock client."""
    return QdrantStore(_client=mock_client)


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------

_CHUNK_METADATA = {
    "chunk_id": "abc123def456",
    "chunk_index": 0,
    "chunking_strategy": "fixed_size",
    "source_document": "WHO-MHP-HPS-EML-2023.02-eng.pdf",
    "document_type": "essential_medicines_list",
    "section_title": "2.3 Medicines for palliative care",
    "page_number": 3,
}


def _make_chunks(n: int = 2) -> list[dict]:
    chunks = []
    for i in range(n):
        meta = {**_CHUNK_METADATA, "chunk_id": f"chunk_id_{i:04d}", "chunk_index": i}
        chunks.append({"text": f"chunk text {i}", "metadata": meta})
    return chunks


def _make_embeddings(n: int = 2, dim: int = 384) -> list[np.ndarray]:
    return [np.zeros(dim, dtype=np.float32) for _ in range(n)]


# ---------------------------------------------------------------------------
# create_collection tests
# ---------------------------------------------------------------------------


def test_create_collection_calls_client(store: QdrantStore, mock_client: MagicMock) -> None:
    store.create_collection(vector_size=384)
    mock_client.create_collection.assert_called_once()
    call_kwargs = mock_client.create_collection.call_args.kwargs
    assert call_kwargs["collection_name"] == "medical_docs"


def test_create_collection_vector_size(store: QdrantStore, mock_client: MagicMock) -> None:
    from qdrant_client.models import Distance, VectorParams

    store.create_collection(vector_size=384)
    call_kwargs = mock_client.create_collection.call_args.kwargs
    vc = call_kwargs["vectors_config"]
    assert isinstance(vc, VectorParams)
    assert vc.size == 384
    assert vc.distance == Distance.COSINE


def test_create_collection_creates_payload_indexes(
    store: QdrantStore, mock_client: MagicMock
) -> None:
    store.create_collection()
    calls = mock_client.create_payload_index.call_args_list
    field_names = {c.kwargs["field_name"] for c in calls}
    assert "document_type" in field_names
    assert "section_title" in field_names


def test_create_collection_skips_if_exists(store: QdrantStore, mock_client: MagicMock) -> None:
    """If the collection already exists, create_collection must not call create_collection."""
    existing = MagicMock()
    existing.name = "medical_docs"
    mock_client.get_collections.return_value.collections = [existing]

    store.create_collection()
    mock_client.create_collection.assert_not_called()


def test_create_collection_wraps_exception(mock_client: MagicMock) -> None:
    mock_client.get_collections.side_effect = RuntimeError("refused")
    store = QdrantStore(_client=mock_client)
    with pytest.raises(ConnectionError, match="Cannot connect to Qdrant"):
        store.create_collection()


# ---------------------------------------------------------------------------
# upsert_chunks tests
# ---------------------------------------------------------------------------


def test_upsert_chunks_calls_upsert(store: QdrantStore, mock_client: MagicMock) -> None:
    store.upsert_chunks(_make_chunks(2), _make_embeddings(2))
    mock_client.upsert.assert_called_once()


def test_upsert_returns_count(store: QdrantStore, mock_client: MagicMock) -> None:
    result = store.upsert_chunks(_make_chunks(3), _make_embeddings(3))
    assert result == 3


def test_upsert_point_ids_match_chunk_ids(store: QdrantStore, mock_client: MagicMock) -> None:
    chunks = _make_chunks(2)
    store.upsert_chunks(chunks, _make_embeddings(2))
    points = mock_client.upsert.call_args.kwargs["points"]
    for point, chunk in zip(points, chunks):
        assert point.id == chunk["metadata"]["chunk_id"]


def test_upsert_payload_contains_text(store: QdrantStore, mock_client: MagicMock) -> None:
    chunks = _make_chunks(2)
    store.upsert_chunks(chunks, _make_embeddings(2))
    points = mock_client.upsert.call_args.kwargs["points"]
    for point, chunk in zip(points, chunks):
        assert "text" in point.payload
        assert point.payload["text"] == chunk["text"]


def test_upsert_payload_contains_metadata(store: QdrantStore, mock_client: MagicMock) -> None:
    chunks = _make_chunks(2)
    store.upsert_chunks(chunks, _make_embeddings(2))
    points = mock_client.upsert.call_args.kwargs["points"]
    for point, chunk in zip(points, chunks):
        for key in chunk["metadata"]:
            assert key in point.payload


def test_upsert_empty_input(store: QdrantStore, mock_client: MagicMock) -> None:
    result = store.upsert_chunks([], [])
    assert result == 0
    mock_client.upsert.assert_not_called()


def test_upsert_wraps_exception(mock_client: MagicMock) -> None:
    mock_client.upsert.side_effect = RuntimeError("refused")
    store = QdrantStore(_client=mock_client)
    with pytest.raises(ConnectionError, match="Cannot connect to Qdrant"):
        store.upsert_chunks(_make_chunks(1), _make_embeddings(1))


# ---------------------------------------------------------------------------
# search tests
# ---------------------------------------------------------------------------


def _fake_hit(chunk_id: str, score: float, payload: dict) -> MagicMock:
    hit = MagicMock()
    hit.id = chunk_id
    hit.score = score
    hit.payload = payload
    return hit


def test_search_returns_list_of_dicts(store: QdrantStore, mock_client: MagicMock) -> None:
    payload = {"text": "some text", **_CHUNK_METADATA}
    mock_client.search.return_value = [_fake_hit("abc", 0.9, payload)]
    result = store.search(np.zeros(384, dtype=np.float32), top_k=5)
    assert isinstance(result, list)
    assert len(result) == 1
    assert "chunk" in result[0]
    assert "score" in result[0]


def test_search_passes_top_k(store: QdrantStore, mock_client: MagicMock) -> None:
    mock_client.search.return_value = []
    store.search(np.zeros(384, dtype=np.float32), top_k=10)
    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["limit"] == 10


def test_search_with_filter(store: QdrantStore, mock_client: MagicMock) -> None:
    from qdrant_client.models import Filter

    mock_client.search.return_value = []
    store.search(
        np.zeros(384, dtype=np.float32),
        top_k=5,
        filters={"document_type": "essential_medicines_list"},
    )
    call_kwargs = mock_client.search.call_args.kwargs
    assert isinstance(call_kwargs["query_filter"], Filter)


def test_search_no_filter(store: QdrantStore, mock_client: MagicMock) -> None:
    mock_client.search.return_value = []
    store.search(np.zeros(384, dtype=np.float32), top_k=5, filters=None)
    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["query_filter"] is None


def test_search_unknown_filter_key_ignored(store: QdrantStore, mock_client: MagicMock) -> None:
    """Unknown filter keys must not raise and must produce no Filter object."""
    mock_client.search.return_value = []
    store.search(
        np.zeros(384, dtype=np.float32),
        top_k=5,
        filters={"unknown_key": "value"},
    )
    call_kwargs = mock_client.search.call_args.kwargs
    assert call_kwargs["query_filter"] is None


def test_search_score_in_result(store: QdrantStore, mock_client: MagicMock) -> None:
    payload = {"text": "chunk", **_CHUNK_METADATA}
    mock_client.search.return_value = [_fake_hit("abc", 0.87, payload)]
    result = store.search(np.zeros(384, dtype=np.float32))
    assert abs(result[0]["score"] - 0.87) < 1e-6


def test_search_wraps_exception(mock_client: MagicMock) -> None:
    mock_client.search.side_effect = RuntimeError("refused")
    store = QdrantStore(_client=mock_client)
    with pytest.raises(ConnectionError, match="Cannot connect to Qdrant"):
        store.search(np.zeros(384, dtype=np.float32))


# ---------------------------------------------------------------------------
# delete_collection tests
# ---------------------------------------------------------------------------


def test_delete_collection_calls_client(store: QdrantStore, mock_client: MagicMock) -> None:
    store.delete_collection()
    mock_client.delete_collection.assert_called_once_with(
        collection_name="medical_docs"
    )


def test_delete_collection_wraps_exception(mock_client: MagicMock) -> None:
    mock_client.delete_collection.side_effect = RuntimeError("refused")
    store = QdrantStore(_client=mock_client)
    with pytest.raises(ConnectionError, match="Cannot connect to Qdrant"):
        store.delete_collection()


# ---------------------------------------------------------------------------
# get_collection_info tests
# ---------------------------------------------------------------------------


def test_get_collection_info_returns_dict(store: QdrantStore, mock_client: MagicMock) -> None:
    info_mock = MagicMock()
    info_mock.vectors_count = 42
    info_mock.points_count = 42
    info_mock.status = "green"
    mock_client.get_collection.return_value = info_mock

    result = store.get_collection_info()
    assert isinstance(result, dict)
    assert result["vectors_count"] == 42
    assert "status" in result


def test_get_collection_info_wraps_exception(mock_client: MagicMock) -> None:
    mock_client.get_collection.side_effect = RuntimeError("refused")
    store = QdrantStore(_client=mock_client)
    with pytest.raises(ConnectionError, match="Cannot connect to Qdrant"):
        store.get_collection_info()


# ---------------------------------------------------------------------------
# get_store() singleton tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_store_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset both singletons before/after every test."""
    import src.config as cfg

    cfg._settings = None
    vs_module._store = None
    yield
    cfg._settings = None
    vs_module._store = None


def _patch_store(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkeypatch QdrantStore so get_store() never creates a real client."""
    monkeypatch.setattr(
        vs_module,
        "QdrantStore",
        lambda host, port, collection_name: QdrantStore(
            host=host,
            port=port,
            collection_name=collection_name,
            _client=MagicMock(),
        ),
    )


def test_get_store_returns_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_store(monkeypatch)
    result = get_store()
    assert isinstance(result, QdrantStore)


def test_get_store_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_store(monkeypatch)
    s1 = get_store()
    s2 = get_store()
    assert s1 is s2


def test_get_store_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_store(monkeypatch)
    s1 = get_store()
    vs_module._store = None
    s2 = get_store()
    assert s1 is not s2


def test_get_store_uses_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("QDRANT_HOST", "qdrant-host")
    monkeypatch.setenv("QDRANT_PORT", "6334")
    monkeypatch.setenv("COLLECTION_NAME", "test_collection")

    captured: dict = {}

    def fake_store(host: str, port: int, collection_name: str) -> QdrantStore:
        captured["host"] = host
        captured["port"] = port
        captured["collection_name"] = collection_name
        return QdrantStore(
            host=host, port=port, collection_name=collection_name, _client=MagicMock()
        )

    monkeypatch.setattr(vs_module, "QdrantStore", fake_store)
    get_store()
    assert captured["host"] == "qdrant-host"
    assert captured["port"] == 6334
    assert captured["collection_name"] == "test_collection"
