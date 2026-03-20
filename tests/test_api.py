"""Tests for app.main — POST /ask and GET /health endpoints.

All tests run against an in-process ASGI app via Starlette's TestClient.
No real Qdrant or Groq calls are made: the pipeline, LLM client, and store
are replaced with MagicMocks through FastAPI's dependency_overrides system.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared sample data (same shape as test_generation.py helpers)
# ---------------------------------------------------------------------------

_META = {
    "source_document": "WHO-MHP-HPS-EML-2023.02-eng.pdf",
    "document_type": "essential_medicines_list",
    "section_title": "2.3 Medicines for palliative care",
    "page_number": 3,
    "chunk_id": "abc123",
    "chunk_index": 0,
    "chunking_strategy": "fixed_size",
}


def _make_chunks(n: int = 2, scores: list[float] | None = None) -> list[dict]:
    if scores is None:
        scores = [0.9 - i * 0.1 for i in range(n)]
    chunks = []
    for i in range(n):
        meta = {**_META, "chunk_id": f"cid_{i}", "page_number": i + 1}
        chunks.append(
            {
                "chunk_text": f"Chunk text number {i}.",
                "metadata": meta,
                "score": scores[i],
                "reranker_score": 5.0 - i,
            }
        )
    return chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_pipeline() -> MagicMock:
    """Pipeline that returns two high-score chunks (above refusal threshold)."""
    m = MagicMock()
    m.retrieve.return_value = _make_chunks(2)
    return m


@pytest.fixture()
def mock_llm() -> MagicMock:
    """LLM client that returns a deterministic answer with a citation marker."""
    m = MagicMock()
    m.generate.return_value = "The dose is 25 mg [1]."
    m.model = "llama-3.3-70b-versatile"
    return m


@pytest.fixture()
def mock_store() -> MagicMock:
    """Store that returns a minimal collection-info dict."""
    m = MagicMock()
    m.get_collection_info.return_value = {
        "vectors_count": 100,
        "points_count": 100,
        "status": "green",
    }
    return m


@pytest.fixture()
def client(
    mock_pipeline: MagicMock,
    mock_llm: MagicMock,
    mock_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """TestClient with all three dependency functions overridden."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    import src.config as cfg

    cfg._settings = None

    from app.main import _get_llm_client, _get_pipeline, _get_store, app

    app.dependency_overrides[_get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[_get_llm_client] = lambda: mock_llm
    app.dependency_overrides[_get_store] = lambda: mock_store

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    cfg._settings = None


# ---------------------------------------------------------------------------
# POST /ask — happy path
# ---------------------------------------------------------------------------


def test_ask_returns_200(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "What is the dose of amitriptyline?"})
    assert resp.status_code == 200


def test_ask_response_has_answer(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "What is the dose of amitriptyline?"})
    data = resp.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert data["answer"]


def test_ask_response_has_sources(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "What is the dose of amitriptyline?"})
    data = resp.json()
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_ask_response_has_confidence(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "What is the dose of amitriptyline?"})
    data = resp.json()
    assert "confidence" in data
    assert isinstance(data["confidence"], float)


def test_ask_response_has_model_version(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "What is the dose of amitriptyline?"})
    data = resp.json()
    assert "model_version" in data
    assert isinstance(data["model_version"], str)


def test_ask_passes_question_to_pipeline(
    client: TestClient, mock_pipeline: MagicMock
) -> None:
    question = "What is the dose of amitriptyline?"
    client.post("/ask", json={"question": question})
    call_kwargs = mock_pipeline.retrieve.call_args
    assert call_kwargs.args[0] == question or call_kwargs.kwargs.get("query") == question


def test_ask_passes_filters_to_pipeline(
    client: TestClient, mock_pipeline: MagicMock
) -> None:
    filters = {"document_type": "essential_medicines_list"}
    client.post("/ask", json={"question": "What is the dose?", "filters": filters})
    assert mock_pipeline.retrieve.call_args.kwargs["filters"] == filters


def test_ask_sources_populated_from_citations(client: TestClient) -> None:
    """[1] in mock LLM answer maps to one CitationSource in sources."""
    resp = client.post("/ask", json={"question": "What is the dose of amitriptyline?"})
    data = resp.json()
    assert len(data["sources"]) == 1
    src = data["sources"][0]
    assert "claim" in src
    assert "source_chunk" in src
    assert "page_number" in src
    assert "source_document" in src


def test_ask_confidence_is_max_score(client: TestClient) -> None:
    """confidence == max cosine score across retrieved chunks (0.9)."""
    resp = client.post("/ask", json={"question": "What is the dose of amitriptyline?"})
    data = resp.json()
    assert abs(data["confidence"] - 0.9) < 1e-6


# ---------------------------------------------------------------------------
# POST /ask — refusal path
# ---------------------------------------------------------------------------


@pytest.fixture()
def refusal_client(
    mock_llm: MagicMock,
    mock_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Client whose pipeline returns chunks with scores below the threshold."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    import src.config as cfg

    cfg._settings = None

    refusing_pipeline = MagicMock()
    refusing_pipeline.retrieve.return_value = []  # empty → should_refuse returns True

    from app.main import _get_llm_client, _get_pipeline, _get_store, app

    app.dependency_overrides[_get_pipeline] = lambda: refusing_pipeline
    app.dependency_overrides[_get_llm_client] = lambda: mock_llm
    app.dependency_overrides[_get_store] = lambda: mock_store

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    cfg._settings = None


def test_ask_refusal_returns_200(refusal_client: TestClient) -> None:
    resp = refusal_client.post("/ask", json={"question": "What is the dose?"})
    assert resp.status_code == 200


def test_ask_refusal_answer_text(refusal_client: TestClient) -> None:
    resp = refusal_client.post("/ask", json={"question": "What is the dose?"})
    assert resp.json()["answer"] == "I cannot answer from the provided documents."


def test_ask_refusal_empty_sources(refusal_client: TestClient) -> None:
    resp = refusal_client.post("/ask", json={"question": "What is the dose?"})
    assert resp.json()["sources"] == []


def test_ask_refusal_confidence_zero(refusal_client: TestClient) -> None:
    """Confidence is 0.0 when no chunks are retrieved."""
    resp = refusal_client.post("/ask", json={"question": "What is the dose?"})
    assert resp.json()["confidence"] == 0.0


# ---------------------------------------------------------------------------
# POST /ask — error paths
# ---------------------------------------------------------------------------


def test_ask_empty_question_returns_400(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 400


def test_ask_whitespace_question_returns_400(client: TestClient) -> None:
    resp = client.post("/ask", json={"question": "   "})
    assert resp.status_code == 400


def test_ask_pipeline_connection_error_returns_503(
    mock_llm: MagicMock,
    mock_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    import src.config as cfg

    cfg._settings = None

    failing_pipeline = MagicMock()
    failing_pipeline.retrieve.side_effect = ConnectionError("Qdrant down")

    from app.main import _get_llm_client, _get_pipeline, _get_store, app

    app.dependency_overrides[_get_pipeline] = lambda: failing_pipeline
    app.dependency_overrides[_get_llm_client] = lambda: mock_llm
    app.dependency_overrides[_get_store] = lambda: mock_store

    with TestClient(app) as c:
        resp = c.post("/ask", json={"question": "What is the dose?"})

    app.dependency_overrides.clear()
    cfg._settings = None

    assert resp.status_code == 503


def test_ask_pipeline_unexpected_error_returns_500(
    mock_llm: MagicMock,
    mock_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    import src.config as cfg

    cfg._settings = None

    crashing_pipeline = MagicMock()
    crashing_pipeline.retrieve.side_effect = RuntimeError("unexpected crash")

    from app.main import _get_llm_client, _get_pipeline, _get_store, app

    app.dependency_overrides[_get_pipeline] = lambda: crashing_pipeline
    app.dependency_overrides[_get_llm_client] = lambda: mock_llm
    app.dependency_overrides[_get_store] = lambda: mock_store

    with TestClient(app) as c:
        resp = c.post("/ask", json={"question": "What is the dose?"})

    app.dependency_overrides.clear()
    cfg._settings = None

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_status_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.json()["status"] == "ok"


def test_health_model_version(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    resp = client.get("/health")
    data = resp.json()
    assert "model_version" in data
    assert isinstance(data["model_version"], str)


def test_health_collection_info_present(client: TestClient) -> None:
    resp = client.get("/health")
    data = resp.json()
    assert "collection_info" in data
    assert isinstance(data["collection_info"], dict)


def test_health_degraded_when_store_unavailable(
    mock_pipeline: MagicMock,
    mock_llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    import src.config as cfg

    cfg._settings = None

    degraded_store = MagicMock()
    degraded_store.get_collection_info.side_effect = ConnectionError("Qdrant down")

    from app.main import _get_llm_client, _get_pipeline, _get_store, app

    app.dependency_overrides[_get_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[_get_llm_client] = lambda: mock_llm
    app.dependency_overrides[_get_store] = lambda: degraded_store

    with TestClient(app) as c:
        resp = c.get("/health")

    app.dependency_overrides.clear()
    cfg._settings = None

    assert resp.status_code == 200
    assert resp.json()["status"] == "degraded"
