"""Tests for Phase 2 FastAPI endpoints — POST /ask-multimodal and GET /images/{image_id}.

All ML dependencies are mocked via FastAPI dependency_overrides.  No real
models, Qdrant, or LLM calls are made.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from src.retrieval.multimodal_pipeline import RetrievalResult


# ---------------------------------------------------------------------------
# Sample data helpers
# ---------------------------------------------------------------------------


def _make_text_chunk(chunk_id: str = "c1", score: float = 0.9) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": "Amoxicillin 500mg three times daily for 7 days",
        "type": "text",
        "source_document": "who_guidelines.pdf",
        "page_number": 5,
        "score": score,
        "rrf_score": 0.03,
        "reranker_score": 8.0,
    }


def _make_image(image_id: str = "doc_p1_x5") -> dict:
    return {
        "image_id": image_id,
        "image_path": f"data/extracted_images/{image_id}.png",
        "caption": "chest X-ray showing bilateral infiltrates",
        "source_pdf": "radiology_atlas.pdf",
        "page_number": 3,
        "type": "image",
        "score": 0.8,
        "rrf_score": 0.025,
    }


def _make_retrieval_result(
    text_chunks: list | None = None,
    images: list | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        text_chunks=text_chunks if text_chunks is not None else [_make_text_chunk()],
        images=images if images is not None else [_make_image()],
        fusion_scores={},
        retrieval_time_ms=42.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_mm_pipeline() -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = _make_retrieval_result()
    return m


@pytest.fixture()
def mock_llm() -> MagicMock:
    m = MagicMock()
    m.generate.return_value = "The recommended treatment is amoxicillin."
    m.generate_with_vision.return_value = "Vision answer: bilateral infiltrates visible."
    m.model = "llama-3.3-70b-versatile"
    return m


@pytest.fixture()
def client(
    mock_mm_pipeline: MagicMock,
    mock_llm: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """TestClient with multimodal + existing dependencies overridden."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")

    import src.config as cfg
    cfg._settings = None

    from app.main import (
        _get_llm_client,
        _get_multimodal_pipeline,
        _get_pipeline,
        _get_store,
        app,
    )

    # Override multimodal deps
    app.dependency_overrides[_get_multimodal_pipeline] = lambda: mock_mm_pipeline
    app.dependency_overrides[_get_llm_client] = lambda: mock_llm

    # Override P1 deps so /ask and /health still work
    mock_p1_pipeline = MagicMock()
    mock_p1_pipeline.retrieve.return_value = []
    mock_store = MagicMock()
    mock_store.get_collection_info.return_value = {"vectors_count": 0, "points_count": 0, "status": "green"}
    app.dependency_overrides[_get_pipeline] = lambda: mock_p1_pipeline
    app.dependency_overrides[_get_store] = lambda: mock_store

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()
    cfg._settings = None


# ---------------------------------------------------------------------------
# POST /ask-multimodal
# ---------------------------------------------------------------------------


def test_ask_multimodal_returns_200(client: TestClient) -> None:
    resp = client.post("/ask-multimodal", json={"question": "What does the chest X-ray show?"})
    assert resp.status_code == 200


def test_ask_multimodal_response_has_fields(client: TestClient) -> None:
    resp = client.post("/ask-multimodal", json={"question": "What does the chest X-ray show?"})
    data = resp.json()
    for field in ("answer", "text_sources", "image_sources", "used_vision_model", "retrieval_time_ms", "model_version"):
        assert field in data, f"Missing field: {field}"
    assert isinstance(data["answer"], str)
    assert isinstance(data["image_sources"], list)
    assert isinstance(data["used_vision_model"], bool)
    assert isinstance(data["retrieval_time_ms"], float)


def test_ask_multimodal_with_vision(client: TestClient, mock_llm: MagicMock) -> None:
    """When use_vision=True and images exist, generate_with_vision is called."""
    resp = client.post(
        "/ask-multimodal",
        json={"question": "Describe the X-ray findings", "use_vision": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["used_vision_model"] is True
    mock_llm.generate_with_vision.assert_called_once()


def test_ask_multimodal_without_images(
    client: TestClient,
    mock_mm_pipeline: MagicMock,
    mock_llm: MagicMock,
) -> None:
    """When include_images=False, image_sources is empty and generate() is used."""
    mock_mm_pipeline.retrieve.return_value = _make_retrieval_result(
        text_chunks=[_make_text_chunk()],
        images=[_make_image()],
    )
    resp = client.post(
        "/ask-multimodal",
        json={"question": "What is the dose?", "include_images": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["image_sources"] == []
    assert data["used_vision_model"] is False
    mock_llm.generate.assert_called_once()


# ---------------------------------------------------------------------------
# GET /images/{image_id}
# ---------------------------------------------------------------------------


def test_get_image_returns_file(client: TestClient, tmp_path, monkeypatch) -> None:
    """GET /images/{id} returns 200 and image/png when the file exists."""
    from PIL import Image

    img = Image.new("RGB", (50, 50), color=(100, 100, 200))
    img_dir = tmp_path / "extracted_images"
    img_dir.mkdir()
    (img_dir / "test_p1_x5.png").write_bytes(b"")
    img.save(str(img_dir / "test_p1_x5.png"))

    import src.config as cfg
    cfg._settings = None
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("EXTRACTED_IMAGES_DIR", str(img_dir))
    cfg._settings = None  # force reload with new env

    resp = client.get("/images/test_p1_x5")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/png")


def test_get_image_404(client: TestClient, monkeypatch) -> None:
    """GET /images/{id} returns 404 when the image does not exist."""
    import src.config as cfg
    monkeypatch.setenv("EXTRACTED_IMAGES_DIR", "/nonexistent/path")
    cfg._settings = None

    resp = client.get("/images/no_such_image")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Backward compatibility — existing endpoints still work
# ---------------------------------------------------------------------------


def test_existing_ask_still_works(client: TestClient) -> None:
    """POST /ask returns 200 (P1 backward compatibility)."""
    resp = client.post("/ask", json={"question": "What is amoxicillin?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data


def test_existing_health_still_works(client: TestClient) -> None:
    """GET /health returns 200 with status field (P1 backward compatibility)."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "model_version" in data
