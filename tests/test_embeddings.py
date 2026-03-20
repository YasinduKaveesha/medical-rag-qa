"""Tests for src.embeddings.encoder — EmbeddingEncoder and get_encoder()."""

from __future__ import annotations

import numpy as np
import pytest

import src.embeddings.encoder as enc_module
from src.embeddings.encoder import EmbeddingEncoder, get_encoder  # noqa: E402

# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockSentenceTransformer:
    """Minimal SentenceTransformer stand-in that never loads real weights."""

    DIM = 384

    def encode(
        self,
        input: str | list[str],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        if isinstance(input, str):
            return np.zeros(self.DIM, dtype=np.float32)
        return np.zeros((len(input), self.DIM), dtype=np.float32)

    def get_sentence_embedding_dimension(self) -> int:
        return self.DIM


@pytest.fixture
def encoder() -> EmbeddingEncoder:
    """EmbeddingEncoder backed by the mock model."""
    return EmbeddingEncoder(_model=MockSentenceTransformer())


# ---------------------------------------------------------------------------
# encode() tests
# ---------------------------------------------------------------------------


def test_encode_returns_ndarray(encoder):
    result = encoder.encode("What is the recommended dose of amitriptyline?")
    assert isinstance(result, np.ndarray)


def test_encode_shape(encoder):
    result = encoder.encode("Metformin is used in type 2 diabetes management.")
    assert result.shape == (384,)


def test_encode_dtype(encoder):
    result = encoder.encode("Some medical text.")
    assert result.dtype == np.float32


def test_encode_empty_string(encoder):
    """Empty string must not raise — model returns a valid zero-ish vector."""
    result = encoder.encode("")
    assert isinstance(result, np.ndarray)
    assert result.shape == (384,)


# ---------------------------------------------------------------------------
# encode_batch() tests
# ---------------------------------------------------------------------------


def test_encode_batch_returns_list(encoder):
    result = encoder.encode_batch(["text one", "text two"])
    assert isinstance(result, list)


def test_encode_batch_length(encoder):
    texts = ["sentence one", "sentence two", "sentence three"]
    result = encoder.encode_batch(texts)
    assert len(result) == len(texts)


def test_encode_batch_each_is_ndarray(encoder):
    result = encoder.encode_batch(["a", "b", "c"])
    for item in result:
        assert isinstance(item, np.ndarray)


def test_encode_batch_each_shape(encoder):
    result = encoder.encode_batch(["first text", "second text"])
    for item in result:
        assert item.shape == (384,)


def test_encode_batch_each_dtype(encoder):
    result = encoder.encode_batch(["text"])
    assert result[0].dtype == np.float32


def test_encode_batch_empty_input(encoder):
    assert encoder.encode_batch([]) == []


def test_encode_batch_single_item(encoder):
    result = encoder.encode_batch(["only one"])
    assert len(result) == 1
    assert result[0].shape == (384,)


def test_encode_batch_respects_batch_size():
    """With 10 texts and batch_size=3, model.encode must be called 4 times."""

    class CountingMock(MockSentenceTransformer):
        def __init__(self):
            self.call_count = 0

        def encode(self, input, show_progress_bar=False, convert_to_numpy=True):
            self.call_count += 1
            return super().encode(input, show_progress_bar, convert_to_numpy)

    mock = CountingMock()
    enc = EmbeddingEncoder(_model=mock)
    texts = [f"text {i}" for i in range(10)]
    enc.encode_batch(texts, batch_size=3)
    # ceil(10 / 3) = 4 batches
    assert mock.call_count == 4


def test_encode_batch_order_preserved(encoder):
    """Result list must be in the same order as the input list."""
    # Use a mock that returns distinct vectors per call to verify ordering
    class IndexedMock(MockSentenceTransformer):
        def __init__(self):
            self._call = 0

        def encode(self, input, show_progress_bar=False, convert_to_numpy=True):
            n = 1 if isinstance(input, str) else len(input)
            out = np.full((n, self.DIM), self._call, dtype=np.float32)
            self._call += 1
            return out

    enc = EmbeddingEncoder(_model=IndexedMock())
    result = enc.encode_batch(["a", "b", "c", "d"], batch_size=2)
    # batch 0 → value 0.0, batch 1 → value 1.0
    assert result[0][0] == 0.0
    assert result[2][0] == 1.0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_embedding_dim_property(encoder):
    assert encoder.embedding_dim == 384


def test_model_name_property():
    enc = EmbeddingEncoder(model_name="test-model", _model=MockSentenceTransformer())
    assert enc.model_name == "test-model"


# ---------------------------------------------------------------------------
# get_encoder() singleton tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch):
    """Reset both singletons before and after every test in this module."""
    import src.config as cfg

    cfg._settings = None
    enc_module._encoder = None
    yield
    cfg._settings = None
    enc_module._encoder = None


def _patch_encoder(monkeypatch) -> None:
    """Monkeypatch EmbeddingEncoder so get_encoder() never loads real weights."""
    monkeypatch.setattr(
        enc_module,
        "EmbeddingEncoder",
        lambda model_name: EmbeddingEncoder(
            model_name=model_name,
            _model=MockSentenceTransformer(),
        ),
    )


def test_get_encoder_returns_instance(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_encoder(monkeypatch)
    result = get_encoder()
    assert isinstance(result, EmbeddingEncoder)


def test_get_encoder_singleton(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_encoder(monkeypatch)
    e1 = get_encoder()
    e2 = get_encoder()
    assert e1 is e2


def test_get_encoder_reset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    _patch_encoder(monkeypatch)
    e1 = get_encoder()
    enc_module._encoder = None
    e2 = get_encoder()
    assert e1 is not e2


def test_get_encoder_uses_settings_model_name(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    captured: list[str] = []

    def fake_encoder(model_name: str) -> EmbeddingEncoder:
        captured.append(model_name)
        return EmbeddingEncoder(model_name=model_name, _model=MockSentenceTransformer())

    monkeypatch.setattr(enc_module, "EmbeddingEncoder", fake_encoder)
    get_encoder()
    assert captured[0] == "sentence-transformers/all-MiniLM-L6-v2"
