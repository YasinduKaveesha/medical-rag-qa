"""Tests for src.embeddings.clip_encoder — Module 3: CLIP Encoder.

Fast tests bypass CLIPEncoder.__init__ (using __new__ + direct attribute
injection) so no CLIP download or transformers CLIP import is required.
The single @pytest.mark.slow test loads the real model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.embeddings.clip_encoder import CLIPEncoder


# ---------------------------------------------------------------------------
# Helpers — mock construction bypasses __init__
# ---------------------------------------------------------------------------

_DIM = 512


def _fake_features(value: float = 0.5) -> MagicMock:
    """Return a mock tensor whose [0].cpu().numpy() gives a (512,) float32 array."""
    raw = np.full((_DIM,), value, dtype=np.float32)
    tensor_item = MagicMock()
    tensor_item.cpu.return_value.numpy.return_value = raw
    features = MagicMock()
    features.__getitem__ = lambda self, idx: tensor_item
    return features


def _build_encoder(img_value: float = 0.5, txt_value: float = 0.5) -> CLIPEncoder:
    """Build a CLIPEncoder with mocked internals — no __init__, no download."""
    proc = MagicMock()
    # processor(images/text=..., return_tensors="pt", ...) -> dict of tensors
    proc.return_value = {"pixel_values": MagicMock(), "input_ids": MagicMock()}

    model = MagicMock()
    model.get_image_features.return_value = _fake_features(img_value)
    model.get_text_features.return_value = _fake_features(txt_value)
    model.to.return_value = model

    encoder = CLIPEncoder.__new__(CLIPEncoder)
    encoder._model_name = "openai/clip-vit-base-patch32"
    encoder._device = "cpu"
    encoder._processor = proc
    encoder._model = model
    return encoder


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_encode_image_returns_512_dim():
    """encode_image returns a (512,) array."""
    enc = _build_encoder()
    img = Image.new("RGB", (224, 224))
    result = enc.encode_image(img)
    assert result.shape == (_DIM,)


def test_encode_text_returns_512_dim():
    """encode_text returns a (512,) array."""
    enc = _build_encoder()
    result = enc.encode_text("chest X-ray with bilateral infiltrates")
    assert result.shape == (_DIM,)


def test_encode_image_from_pil():
    """encode_image accepts a PIL Image directly."""
    enc = _build_encoder()
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    result = enc.encode_image(img)
    assert isinstance(result, np.ndarray)


def test_encode_image_from_path(tmp_path):
    """encode_image accepts a file-path string."""
    enc = _build_encoder()
    img = Image.new("RGB", (100, 100))
    path = str(tmp_path / "test.png")
    img.save(path)
    result = enc.encode_image(path)
    assert isinstance(result, np.ndarray)
    assert result.shape == (_DIM,)


def test_image_embeddings_normalized():
    """encode_image returns an L2-normalised vector (norm ≈ 1.0)."""
    enc = _build_encoder(img_value=3.0)
    img = Image.new("RGB", (224, 224))
    result = enc.encode_image(img)
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-5, f"norm={norm}"


def test_text_embeddings_normalized():
    """encode_text returns an L2-normalised vector (norm ≈ 1.0)."""
    enc = _build_encoder(txt_value=2.0)
    result = enc.encode_text("pulmonary oedema management")
    norm = float(np.linalg.norm(result))
    assert abs(norm - 1.0) < 1e-5, f"norm={norm}"


def test_compute_similarity_range():
    """compute_similarity returns a float in [-1, 1]."""
    enc = _build_encoder()
    a = np.random.randn(_DIM).astype(np.float32)
    a /= np.linalg.norm(a)
    b = np.random.randn(_DIM).astype(np.float32)
    b /= np.linalg.norm(b)
    sim = enc.compute_similarity(a, b)
    assert isinstance(sim, float)
    assert -1.0 <= sim <= 1.0


def test_compute_similarity_identical():
    """compute_similarity of a vector with itself is ≈ 1.0."""
    enc = _build_encoder()
    v = np.random.randn(_DIM).astype(np.float32)
    v /= np.linalg.norm(v)
    sim = enc.compute_similarity(v, v)
    assert abs(sim - 1.0) < 1e-5, f"sim={sim}"


def test_encode_images_batch():
    """encode_images_batch returns a list of numpy arrays."""
    enc = _build_encoder()
    images = [Image.new("RGB", (100, 100)) for _ in range(4)]
    results = enc.encode_images_batch(images, batch_size=2)
    assert isinstance(results, list)
    assert all(isinstance(r, np.ndarray) for r in results)


def test_encode_texts_batch():
    """encode_texts_batch returns a list of numpy arrays."""
    enc = _build_encoder()
    texts = ["query one", "query two", "query three"]
    results = enc.encode_texts_batch(texts, batch_size=2)
    assert isinstance(results, list)
    assert all(isinstance(r, np.ndarray) for r in results)


def test_batch_correct_count():
    """Both batch methods return exactly one embedding per input."""
    enc = _build_encoder()
    images = [Image.new("RGB", (100, 100)) for _ in range(5)]
    texts = ["t1", "t2", "t3", "t4", "t5"]
    assert len(enc.encode_images_batch(images, batch_size=2)) == 5
    assert len(enc.encode_texts_batch(texts, batch_size=2)) == 5


@pytest.mark.slow
def test_clip_integration(sample_image):
    """Real CLIP model produces a normalised (512,) embedding for an image."""
    enc = CLIPEncoder(model_name="openai/clip-vit-base-patch32", device="cpu")
    img_vec = enc.encode_image(sample_image)
    assert img_vec.shape == (_DIM,)
    assert abs(float(np.linalg.norm(img_vec)) - 1.0) < 1e-5

    txt_vec = enc.encode_text("a medical diagram")
    assert txt_vec.shape == (_DIM,)
    sim = enc.compute_similarity(img_vec, txt_vec)
    assert -1.0 <= sim <= 1.0
