"""Tests for src.ingestion.image_captioner — Module 2: Image Captioning.

Fast tests bypass ImageCaptioner.__init__ entirely (using __new__) so that no
BLIP model download or transformers BLIP import occurs.  The single
@pytest.mark.slow test loads the real model and requires internet access.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.ingestion.image_captioner import CaptionedImage, ImageCaptioner
from src.ingestion.image_extractor import ExtractedImage


# ---------------------------------------------------------------------------
# Helpers — mock construction bypasses __init__
# ---------------------------------------------------------------------------


def _make_mock_processor(decoded_text: str) -> MagicMock:
    """Mock BlipProcessor: decode() returns *decoded_text*."""
    proc = MagicMock()
    # proc(images=..., return_tensors="pt") returns a dict-like object
    proc.return_value = {"pixel_values": MagicMock()}
    proc.decode.return_value = decoded_text
    return proc


def _make_mock_model() -> MagicMock:
    """Mock BlipForConditionalGeneration: generate() returns fake token ids."""
    model = MagicMock()
    model.generate.return_value = [[101, 102, 103]]
    model.to.return_value = model
    return model


def _build_captioner(decoded_text: str = "a chest x-ray image") -> ImageCaptioner:
    """Build an ImageCaptioner with mocked internals — no __init__, no download."""
    proc = _make_mock_processor(decoded_text)
    model = _make_mock_model()

    captioner = ImageCaptioner.__new__(ImageCaptioner)
    captioner._model_name = "Salesforce/blip-image-captioning-base"
    captioner._device = "cpu"
    captioner._processor = proc
    captioner._model = model
    return captioner


def _sample_extracted_image(tmp_path: Path, idx: int = 1) -> ExtractedImage:
    """Create a real PNG on disk and return a matching ExtractedImage."""
    img = Image.new("RGB", (100, 100), color=(idx * 50 % 256, 0, 0))
    path = str(tmp_path / f"img_{idx}.png")
    img.save(path)
    return ExtractedImage(
        image_path=path,
        source_pdf="test_doc.pdf",
        page_number=idx,
        xref=idx,
        width=100,
        height=100,
        image_id=f"test_doc_p{idx}_x{idx}",
    )


# ---------------------------------------------------------------------------
# Fast tests (mocked model)
# ---------------------------------------------------------------------------


def test_caption_single_image():
    """caption_image returns a string for a PIL Image input (mocked model)."""
    captioner = _build_captioner("a lung diagram")
    img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    result = captioner.caption_image(img)
    assert isinstance(result, str)


def test_caption_returns_nonempty_string():
    """caption_image returns a non-empty string (mocked model)."""
    captioner = _build_captioner("a medical illustration")
    img = Image.new("RGB", (200, 200))
    result = captioner.caption_image(img)
    assert len(result) > 0


def test_caption_cleans_blip_artifacts():
    """'arafed' prefix returned by BLIP is stripped from the caption."""
    captioner = _build_captioner("arafed a cat sitting on a table")
    img = Image.new("RGB", (200, 200))
    result = captioner.caption_image(img)
    assert not result.lower().startswith("arafed"), f"Got: {result!r}"
    assert "a cat sitting on a table" in result


def test_caption_from_filepath(tmp_path):
    """caption_image accepts a file-path string and loads the image."""
    captioner = _build_captioner("an x-ray of the thorax")
    img = Image.new("RGB", (150, 150), color=(200, 200, 200))
    path = str(tmp_path / "test.png")
    img.save(path)
    result = captioner.caption_image(path)
    assert isinstance(result, str)
    assert len(result) > 0


def test_caption_from_pil_image():
    """caption_image works when passed a PIL Image directly."""
    captioner = _build_captioner("a flowchart of treatment steps")
    img = Image.new("RGB", (100, 100))
    result = captioner.caption_image(img)
    assert isinstance(result, str)


def test_caption_batch_returns_list():
    """caption_batch returns a list."""
    captioner = _build_captioner("a diagram")
    images = [Image.new("RGB", (100, 100)) for _ in range(4)]
    results = captioner.caption_batch(images, batch_size=2)
    assert isinstance(results, list)


def test_caption_batch_correct_count():
    """caption_batch returns one caption per input image."""
    captioner = _build_captioner("a diagram")
    images = [Image.new("RGB", (100, 100)) for _ in range(5)]
    results = captioner.caption_batch(images, batch_size=2)
    assert len(results) == 5


def test_caption_extracted_images_returns_captioned(tmp_path):
    """caption_extracted_images returns CaptionedImage objects."""
    captioner = _build_captioner("an anatomical illustration")
    ext_imgs = [_sample_extracted_image(tmp_path, i) for i in range(1, 4)]
    results = captioner.caption_extracted_images(ext_imgs)
    assert all(isinstance(r, CaptionedImage) for r in results)
    assert len(results) == 3


def test_captioned_image_has_all_fields(tmp_path):
    """CaptionedImage carries all ExtractedImage fields + caption + caption_model."""
    captioner = _build_captioner("blood cell diagram")
    ext_img = _sample_extracted_image(tmp_path, idx=1)
    result = captioner.caption_extracted_images([ext_img])[0]

    assert result.image_path == ext_img.image_path
    assert result.source_pdf == ext_img.source_pdf
    assert result.page_number == ext_img.page_number
    assert result.xref == ext_img.xref
    assert result.width == ext_img.width
    assert result.height == ext_img.height
    assert result.image_id == ext_img.image_id
    assert isinstance(result.caption, str) and len(result.caption) > 0
    assert isinstance(result.caption_model, str) and len(result.caption_model) > 0


# ---------------------------------------------------------------------------
# Slow test — real BLIP model (skipped by default with -k "not slow")
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_caption_integration(sample_image):
    """Real BLIP model produces a non-empty, 'arafed'-free caption."""
    captioner = ImageCaptioner(
        model_name="Salesforce/blip-image-captioning-base", device="cpu"
    )
    result = captioner.caption_image(sample_image)
    assert isinstance(result, str)
    assert len(result) > 0
    assert not result.lower().startswith("arafed"), f"Artifact not cleaned: {result!r}"
