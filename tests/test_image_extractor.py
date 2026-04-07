"""Tests for src.ingestion.image_extractor — Module 1: Image Extraction."""

from __future__ import annotations

import io
import os
from dataclasses import fields
from pathlib import Path

import pytest
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.ingestion.image_extractor import ExtractedImage, ImageExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_text_only_pdf(path: str) -> str:
    """Create a single-page PDF with only text (no images)."""
    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(72, 750, "This PDF has no embedded images.")
    c.save()
    return path


def _make_tiny_image_pdf(path: str, tmp_path: Path) -> str:
    """Create a PDF with a single tiny (10x10) embedded image."""
    from reportlab.lib.utils import ImageReader

    tiny = Image.new("RGB", (10, 10), color=(128, 128, 128))
    png = str(tmp_path / "tiny.png")
    tiny.save(png)

    c = canvas.Canvas(path, pagesize=A4)
    c.drawImage(ImageReader(png), 72, 700, width=10, height=10)
    c.save()
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extracted_image_dataclass_fields():
    """ExtractedImage has all required fields with correct names."""
    field_names = {f.name for f in fields(ExtractedImage)}
    assert field_names == {
        "image_path",
        "source_pdf",
        "page_number",
        "xref",
        "width",
        "height",
        "image_id",
    }


def test_extract_creates_output_dir(tmp_path):
    """ImageExtractor creates the output directory on init."""
    new_dir = str(tmp_path / "new_subdir" / "images")
    assert not os.path.exists(new_dir)
    ImageExtractor(output_dir=new_dir)
    assert os.path.isdir(new_dir)


def test_extract_from_pdf_with_images(tmp_path, sample_pdf_with_images):
    """extract_images_from_pdf returns at least one ExtractedImage for a PDF with images."""
    extractor = ImageExtractor(output_dir=str(tmp_path / "out"))
    results = extractor.extract_images_from_pdf(sample_pdf_with_images)
    assert len(results) >= 1
    assert all(isinstance(r, ExtractedImage) for r in results)


def test_extract_returns_correct_metadata(tmp_path, sample_pdf_with_images):
    """Extracted images have correct source_pdf and page_number metadata."""
    extractor = ImageExtractor(output_dir=str(tmp_path / "out"))
    results = extractor.extract_images_from_pdf(sample_pdf_with_images)
    assert len(results) >= 1
    img = results[0]
    assert img.source_pdf == Path(sample_pdf_with_images).name
    assert img.page_number >= 1
    assert img.xref > 0
    assert img.width >= 50
    assert img.height >= 50


def test_extract_skips_tiny_images(tmp_path):
    """Images smaller than 50x50 are not returned."""
    pdf_path = str(tmp_path / "tiny_image.pdf")
    _make_tiny_image_pdf(pdf_path, tmp_path)
    extractor = ImageExtractor(output_dir=str(tmp_path / "out"))
    results = extractor.extract_images_from_pdf(pdf_path)
    # Tiny 10x10 image must be filtered out
    assert all(r.width >= 50 and r.height >= 50 for r in results)


def test_extract_handles_no_images(tmp_path):
    """extract_images_from_pdf returns empty list for text-only PDF."""
    pdf_path = str(tmp_path / "text_only.pdf")
    _make_text_only_pdf(pdf_path)
    extractor = ImageExtractor(output_dir=str(tmp_path / "out"))
    results = extractor.extract_images_from_pdf(pdf_path)
    assert results == []


def test_extract_handles_missing_pdf(tmp_path):
    """extract_images_from_pdf raises FileNotFoundError for non-existent PDF."""
    extractor = ImageExtractor(output_dir=str(tmp_path / "out"))
    with pytest.raises(FileNotFoundError):
        extractor.extract_images_from_pdf(str(tmp_path / "does_not_exist.pdf"))


def test_extract_deduplicates_xrefs(tmp_path, sample_pdf_with_images):
    """The same xref appearing on multiple pages is only returned once."""
    extractor = ImageExtractor(output_dir=str(tmp_path / "out"))
    results = extractor.extract_images_from_pdf(sample_pdf_with_images)
    xrefs = [r.xref for r in results]
    assert len(xrefs) == len(set(xrefs)), "Duplicate xrefs found in results"


def test_save_image_creates_file(tmp_path, sample_image):
    """_save_image writes a PNG file to the output directory."""
    out_dir = str(tmp_path / "out")
    extractor = ImageExtractor(output_dir=out_dir)

    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    saved = extractor._save_image(image_bytes, xref=99, pdf_path="test_doc.pdf", page_num=1)
    assert os.path.isfile(saved)


def test_save_image_naming_convention(tmp_path, sample_image):
    """_save_image uses the {stem}_p{page}_x{xref}.png naming scheme."""
    out_dir = str(tmp_path / "out")
    extractor = ImageExtractor(output_dir=out_dir)

    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    saved = extractor._save_image(image_bytes, xref=7, pdf_path="my_report.pdf", page_num=3)
    assert Path(saved).name == "my_report_p3_x7.png"


def test_is_valid_image_accepts_large(sample_image):
    """_is_valid_image returns True for a 200x200 image."""
    extractor = ImageExtractor(output_dir="/tmp")
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    assert extractor._is_valid_image(buf.getvalue()) is True


def test_is_valid_image_rejects_small():
    """_is_valid_image returns False for a 10x10 image."""
    extractor = ImageExtractor(output_dir="/tmp")
    tiny = Image.new("RGB", (10, 10), color=(0, 0, 0))
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    assert extractor._is_valid_image(buf.getvalue()) is False


def test_is_valid_image_rejects_corrupted():
    """_is_valid_image returns False for non-image bytes."""
    extractor = ImageExtractor(output_dir="/tmp")
    assert extractor._is_valid_image(b"not an image at all") is False
