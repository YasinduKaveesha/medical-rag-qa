import matplotlib

matplotlib.use("Agg")


import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def sample_image() -> Image.Image:
    """200x200 RGB PIL Image with coloured rectangles and circles for testing."""
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Rectangle
    draw.rectangle([20, 20, 80, 80], fill=(255, 0, 0), outline=(0, 0, 0))
    # Circle (ellipse)
    draw.ellipse([100, 20, 180, 100], fill=(0, 0, 255), outline=(0, 0, 0))
    # Second rectangle
    draw.rectangle([20, 110, 180, 170], fill=(0, 200, 0), outline=(0, 0, 0))
    return img


@pytest.fixture
def sample_pdf_with_images(tmp_path, sample_image) -> str:
    """Reportlab-generated 1-page PDF containing text and one embedded PNG image."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    # Save the PIL image to a temporary PNG
    png_path = str(tmp_path / "test_image.png")
    sample_image.save(png_path)

    pdf_path = str(tmp_path / "test_document.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)
    c.setFont("Helvetica", 12)
    c.drawString(72, 750, "Test medical document page 1")
    c.drawString(72, 730, "Amoxicillin 500mg three times daily for 7 days.")
    # Embed the image
    c.drawImage(ImageReader(png_path), 72, 500, width=150, height=150)
    c.save()
    return pdf_path


@pytest.fixture
def sample_text_chunks() -> list:
    """Three text chunk dicts for use in multimodal pipeline tests."""
    return [
        {
            "text": "Amoxicillin 500mg three times daily for 7 days",
            "source_document": "test_doc.pdf",
            "page_number": 1,
            "section_title": "Treatment",
            "chunk_id": "chunk_001",
            "score": 0.85,
        },
        {
            "text": "Chest X-ray shows bilateral infiltrates",
            "source_document": "test_doc.pdf",
            "page_number": 2,
            "section_title": "Diagnosis",
            "chunk_id": "chunk_002",
            "score": 0.72,
        },
        {
            "text": "Monitor blood glucose levels every 4 hours",
            "source_document": "test_doc.pdf",
            "page_number": 3,
            "section_title": "Monitoring",
            "chunk_id": "chunk_003",
            "score": 0.65,
        },
    ]


@pytest.fixture
def temp_image_dir(tmp_path) -> str:
    """Temporary directory path for extracted images."""
    d = tmp_path / "extracted_images"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@pytest.fixture
def in_memory_qdrant():
    """In-memory QdrantClient for fast vector store tests (no Docker needed)."""
    from qdrant_client import QdrantClient

    return QdrantClient(":memory:")
