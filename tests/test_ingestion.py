"""Tests for src.ingestion.pdf_parser and src.ingestion.metadata.

All tests use real PDFs from data/raw/.  The primary fixture is
WHO-MHP-HPS-EML-2023.02-eng.pdf (71 pages, has tables, fast to parse).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.metadata import _infer_document_type, attach_metadata
from src.ingestion.pdf_parser import parse_pdf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
WHO_EML = DATA_DIR / "WHO-MHP-HPS-EML-2023.02-eng.pdf"
WHO_CLM = DATA_DIR / "9789240081888-eng.pdf"
IDF_PDF = DATA_DIR / "IDF_Rec_2025.pdf"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def who_eml_pages():
    """Parsed pages from WHO-EML — parsed once for the whole module."""
    return parse_pdf(WHO_EML)


@pytest.fixture(scope="module")
def who_eml_with_metadata(who_eml_pages):
    """WHO-EML pages with metadata attached."""
    return attach_metadata(
        who_eml_pages,
        source_document=WHO_EML.name,
        pdf_path=str(WHO_EML),
    )


# ---------------------------------------------------------------------------
# pdf_parser tests
# ---------------------------------------------------------------------------


def test_parse_pdf_returns_list(who_eml_pages):
    """parse_pdf returns a list."""
    assert isinstance(who_eml_pages, list)


def test_parse_pdf_non_empty(who_eml_pages):
    """parse_pdf returns at least one page."""
    assert len(who_eml_pages) > 0


def test_parse_pdf_expected_keys(who_eml_pages):
    """Every page dict has the required keys."""
    required = {"page_number", "text", "tables"}
    for page in who_eml_pages:
        assert required.issubset(page.keys()), f"Missing keys on page {page.get('page_number')}"


def test_parse_pdf_no_blank_pages(who_eml_pages):
    """No returned page has empty text."""
    for page in who_eml_pages:
        assert page["text"].strip(), f"Blank text on page {page['page_number']}"


def test_parse_pdf_page_numbers_are_positive(who_eml_pages):
    """All page numbers are positive integers."""
    for page in who_eml_pages:
        assert isinstance(page["page_number"], int)
        assert page["page_number"] >= 1


def test_parse_pdf_page_numbers_are_ascending(who_eml_pages):
    """Page numbers are strictly ascending."""
    nums = [p["page_number"] for p in who_eml_pages]
    assert nums == sorted(nums)


def test_parse_pdf_page_count_within_total(who_eml_pages):
    """Returned page count is ≤ total pages in the PDF (blanks are skipped)."""
    import fitz

    with fitz.open(str(WHO_EML)) as doc:
        total = len(doc)
    assert len(who_eml_pages) <= total


def test_parse_pdf_tables_list_type(who_eml_pages):
    """The tables field is always a list."""
    for page in who_eml_pages:
        assert isinstance(page["tables"], list)


def test_parse_pdf_tables_extracted(who_eml_pages):
    """At least one page in WHO-EML has a non-empty tables list."""
    has_table = any(len(p["tables"]) > 0 for p in who_eml_pages)
    assert has_table, "Expected at least one table to be extracted from WHO-EML"


def test_parse_pdf_table_is_markdown(who_eml_pages):
    """Extracted tables contain markdown pipe characters."""
    for page in who_eml_pages:
        for table_md in page["tables"]:
            assert "|" in table_md, "Table markdown should contain '|'"


def test_parse_pdf_file_not_found():
    """parse_pdf raises FileNotFoundError for a missing path."""
    with pytest.raises(FileNotFoundError):
        parse_pdf("data/raw/does_not_exist.pdf")


def test_parse_pdf_idf_multi_column():
    """IDF PDF (3-column) parses without error and returns pages."""
    pages = parse_pdf(IDF_PDF)
    assert len(pages) > 0
    for page in pages:
        assert page["text"].strip()


def test_parse_pdf_who_clm_two_column():
    """WHO Climate doc (2-column) parses without error and returns pages."""
    pages = parse_pdf(WHO_CLM)
    assert len(pages) > 0


# ---------------------------------------------------------------------------
# metadata tests
# ---------------------------------------------------------------------------


def test_attach_metadata_adds_fields(who_eml_with_metadata):
    """All four metadata fields are present on every page."""
    required = {"source_document", "document_type", "section_title", "chunk_id"}
    for page in who_eml_with_metadata:
        assert required.issubset(page.keys()), (
            f"Missing metadata keys on page {page.get('page_number')}"
        )


def test_attach_metadata_source_document(who_eml_with_metadata):
    """source_document matches the filename passed in."""
    for page in who_eml_with_metadata:
        assert page["source_document"] == WHO_EML.name


def test_attach_metadata_document_type(who_eml_with_metadata):
    """document_type is 'essential_medicines_list' for WHO-EML."""
    for page in who_eml_with_metadata:
        assert page["document_type"] == "essential_medicines_list"


def test_attach_metadata_chunk_ids_unique(who_eml_with_metadata):
    """All chunk_id values are unique across the document."""
    ids = [p["chunk_id"] for p in who_eml_with_metadata]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"


def test_attach_metadata_chunk_id_format(who_eml_with_metadata):
    """chunk_id follows the {stem}_p{NNNN} format."""
    import re

    stem = WHO_EML.stem
    pattern = re.compile(rf"^{re.escape(stem)}_p\d{{4}}$")
    for page in who_eml_with_metadata:
        assert pattern.match(page["chunk_id"]), (
            f"Unexpected chunk_id format: {page['chunk_id']}"
        )


def test_attach_metadata_section_title_never_empty(who_eml_with_metadata):
    """section_title is never an empty string."""
    for page in who_eml_with_metadata:
        assert page["section_title"], (
            f"Empty section_title on page {page['page_number']}"
        )


def test_attach_metadata_section_title_coverage(who_eml_with_metadata):
    """At least 30% of pages have a non-'Unknown' section title."""
    total = len(who_eml_with_metadata)
    known = sum(1 for p in who_eml_with_metadata if p["section_title"] != "Unknown")
    ratio = known / total
    assert ratio >= 0.30, f"Only {ratio:.0%} of pages have a known section title"


def test_infer_document_type_who_eml():
    """_infer_document_type returns correct type for WHO-EML filename."""
    assert _infer_document_type("WHO-MHP-HPS-EML-2023.02-eng.pdf") == "essential_medicines_list"


def test_infer_document_type_idf():
    """_infer_document_type returns correct type for IDF filename."""
    assert _infer_document_type("IDF_Rec_2025.pdf") == "clinical_recommendation"


def test_infer_document_type_who_clm():
    """_infer_document_type returns correct type for WHO ISBN filename."""
    assert _infer_document_type("9789240081888-eng.pdf") == "who_guideline"


def test_infer_document_type_fallback():
    """_infer_document_type falls back to 'medical_document' for unknown files."""
    assert _infer_document_type("random_file.pdf") == "medical_document"
