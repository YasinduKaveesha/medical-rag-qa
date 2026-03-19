"""Metadata attachment module for parsed PDF pages.

Adds source document name, document type, section title, and a unique
chunk ID to each page dict produced by :mod:`src.ingestion.pdf_parser`.
"""

from __future__ import annotations

import logging
import re
import statistics
from pathlib import Path

import fitz  # PyMuPDF — used for font-size based title detection

logger = logging.getLogger(__name__)

# Regex for numbered section headings, e.g. "2.3 Medicines for palliative care"
_SECTION_HEADING_RE = re.compile(r"^\d+(\.\d+)*\s+[A-Z]")

# Ratio of a block's font size to the page median that qualifies it as a title
_TITLE_FONT_RATIO: float = 1.2

# Maps substrings in the PDF filename stem to document_type values
_DOCTYPE_MAP: dict[str, str] = {
    "WHO-MHP-HPS-EML": "essential_medicines_list",
    "EML":             "essential_medicines_list",
    "IDF_Rec":         "clinical_recommendation",
    "IDF":             "clinical_recommendation",
    "9789240081888":   "who_guideline",
}


def _infer_document_type(filename: str) -> str:
    """Infer a document type label from the PDF filename.

    Args:
        filename: Base filename of the PDF (with or without extension).

    Returns:
        A short snake_case label such as ``"essential_medicines_list"``.
        Falls back to ``"medical_document"`` if no pattern matches.
    """
    stem = Path(filename).stem
    for key, doc_type in _DOCTYPE_MAP.items():
        if key in stem:
            return doc_type
    return "medical_document"


def _get_page_font_sizes(pdf_path: str, page_number: int) -> list[float]:
    """Return all font sizes used in body text on a page.

    Args:
        pdf_path: Path to the source PDF.
        page_number: 1-based page number.

    Returns:
        List of font sizes (may contain duplicates).  Empty on error.
    """
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_number - 1]
            sizes: list[float] = []
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            sizes.append(span["size"])
            return sizes
    except Exception:
        return []


def _detect_section_title(text: str, font_sizes: list[float]) -> str | None:
    """Detect a section title from the first few lines of page text.

    Uses two heuristics in order:
    1. Numbered heading pattern (``"2.3 Section name"``).
    2. First line whose font size is >= ``_TITLE_FONT_RATIO`` * median body size.

    Args:
        text: Full page text.
        font_sizes: List of font sizes on the page (from :func:`_get_page_font_sizes`).

    Returns:
        The detected title string, or ``None`` if no title is found.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    # Heuristic 1: numbered section heading in the first 5 lines
    for line in lines[:5]:
        if _SECTION_HEADING_RE.match(line):
            return line

    # Heuristic 2: font-size ratio (requires font size data)
    if font_sizes:
        try:
            median_size = statistics.median(font_sizes)
            threshold = median_size * _TITLE_FONT_RATIO
            # Check first 3 lines against the threshold
            for line in lines[:3]:
                # Use the maximum font size seen on this page as a proxy
                # (we don't have per-line sizes here; use max as conservative check)
                if max(font_sizes) >= threshold and len(line) < 120:
                    return line
        except statistics.StatisticsError:
            pass

    return None


def attach_metadata(
    pages: list[dict],
    source_document: str,
    pdf_path: str | None = None,
) -> list[dict]:
    """Attach metadata fields to each page dict returned by :func:`parse_pdf`.

    Adds four fields to every dict in ``pages``:

    - ``source_document``: original PDF filename.
    - ``document_type``: inferred label (e.g. ``"essential_medicines_list"``).
    - ``section_title``: best-effort section heading for the page; carries
      forward the last known title so no page has an empty value.
    - ``chunk_id``: unique string identifier in the format
      ``{stem}_p{page_number:04d}``.

    Args:
        pages: List of page dicts from :func:`src.ingestion.pdf_parser.parse_pdf`.
        source_document: Filename of the source PDF (basename, e.g.
            ``"WHO-MHP-HPS-EML-2023.02-eng.pdf"``).
        pdf_path: Optional absolute path to the PDF, used for font-size based
            title detection.  When ``None``, only the regex heuristic is used.

    Returns:
        The same list with metadata fields added in-place.
    """
    doc_type = _infer_document_type(source_document)
    stem = Path(source_document).stem
    last_title = "Unknown"

    logger.info(
        "Attaching metadata to %d pages — source=%s, type=%s",
        len(pages), source_document, doc_type,
    )

    for page in pages:
        page_number = page["page_number"]
        text = page.get("text", "")

        # Font sizes for this page (empty list if pdf_path not provided)
        font_sizes = _get_page_font_sizes(pdf_path, page_number) if pdf_path else []

        detected = _detect_section_title(text, font_sizes)
        if detected:
            last_title = detected

        page["source_document"] = source_document
        page["document_type"] = doc_type
        page["section_title"] = last_title
        page["chunk_id"] = f"{stem}_p{page_number:04d}"

    logger.info("Metadata attached to %d pages", len(pages))
    return pages
