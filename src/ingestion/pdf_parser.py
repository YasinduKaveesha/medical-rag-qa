"""PDF parsing module using PyMuPDF (fitz).

Extracts text and tables from PDF files, handling multi-column layouts,
repeating headers/footers, and blank pages.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Vertical margins to strip repeating headers and footers.
# Based on inspection: WHO-EML header sits at y≈36–53 on 842pt A4 pages.
_HEADER_CUTOFF: float = 55.0
_FOOTER_MARGIN: float = 50.0

# Minimum horizontal gap (points) between text-block x0 values that
# indicates a new column.
_COLUMN_GAP: float = 100.0


def _detect_columns(x0_values: list[float]) -> list[float]:
    """Return sorted column boundary x-positions from a list of block x0 values.

    Consecutive x0 values separated by more than ``_COLUMN_GAP`` are treated
    as distinct columns.

    Args:
        x0_values: Sorted list of x0 positions from text blocks on a page.

    Returns:
        List of x0 values that represent the start of each column.
    """
    if not x0_values:
        return []
    columns = [x0_values[0]]
    for x in x0_values[1:]:
        if x - columns[-1] > _COLUMN_GAP:
            columns.append(x)
    return columns


def _column_index(x0: float, columns: list[float]) -> int:
    """Return the zero-based column index for a block at position ``x0``.

    Args:
        x0: Left edge of the block.
        columns: Sorted list of column start positions from :func:`_detect_columns`.

    Returns:
        Index of the column this block belongs to.
    """
    for i, col_x in enumerate(reversed(columns)):
        if x0 >= col_x - 5:  # 5pt tolerance for minor alignment variation
            return len(columns) - 1 - i
    return 0


def _extract_tables(page: fitz.Page) -> list[str]:
    """Extract tables from a page and serialise each to markdown.

    Args:
        page: A :class:`fitz.Page` object.

    Returns:
        List of markdown-formatted table strings.  Empty list if no tables found.
    """
    markdown_tables: list[str] = []
    try:
        finder = page.find_tables()
        for table in finder.tables:
            rows = table.extract()
            if not rows:
                continue
            lines: list[str] = []
            header = rows[0]
            lines.append("| " + " | ".join(str(c or "") for c in header) + " |")
            lines.append("| " + " | ".join("---" for _ in header) + " |")
            for row in rows[1:]:
                lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
            markdown_tables.append("\n".join(lines))
    except Exception:
        logger.debug("Table extraction failed on page %d — skipping", page.number + 1)
    return markdown_tables


def _blocks_to_text(page: fitz.Page) -> str:
    """Extract body text from a page in correct reading order.

    Strips repeating headers (y < ``_HEADER_CUTOFF``) and footers
    (y > ``page.height - _FOOTER_MARGIN``).  Handles multi-column layouts by
    sorting blocks by ``(column_index, y0)`` before joining.

    Args:
        page: A :class:`fitz.Page` object.

    Returns:
        Cleaned body text as a single string.
    """
    page_height = page.rect.height
    footer_cutoff = page_height - _FOOTER_MARGIN

    raw = page.get_text("dict")
    text_blocks = [b for b in raw["blocks"] if b["type"] == 0]

    # Filter header / footer bands
    body_blocks = [
        b for b in text_blocks
        if b["bbox"][1] >= _HEADER_CUTOFF and b["bbox"][3] <= footer_cutoff
    ]

    if not body_blocks:
        return ""

    # Detect columns
    x0_values = sorted(set(round(b["bbox"][0]) for b in body_blocks))
    columns = _detect_columns(x0_values)

    # Sort by (column_index, y0) for correct reading order
    body_blocks.sort(key=lambda b: (_column_index(b["bbox"][0], columns), b["bbox"][1]))

    lines: list[str] = []
    for block in body_blocks:
        for line in block["lines"]:
            line_text = "".join(span["text"] for span in line["spans"]).strip()
            if line_text:
                lines.append(line_text)

    return "\n".join(lines)


def parse_pdf(path: str | Path) -> list[dict]:
    """Parse a PDF file and return one record per non-blank page.

    Each record contains the page number, extracted body text, and any tables
    serialised as markdown strings.  Blank pages (no text after stripping
    headers/footers) are skipped.

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        List of dicts, each with keys:
            - ``page_number`` (int): 1-based page number in the source PDF.
            - ``text`` (str): Cleaned body text.  Table markdown is appended
              at the end, separated by a blank line.
            - ``tables`` (list[str]): Raw markdown string for each table found
              on the page.  Empty list when no tables are present.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    logger.info("Parsing PDF: %s", path.name)
    pages: list[dict] = []

    with fitz.open(str(path)) as doc:
        total = len(doc)
        logger.info("Total pages in %s: %d", path.name, total)

        for page in doc:
            page_num = page.number + 1  # 1-based
            text = _blocks_to_text(page)
            tables = _extract_tables(page)

            # Append table markdown to body text so it isn't lost in chunking
            if tables:
                text = text + "\n\n" + "\n\n".join(tables)

            if not text.strip():
                logger.debug("Skipping blank page %d in %s", page_num, path.name)
                continue

            pages.append({
                "page_number": page_num,
                "text": text.strip(),
                "tables": tables,
            })

    logger.info("Parsed %d non-blank pages from %s", len(pages), path.name)
    return pages
