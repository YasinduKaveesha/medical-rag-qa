"""Image extraction from PDF files using PyMuPDF.

Extracts embedded images from clinical PDF documents, validates them for
minimum size, deduplicates by xref, and saves them as PNG files.

Typical usage
-------------
::

    from src.ingestion.image_extractor import ImageExtractor

    extractor = ImageExtractor(output_dir="data/extracted_images")
    images = extractor.extract_images_from_pdf("data/raw/WHO-EML-2023.pdf")
    # images[i].image_path, .source_pdf, .page_number, .image_id, ...
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Metadata for a single image extracted from a PDF.

    Attributes:
        image_path: Relative path to the saved PNG file.
        source_pdf: Filename of the source PDF (basename).
        page_number: 1-based page number where the image was found.
        xref: PyMuPDF cross-reference number — unique per image in the PDF.
        width: Image width in pixels.
        height: Image height in pixels.
        image_id: Unique identifier in the format ``{pdf_stem}_p{page}_x{xref}``.
    """

    image_path: str
    source_pdf: str
    page_number: int
    xref: int
    width: int
    height: int
    image_id: str


class ImageExtractor:
    """Extract and save embedded images from PDF files.

    Uses PyMuPDF to iterate pages, extract images by xref, validate minimum
    dimensions, deduplicate across pages, and save as PNG files.

    Args:
        output_dir: Directory to save extracted PNG images.  Created
            automatically if it does not exist.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.debug("ImageExtractor: output_dir=%s", output_dir)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_images_from_pdf(self, pdf_path: str) -> list[ExtractedImage]:
        """Extract all valid, unique images from a PDF file.

        Opens the PDF, iterates all pages, and delegates per-page extraction
        to :meth:`extract_images_from_page`.  Images are deduplicated across
        pages by their ``xref`` value so the same embedded image appearing on
        multiple pages is only extracted once.

        Args:
            pdf_path: Path to the source PDF file.

        Returns:
            List of :class:`ExtractedImage` instances, one per unique valid
            image found.  Returns an empty list if the PDF contains no images.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("ImageExtractor: extracting from %s", pdf_path_obj.name)

        extracted: list[ExtractedImage] = []
        seen_xrefs: set[int] = set()
        skipped_dup = 0

        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            logger.info(
                "ImageExtractor: %s has %d pages", pdf_path_obj.name, total_pages
            )
            for page in doc:
                page_images = self.extract_images_from_page(
                    page, pdf_path, page.number + 1
                )
                for img in page_images:
                    if img.xref in seen_xrefs:
                        skipped_dup += 1
                        logger.debug(
                            "ImageExtractor: skipping duplicate xref=%d", img.xref
                        )
                        continue
                    seen_xrefs.add(img.xref)
                    extracted.append(img)

        logger.info(
            "ImageExtractor: found %d unique images (%d duplicates skipped) in %s",
            len(extracted),
            skipped_dup,
            pdf_path_obj.name,
        )
        return extracted

    def extract_images_from_page(
        self,
        page: fitz.Page,
        pdf_path: str,
        page_num: int,
    ) -> list[ExtractedImage]:
        """Extract images from a single PDF page.

        Uses ``page.get_images(full=True)`` to enumerate image references,
        extracts each via ``doc.extract_image(xref)``, validates with
        :meth:`_is_valid_image`, and saves with :meth:`_save_image`.

        Args:
            page: A :class:`fitz.Page` object (must belong to an open document).
            pdf_path: Path to the source PDF (used for naming saved files).
            page_num: 1-based page number (used for naming and metadata).

        Returns:
            List of :class:`ExtractedImage` instances from this page.
            Empty list if no valid images found.
        """
        results: list[ExtractedImage] = []
        doc = page.parent  # type: ignore[attr-defined]

        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                img_dict = doc.extract_image(xref)
            except Exception:
                logger.debug(
                    "ImageExtractor: failed to extract xref=%d on page %d", xref, page_num
                )
                continue

            image_bytes = img_dict.get("image", b"")
            if not image_bytes or not self._is_valid_image(image_bytes):
                logger.debug(
                    "ImageExtractor: skipping invalid/tiny image xref=%d page=%d",
                    xref,
                    page_num,
                )
                continue

            try:
                saved_path = self._save_image(image_bytes, xref, pdf_path, page_num)
            except Exception:
                logger.warning(
                    "ImageExtractor: failed to save xref=%d page=%d", xref, page_num
                )
                continue

            # Get dimensions from PIL
            try:
                with Image.open(io.BytesIO(image_bytes)) as pil_img:
                    width, height = pil_img.size
            except Exception:
                width = img_dict.get("width", 0)
                height = img_dict.get("height", 0)

            pdf_stem = Path(pdf_path).stem
            image_id = f"{pdf_stem}_p{page_num}_x{xref}"

            results.append(
                ExtractedImage(
                    image_path=saved_path,
                    source_pdf=Path(pdf_path).name,
                    page_number=page_num,
                    xref=xref,
                    width=width,
                    height=height,
                    image_id=image_id,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _save_image(
        self,
        image_bytes: bytes,
        xref: int,
        pdf_path: str,
        page_num: int,
    ) -> str:
        """Save raw image bytes as a PNG file and return the relative path.

        Args:
            image_bytes: Raw image bytes as returned by ``doc.extract_image()``.
            xref: PyMuPDF xref for naming the file.
            pdf_path: Source PDF path (stem used in filename).
            page_num: 1-based page number (used in filename).

        Returns:
            Relative path to the saved PNG file (relative to project root /
            the working directory at call time).
        """
        stem = Path(pdf_path).stem
        filename = f"{stem}_p{page_num}_x{xref}.png"
        save_path = Path(self._output_dir) / filename

        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            pil_img.save(str(save_path), format="PNG")

        # Return relative path from output_dir parent or as-is
        logger.debug("ImageExtractor: saved %s", save_path)
        return str(save_path)

    def _is_valid_image(
        self,
        image_bytes: bytes,
        min_size: tuple[int, int] = (50, 50),
    ) -> bool:
        """Return True if the image is valid and meets minimum dimension requirements.

        Args:
            image_bytes: Raw image bytes to validate.
            min_size: Minimum ``(width, height)`` in pixels.  Defaults to
                ``(50, 50)`` to filter out decorative icons and line art.

        Returns:
            ``True`` when the image can be decoded and both dimensions meet or
            exceed ``min_size``.  ``False`` for corrupted images or images
            smaller than the threshold.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                return img.width >= min_size[0] and img.height >= min_size[1]
        except Exception:
            return False
