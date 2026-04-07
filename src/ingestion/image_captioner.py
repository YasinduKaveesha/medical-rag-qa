"""Image captioning using Salesforce BLIP.

Generates natural-language captions for extracted PDF images, ready for
embedding into the text collection as ``type="image_caption"`` payloads.

Typical usage
-------------
::

    from src.ingestion.image_captioner import ImageCaptioner

    captioner = ImageCaptioner()
    caption = captioner.caption_image("data/extracted_images/who_p1_x5.png")
    # caption -> "a diagram showing lung anatomy"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Union

from PIL import Image
from tqdm import tqdm

from src.ingestion.image_extractor import ExtractedImage

logger = logging.getLogger(__name__)

# Regex to strip BLIP's "arafed" hallucination prefix (appears at the start)
_ARAFED_RE = re.compile(r"^arafed\s+", re.IGNORECASE)


@dataclass
class CaptionedImage:
    """An extracted image with its generated caption.

    Extends :class:`ExtractedImage` fields with caption metadata.

    Attributes:
        image_path: Relative path to the saved PNG file.
        source_pdf: Filename of the source PDF.
        page_number: 1-based page number.
        xref: PyMuPDF cross-reference number.
        width: Image width in pixels.
        height: Image height in pixels.
        image_id: Unique identifier ``{pdf_stem}_p{page}_x{xref}``.
        caption: Generated natural-language description of the image.
        caption_model: HuggingFace model ID used to generate the caption.
    """

    image_path: str
    source_pdf: str
    page_number: int
    xref: int
    width: int
    height: int
    image_id: str
    caption: str
    caption_model: str


class ImageCaptioner:
    """Generate captions for images using Salesforce BLIP.

    Loads the BLIP image-captioning model once and reuses it for all images.
    Supports single-image, batched, and :class:`ExtractedImage` list inputs.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``"Salesforce/blip-image-captioning-base"``.
        device: Torch device string — ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = "cpu",
    ) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self._model_name = model_name
        self._device = device

        logger.info("ImageCaptioner: loading %s on %s", model_name, device)
        self._processor = BlipProcessor.from_pretrained(model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()
        logger.info("ImageCaptioner: model ready")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def caption_image(self, image: Union[Image.Image, str]) -> str:
        """Generate a caption for a single image.

        Args:
            image: A PIL :class:`~PIL.Image.Image` or a file-path string.

        Returns:
            Cleaned caption string.  BLIP's ``"arafed"`` prefix artefact is
            stripped automatically.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        import torch

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_new_tokens=50)

        caption: str = self._processor.decode(output_ids[0], skip_special_tokens=True)
        return self._clean_caption(caption)

    def caption_batch(
        self,
        images: list[Union[Image.Image, str]],
        batch_size: int = 8,
    ) -> list[str]:
        """Caption a list of images in batches.

        Args:
            images: List of PIL images or file-path strings.
            batch_size: Number of images processed per forward pass.

        Returns:
            List of caption strings in the same order as *images*.
        """
        captions: list[str] = []
        for i in tqdm(range(0, len(images), batch_size), desc="Captioning"):
            batch = images[i : i + batch_size]
            for img in batch:
                captions.append(self.caption_image(img))
        return captions

    def caption_extracted_images(
        self,
        extracted_images: list[ExtractedImage],
    ) -> list[CaptionedImage]:
        """Caption a list of :class:`ExtractedImage` objects.

        Args:
            extracted_images: Images returned by
                :meth:`~src.ingestion.image_extractor.ImageExtractor.extract_images_from_pdf`.

        Returns:
            :class:`CaptionedImage` list with the same ordering.
        """
        logger.info(
            "ImageCaptioner: captioning %d images", len(extracted_images)
        )
        results: list[CaptionedImage] = []
        for ext_img in extracted_images:
            caption = self.caption_image(ext_img.image_path)
            results.append(
                CaptionedImage(
                    image_path=ext_img.image_path,
                    source_pdf=ext_img.source_pdf,
                    page_number=ext_img.page_number,
                    xref=ext_img.xref,
                    width=ext_img.width,
                    height=ext_img.height,
                    image_id=ext_img.image_id,
                    caption=caption,
                    caption_model=self._model_name,
                )
            )
        logger.info("ImageCaptioner: finished captioning %d images", len(results))
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_caption(caption: str) -> str:
        """Strip whitespace and remove BLIP's ``"arafed"`` prefix artefact."""
        caption = caption.strip()
        caption = _ARAFED_RE.sub("", caption)
        return caption.strip()
