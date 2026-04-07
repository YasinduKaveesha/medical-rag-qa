"""CLIP encoder for joint image-text embeddings.

Wraps ``openai/clip-vit-base-patch32`` (512-dim) for both image and text
encoding.  All embeddings are L2-normalised so cosine similarity reduces to
a dot product and scores lie in [-1, 1].

Typical usage
-------------
::

    from src.embeddings.clip_encoder import get_clip_encoder

    encoder = get_clip_encoder()
    img_vec  = encoder.encode_image("data/extracted_images/who_p1_x5.png")
    txt_vec  = encoder.encode_text("chest X-ray showing bilateral infiltrates")
    score    = encoder.compute_similarity(img_vec, txt_vec)
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class CLIPEncoder:
    """Joint image-text encoder using OpenAI CLIP via HuggingFace Transformers.

    Loads ``CLIPModel`` and ``CLIPProcessor`` once on construction, then
    provides single-item and batched encoding for both images and text.
    Use :func:`get_clip_encoder` to obtain the process-level singleton.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``"openai/clip-vit-base-patch32"`` (512-dim).
        device: Torch device string — ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._model_name = model_name
        self._device = device

        logger.info("CLIPEncoder: loading %s on %s", model_name, device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()
        logger.info("CLIPEncoder: model ready")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """HuggingFace model identifier used by this encoder."""
        return self._model_name

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """Encode a single image into an L2-normalised 512-dim vector.

        Args:
            image: A PIL :class:`~PIL.Image.Image` or a file-path string.

        Returns:
            1-D ``np.float32`` array of shape ``(512,)``, L2-normalised.
        """
        import torch

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_image_features(**inputs)

        vec = features[0].cpu().numpy().astype(np.float32)
        return self._l2_normalize(vec)

    def encode_images_batch(
        self,
        images: list[Union[Image.Image, str]],
        batch_size: int = 16,
    ) -> list[np.ndarray]:
        """Encode a list of images in batches.

        Args:
            images: List of PIL images or file-path strings.
            batch_size: Number of images per forward pass.

        Returns:
            List of L2-normalised ``(512,)`` float32 arrays in input order.
        """
        results: list[np.ndarray] = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            for img in images[i : i + batch_size]:
                results.append(self.encode_image(img))
        return results

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(self, text: str) -> np.ndarray:
        """Encode a single text string into an L2-normalised 512-dim vector.

        Args:
            text: Input text (medical query or image caption).

        Returns:
            1-D ``np.float32`` array of shape ``(512,)``, L2-normalised.
        """
        import torch

        inputs = self._processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            features = self._model.get_text_features(**inputs)

        vec = features[0].cpu().numpy().astype(np.float32)
        return self._l2_normalize(vec)

    def encode_texts_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        """Encode a list of text strings in batches.

        Args:
            texts: List of input strings.
            batch_size: Number of texts per forward pass.

        Returns:
            List of L2-normalised ``(512,)`` float32 arrays in input order.
        """
        results: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            for text in texts[i : i + batch_size]:
                results.append(self.encode_text(text))
        return results

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    def compute_similarity(
        self, emb_a: np.ndarray, emb_b: np.ndarray
    ) -> float:
        """Compute cosine similarity between two L2-normalised embeddings.

        Because both embeddings are L2-normalised, cosine similarity equals
        the dot product and lies in [-1, 1].

        Args:
            emb_a: First embedding vector (shape ``(512,)``).
            emb_b: Second embedding vector (shape ``(512,)``).

        Returns:
            Scalar cosine similarity in [-1.0, 1.0].
        """
        return float(np.dot(emb_a, emb_b))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        """Return *vec* divided by its L2 norm (handles zero vector safely)."""
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            return vec
        return vec / norm


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_clip_encoder: CLIPEncoder | None = None


def get_clip_encoder() -> CLIPEncoder:
    """Return the process-level singleton :class:`CLIPEncoder`.

    On the first call reads ``Settings.clip_model`` and ``Settings.device``
    from ``.env`` and loads the model.  Subsequent calls return the cached
    instance without reloading.

    Returns:
        The singleton :class:`CLIPEncoder` instance.
    """
    global _clip_encoder
    if _clip_encoder is None:
        settings = get_settings()
        logger.info("Initialising singleton CLIPEncoder: %s", settings.clip_model)
        _clip_encoder = CLIPEncoder(
            model_name=settings.clip_model,
            device=settings.device,
        )
    return _clip_encoder
