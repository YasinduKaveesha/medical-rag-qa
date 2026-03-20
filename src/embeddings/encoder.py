"""HuggingFace embedding encoder wrapping sentence-transformers.

Provides a singleton :class:`EmbeddingEncoder` that wraps
``sentence-transformers/all-MiniLM-L6-v2`` (384-dim, CPU-only) and exposes
single-text and batch encoding with tqdm progress tracking.

Typical usage
-------------
::

    from src.embeddings.encoder import get_encoder

    encoder = get_encoder()
    vector  = encoder.encode("What is the dose of amitriptyline?")
    vectors = encoder.encode_batch(["text one", "text two"], batch_size=32)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tqdm import tqdm

from src.config import get_settings

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEncoder:
    """Wrapper around a ``SentenceTransformer`` model for dense text embeddings.

    The model is loaded once in ``__init__`` and reused for every subsequent
    call to :meth:`encode` or :meth:`encode_batch`.  Use :func:`get_encoder`
    to obtain the process-level singleton instead of constructing this class
    directly.

    Args:
        model_name: HuggingFace model identifier.  Defaults to
            ``"sentence-transformers/all-MiniLM-L6-v2"``.
        _model: Optional pre-loaded model object.  When provided it is used
            directly, bypassing ``SentenceTransformer`` loading.  Intended
            for testing only — pass a :class:`MockSentenceTransformer` to
            avoid downloading the real weights.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        _model: Any | None = None,
    ) -> None:
        self._model_name = model_name

        if _model is not None:
            self._model = _model
            logger.debug("EmbeddingEncoder using injected model (test mode)")
        else:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Loading SentenceTransformer: %s", model_name)
            self._model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded — dim=%d", self.embedding_dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """HuggingFace model identifier used by this encoder."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vectors (384 for all-MiniLM-L6-v2)."""
        return int(self._model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a dense embedding vector.

        Args:
            text: Input text.  May be empty — the model returns a valid
                (near-zero) vector for empty strings.

        Returns:
            1-D ``np.float32`` array of shape ``(embedding_dim,)``.
        """
        raw = self._model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return np.array(raw, dtype=np.float32)

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        """Encode a list of texts into dense embedding vectors.

        Splits *texts* into batches of *batch_size* and shows a ``tqdm``
        progress bar when there is more than one batch (suppressed for small
        inputs to reduce noise in tests and interactive use).

        Args:
            texts: List of input strings.  Returns ``[]`` immediately for an
                empty list.
            batch_size: Number of texts per model call.  Defaults to ``32``.

        Returns:
            List of 1-D ``np.float32`` arrays, one per input text, each of
            shape ``(embedding_dim,)``.  Order matches *texts*.
        """
        if not texts:
            return []

        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
        show_bar = len(batches) > 1

        logger.info(
            "Encoding %d texts in %d batch(es) (batch_size=%d)",
            len(texts), len(batches), batch_size,
        )

        results: list[np.ndarray] = []
        for batch in tqdm(batches, desc="Encoding", unit="batch", disable=not show_bar):
            vecs = self._model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
            results.extend(np.array(v, dtype=np.float32) for v in vecs)

        return results


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_encoder: EmbeddingEncoder | None = None


def get_encoder() -> EmbeddingEncoder:
    """Return the process-level singleton :class:`EmbeddingEncoder`.

    On the first call, reads ``Settings.embedding_model`` (from ``.env``) and
    loads the model.  Subsequent calls return the cached instance without
    reloading.

    Returns:
        The singleton :class:`EmbeddingEncoder` instance.
    """
    global _encoder
    if _encoder is None:
        model_name = get_settings().embedding_model
        logger.info("Initialising singleton EmbeddingEncoder: %s", model_name)
        _encoder = EmbeddingEncoder(model_name=model_name)
    return _encoder
