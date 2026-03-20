"""Chunking strategies for splitting parsed PDF page text into retrieval chunks.

All chunkers implement the same interface via :class:`BaseChunker`:

    chunker.chunk(text, metadata) -> list[{"text": str, "metadata": dict}]

Child chunks inherit all parent page metadata and receive their own UUID
``chunk_id``, a sequential ``chunk_index``, and a ``chunking_strategy`` label.

Available strategies
--------------------
- :class:`FixedSizeChunker` — sliding token window (512 tokens, 50 overlap)
- :class:`SentenceChunker` — greedy sentence packing up to 512 tokens
- :class:`SemanticChunker` — cosine-similarity-based topic boundary detection
"""

from __future__ import annotations

import copy
import logging
import re
import uuid
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Sentence boundary regex: split after . ! ? followed by whitespace
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Default model used for token counting and (in SemanticChunker) encoding
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies.

    Subclasses must implement :meth:`chunk`.  The base class provides shared
    token-counting and chunk-building utilities.

    Args:
        model_name: HuggingFace model name used to load the tokenizer for
            token counting.  Defaults to ``"sentence-transformers/all-MiniLM-L6-v2"``.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        logger.debug("Loading tokenizer: %s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> list[dict]:
        """Split *text* into chunks, each carrying inherited *metadata*.

        Args:
            text: Page body text to split.
            metadata: Page-level metadata dict from
                :func:`src.ingestion.metadata.attach_metadata`.

        Returns:
            List of dicts, each with keys ``"text"`` (str) and
            ``"metadata"`` (dict).  Returns an empty list for blank input.
        """

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text* using the loaded tokenizer.

        Args:
            text: Input string.

        Returns:
            Integer token count (does not include special tokens).
        """
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _split_sentences(self, text: str) -> list[str]:
        """Split *text* into sentences using a punctuation-boundary regex.

        Args:
            text: Input string.

        Returns:
            Non-empty sentence strings.
        """
        parts = _SENTENCE_RE.split(text)
        return [s.strip() for s in parts if s.strip()]

    def _build_chunk(
        self,
        text: str,
        metadata: dict,
        index: int,
        strategy: str,
    ) -> dict:
        """Construct a single output chunk dict.

        Deep-copies *metadata* so mutations to one chunk do not affect others.
        Replaces the page-level ``chunk_id`` with a fresh UUID and adds
        ``chunk_index`` and ``chunking_strategy``.

        Args:
            text: Chunk text.
            metadata: Parent page metadata.
            index: Zero-based position of this chunk within the page.
            strategy: Strategy label (``"fixed_size"``, ``"sentence"``,
                or ``"semantic"``).

        Returns:
            Dict with keys ``"text"`` and ``"metadata"``.
        """
        child_meta = copy.deepcopy(metadata)
        child_meta["chunk_id"] = uuid.uuid4().hex
        child_meta["chunk_index"] = index
        child_meta["chunking_strategy"] = strategy
        return {"text": text, "metadata": child_meta}


# ---------------------------------------------------------------------------
# Fixed-size chunker
# ---------------------------------------------------------------------------


class FixedSizeChunker(BaseChunker):
    """Sliding-window chunker that operates at the token level.

    Tokenises the full page text, slides a window of *max_tokens* tokens
    across the token list with a step of ``max_tokens - overlap_tokens``,
    and decodes each window back to a string.  Every chunk is guaranteed
    to contain at most *max_tokens* tokens.

    Args:
        max_tokens: Maximum number of tokens per chunk.  Defaults to ``512``.
        overlap_tokens: Number of tokens shared between consecutive chunks.
            Defaults to ``50``.
        model_name: Tokenizer model name.  Defaults to the embedding model.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        super().__init__(model_name)
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be less than max_tokens")
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, text: str, metadata: dict) -> list[dict]:
        """Split *text* into fixed-size token windows.

        Args:
            text: Page body text.
            metadata: Parent page metadata dict.

        Returns:
            List of chunk dicts.  Empty list if *text* is blank.
        """
        if not text.strip():
            return []

        token_ids: list[int] = self._tokenizer.encode(text, add_special_tokens=False)

        if len(token_ids) <= self._max_tokens:
            decoded = self._tokenizer.decode(token_ids, skip_special_tokens=True)
            return [self._build_chunk(decoded.strip(), metadata, 0, "fixed_size")]

        step = self._max_tokens - self._overlap_tokens
        chunks: list[dict] = []
        start = 0
        idx = 0

        while start < len(token_ids):
            window = token_ids[start : start + self._max_tokens]
            decoded = self._tokenizer.decode(window, skip_special_tokens=True).strip()
            if decoded:
                chunks.append(self._build_chunk(decoded, metadata, idx, "fixed_size"))
                idx += 1
            start += step

        logger.debug(
            "FixedSizeChunker: %d tokens → %d chunks (max=%d, overlap=%d)",
            len(token_ids), len(chunks), self._max_tokens, self._overlap_tokens,
        )
        return chunks


# ---------------------------------------------------------------------------
# Sentence chunker
# ---------------------------------------------------------------------------


class SentenceChunker(BaseChunker):
    """Greedy sentence-packing chunker.

    Splits text into sentences at punctuation boundaries, then greedily
    accumulates sentences into the current chunk until the next sentence
    would push the token count past *max_tokens*.  No inter-chunk overlap
    is applied — sentence boundaries are natural semantic break points.

    A single sentence that exceeds *max_tokens* is emitted as its own chunk
    without truncation; the embedding model will truncate internally.

    Args:
        max_tokens: Token budget per chunk.  Defaults to ``512``.
        model_name: Tokenizer model name.  Defaults to the embedding model.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        super().__init__(model_name)
        self._max_tokens = max_tokens

    def chunk(self, text: str, metadata: dict) -> list[dict]:
        """Split *text* at sentence boundaries up to *max_tokens* per chunk.

        Args:
            text: Page body text.
            metadata: Parent page metadata dict.

        Returns:
            List of chunk dicts.  Empty list if *text* is blank.
        """
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[dict] = []
        current: list[str] = []
        current_tokens = 0
        idx = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            if current and current_tokens + sentence_tokens > self._max_tokens:
                # Flush current buffer
                chunk_text = " ".join(current).strip()
                chunks.append(self._build_chunk(chunk_text, metadata, idx, "sentence"))
                idx += 1
                current = [sentence]
                current_tokens = sentence_tokens
            else:
                current.append(sentence)
                current_tokens += sentence_tokens

        # Flush remainder
        if current:
            chunk_text = " ".join(current).strip()
            chunks.append(self._build_chunk(chunk_text, metadata, idx, "sentence"))

        logger.debug(
            "SentenceChunker: %d sentences → %d chunks (max=%d)",
            len(sentences), len(chunks), self._max_tokens,
        )
        return chunks


# ---------------------------------------------------------------------------
# Semantic chunker
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Float in [-1, 1].  Returns 0.0 if either vector has zero norm.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticChunker(BaseChunker):
    """Embedding-similarity-based chunker.

    Splits text into sentences, embeds them in one batched call, then
    inserts a chunk boundary wherever the cosine similarity between
    consecutive sentence embeddings falls below *similarity_threshold*.
    If a resulting group of sentences still exceeds *max_tokens*, it is
    sub-split using greedy sentence packing.

    The encoder is injected via *encoder* so tests can pass a mock without
    loading the real model.

    Args:
        encoder: Any object with an ``encode(sentences, show_progress_bar)``
            method that returns a 2-D array of shape ``(N, dim)``.  When
            ``None``, loads ``SentenceTransformer(model_name)``.
        similarity_threshold: Cosine similarity below which a boundary is
            inserted.  Defaults to ``0.5``.
        max_tokens: Token cap applied after semantic grouping.  Defaults to
            ``512``.
        model_name: Model used for both the encoder (when *encoder* is
            ``None``) and the tokenizer.  Defaults to the embedding model.
    """

    def __init__(
        self,
        encoder: Any | None = None,
        similarity_threshold: float = 0.5,
        max_tokens: int = 512,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        super().__init__(model_name)
        self._threshold = similarity_threshold
        self._max_tokens = max_tokens

        if encoder is not None:
            self._encoder = encoder
        else:
            # Lazy import to avoid loading sentence-transformers at module level
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.debug("Loading SentenceTransformer encoder: %s", model_name)
            self._encoder = SentenceTransformer(model_name)

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        """Greedily pack *sentences* into token-bounded strings.

        Used to sub-split semantic groups that exceed *max_tokens*.

        Args:
            sentences: Ordered list of sentence strings.

        Returns:
            List of packed text strings.
        """
        groups: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            st = self._count_tokens(sentence)
            if current and current_tokens + st > self._max_tokens:
                groups.append(" ".join(current).strip())
                current = [sentence]
                current_tokens = st
            else:
                current.append(sentence)
                current_tokens += st

        if current:
            groups.append(" ".join(current).strip())
        return groups

    def chunk(self, text: str, metadata: dict) -> list[dict]:
        """Split *text* using embedding cosine similarity between sentences.

        Args:
            text: Page body text.
            metadata: Parent page metadata dict.

        Returns:
            List of chunk dicts.  Empty list if *text* is blank.
        """
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # Single sentence — no similarity to compute
        if len(sentences) == 1:
            return [self._build_chunk(sentences[0], metadata, 0, "semantic")]

        # Embed all sentences in one batch
        embeddings: np.ndarray = self._encoder.encode(
            sentences, show_progress_bar=False
        )

        # Find split boundaries: where similarity drops below threshold
        boundaries: list[int] = []
        for i in range(len(sentences) - 1):
            sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
            if sim < self._threshold:
                boundaries.append(i + 1)  # split before sentence i+1

        # Build sentence groups from boundaries
        groups: list[list[str]] = []
        prev = 0
        for boundary in boundaries:
            groups.append(sentences[prev:boundary])
            prev = boundary
        groups.append(sentences[prev:])

        # Sub-split any group that exceeds max_tokens, then build output
        chunks: list[dict] = []
        idx = 0
        for group in groups:
            group_text = " ".join(group).strip()
            if self._count_tokens(group_text) > self._max_tokens:
                sub_texts = self._pack_sentences(group)
            else:
                sub_texts = [group_text]

            for sub_text in sub_texts:
                if sub_text:
                    chunks.append(self._build_chunk(sub_text, metadata, idx, "semantic"))
                    idx += 1

        logger.debug(
            "SemanticChunker: %d sentences → %d boundaries → %d chunks (threshold=%.2f)",
            len(sentences), len(boundaries), len(chunks), self._threshold,
        )
        return chunks
