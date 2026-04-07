"""Multimodal retrieval pipeline for Phase 2.

Orchestrates four steps in sequence:

1. **Text retrieval** — embed query with MiniLM, search text collection, rerank.
2. **Image retrieval** — embed query with CLIP, search CLIP collection.
3. **Fusion** — merge both ranked lists with Reciprocal Rank Fusion.
4. **Classify** — split fused results into text chunks vs. image results.

Typical usage
-------------
::

    from src.retrieval.multimodal_pipeline import get_multimodal_pipeline

    pipeline = get_multimodal_pipeline()
    result = pipeline.retrieve("chest X-ray bilateral infiltrates", top_k=5)
    # result.text_chunks  — list of text dicts
    # result.images       — list of image/caption dicts
    # result.retrieval_time_ms  — wall-clock latency in ms
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.retrieval.fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from src.config import Settings
    from src.embeddings.clip_encoder import CLIPEncoder
    from src.embeddings.encoder import EmbeddingEncoder
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.vector_store import MultiModalVectorStore

logger = logging.getLogger(__name__)

# Types that are treated as image results in _classify_results
_IMAGE_TYPES = {"image", "image_caption"}


@dataclass
class RetrievalResult:
    """Result of a multimodal retrieval query.

    Attributes:
        text_chunks: Text chunks ranked by RRF + reranker score.
        images: Image / image-caption results ranked by RRF score.
        fusion_scores: Mapping of result key → RRF score for diagnostics.
        retrieval_time_ms: Total wall-clock time in milliseconds.
    """

    text_chunks: list[dict] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    fusion_scores: dict[str, float] = field(default_factory=dict)
    retrieval_time_ms: float = 0.0


class MultiModalRetrievalPipeline:
    """Dual-encoder retrieval pipeline combining text and image search.

    All ML components are injected via constructor so tests can substitute
    lightweight mocks without loading real models or requiring a running
    Qdrant instance.

    Args:
        text_encoder: :class:`~src.embeddings.encoder.EmbeddingEncoder`
            for query embedding (MiniLM, 384-dim).
        clip_encoder: :class:`~src.embeddings.clip_encoder.CLIPEncoder`
            for image-text embedding (CLIP, 512-dim).
        store: :class:`~src.retrieval.vector_store.MultiModalVectorStore`
            with both text and CLIP collections.
        reranker: :class:`~src.retrieval.reranker.CrossEncoderReranker`
            applied only to text candidates.
        config: :class:`~src.config.Settings` instance.
    """

    def __init__(
        self,
        text_encoder: EmbeddingEncoder,
        clip_encoder: CLIPEncoder,
        store: MultiModalVectorStore,
        reranker: CrossEncoderReranker,
        config: Settings,
    ) -> None:
        self._text_encoder = text_encoder
        self._clip_encoder = clip_encoder
        self._store = store
        self._reranker = reranker
        self._config = config
        logger.debug("MultiModalRetrievalPipeline initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Run multimodal retrieval for *query*.

        Executes text retrieval (MiniLM → Qdrant → reranker) and image
        retrieval (CLIP → Qdrant) in sequence, fuses them with RRF, then
        classifies results into text chunks and images.

        Args:
            query: Raw natural-language question string.
            top_k: Total number of fused results to return.  Text and image
                results are not individually capped — fusion decides the mix.

        Returns:
            :class:`RetrievalResult` with ``text_chunks``, ``images``,
            ``fusion_scores``, and ``retrieval_time_ms``.
        """
        t0 = time.perf_counter()
        logger.info("MultiModalRetrievalPipeline.retrieve query=%r top_k=%d", query[:80], top_k)

        text_results = self._retrieve_text(query)
        image_results = self._retrieve_images(query)

        fused = reciprocal_rank_fusion(
            [text_results, image_results],
            k=self._config.rrf_k,
            top_k=top_k,
        )

        text_chunks, images = self._classify_results(fused)
        fusion_scores = {
            (r.get("chunk_id") or r.get("image_id") or r.get("id", "")): r["rrf_score"]
            for r in fused
            if "rrf_score" in r
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "retrieve() done — %.1fms  text_chunks=%d  images=%d",
            elapsed_ms,
            len(text_chunks),
            len(images),
        )

        return RetrievalResult(
            text_chunks=text_chunks,
            images=images,
            fusion_scores=fusion_scores,
            retrieval_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_text(self, query: str, top_k: int = 20) -> list[dict]:
        """Embed → vector search → rerank text chunks.

        Args:
            query: Raw query string.
            top_k: Number of candidates to fetch from Qdrant before reranking.

        Returns:
            Flat list of dicts suitable for :func:`~src.retrieval.fusion.reciprocal_rank_fusion`.
            Each dict has ``chunk_id``, ``text``, ``type="text"``,
            ``score``, and ``reranker_score``.
        """
        query_vec = self._text_encoder.encode(query)
        candidates = self._store.search(query_vec, top_k=top_k)

        if not candidates:
            return []

        reranked = self._reranker.rerank(query, candidates, top_k=top_k)

        # Flatten from reranker format to fusion-ready flat dicts
        results: list[dict] = []
        for item in reranked:
            chunk = item.get("chunk", {})
            flat = {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": chunk.get("text", ""),
                "type": chunk.get("type", "text"),
                "score": item.get("score", 0.0),
                "reranker_score": item.get("reranker_score", 0.0),
                **{k: v for k, v in chunk.items() if k not in ("text", "type")},
            }
            results.append(flat)

        logger.debug("_retrieve_text: %d reranked text results", len(results))
        return results

    def _retrieve_images(self, query: str, top_k: int = 20) -> list[dict]:
        """Embed query with CLIP → search CLIP collection.

        Images are not reranked — CLIP cosine similarity is used directly.

        Args:
            query: Raw query string.
            top_k: Number of image candidates to fetch from Qdrant.

        Returns:
            Flat list of dicts from
            :meth:`~src.retrieval.vector_store.MultiModalVectorStore.search_clip`.
        """
        clip_vec = self._clip_encoder.encode_text(query)
        results = self._store.search_clip(
            self._config.clip_collection_name, clip_vec, top_k=top_k
        )
        logger.debug("_retrieve_images: %d CLIP results", len(results))
        return results

    @staticmethod
    def _classify_results(fused: list[dict]) -> tuple[list[dict], list[dict]]:
        """Split fused results into text chunks and image results.

        Args:
            fused: Output of :func:`~src.retrieval.fusion.reciprocal_rank_fusion`.

        Returns:
            Tuple ``(text_chunks, images)``.  Results whose ``type`` field is
            ``"image"`` or ``"image_caption"`` go to *images*; everything else
            (``"text"`` or absent) goes to *text_chunks*.
        """
        text_chunks: list[dict] = []
        images: list[dict] = []

        for result in fused:
            result_type = result.get("type", "text")
            if result_type in _IMAGE_TYPES:
                images.append(result)
            else:
                text_chunks.append(result)

        return text_chunks, images


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_mm_pipeline: MultiModalRetrievalPipeline | None = None


def get_multimodal_pipeline() -> MultiModalRetrievalPipeline:
    """Return the process-level singleton :class:`MultiModalRetrievalPipeline`.

    On the first call constructs the pipeline using the singleton encoders,
    store, reranker, and settings.  Subsequent calls return the cached instance.

    Returns:
        The singleton :class:`MultiModalRetrievalPipeline` instance.
    """
    global _mm_pipeline
    if _mm_pipeline is None:
        from src.config import get_settings  # noqa: PLC0415
        from src.embeddings.clip_encoder import get_clip_encoder  # noqa: PLC0415
        from src.embeddings.encoder import get_encoder  # noqa: PLC0415
        from src.retrieval.reranker import get_reranker  # noqa: PLC0415
        from src.retrieval.vector_store import get_multimodal_store  # noqa: PLC0415

        logger.info("Initialising singleton MultiModalRetrievalPipeline")
        _mm_pipeline = MultiModalRetrievalPipeline(
            text_encoder=get_encoder(),
            clip_encoder=get_clip_encoder(),
            store=get_multimodal_store(),
            reranker=get_reranker(),
            config=get_settings(),
        )
    return _mm_pipeline
