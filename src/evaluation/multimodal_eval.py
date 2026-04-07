"""Multimodal RAG evaluation — keyword-based metrics without external LLM judges.

Evaluates the :class:`~src.retrieval.multimodal_pipeline.MultiModalRetrievalPipeline`
against a fixed set of 15 test queries split by expected modality (8 image, 7 text).
All metrics are deterministic and require no API calls.

Typical usage
-------------
::

    from src.evaluation.multimodal_eval import MultiModalEvaluator

    evaluator = MultiModalEvaluator(pipeline, text_only_pipeline)
    results = evaluator.evaluate_retrieval(top_k=5)
    print(evaluator.generate_report(results))
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_QUERIES_PATH = Path(__file__).parent / "multimodal_test_queries.json"


def _load_queries() -> list[dict]:
    """Load test queries from the bundled JSON file."""
    with _QUERIES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


class MultiModalEvaluator:
    """Keyword-based evaluator for the multimodal RAG pipeline.

    Compares retrieval quality between the full multimodal pipeline and an
    optional text-only baseline.  All metrics are computed locally without
    any LLM or network calls.

    Args:
        mm_pipeline: A :class:`~src.retrieval.multimodal_pipeline.MultiModalRetrievalPipeline`
            instance (or any object with a ``retrieve(query, top_k) -> RetrievalResult``
            method).
        text_pipeline: Optional text-only pipeline with a
            ``retrieve(query, top_k, filters=None) -> list[dict]`` method.
            When provided, :meth:`compare_text_only_vs_multimodal` is available.
        queries: Optional list of query dicts (overrides the bundled JSON).
    """

    def __init__(
        self,
        mm_pipeline,
        text_pipeline=None,
        queries: list[dict] | None = None,
    ) -> None:
        self._mm_pipeline = mm_pipeline
        self._text_pipeline = text_pipeline
        self._queries: list[dict] = queries if queries is not None else _load_queries()

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate_retrieval(self, top_k: int = 5) -> dict:
        """Run all test queries and compute aggregate metrics.

        For each query the pipeline is called once.  Precision is computed
        per-query by :meth:`image_retrieval_precision` or
        :meth:`text_retrieval_precision` depending on ``expected_type``.

        Args:
            top_k: Number of results to request from the pipeline.

        Returns:
            Dict with keys:
            - ``"total_queries"`` — number of test queries evaluated.
            - ``"image_queries"`` — count of image-type queries.
            - ``"text_queries"`` — count of text-type queries.
            - ``"image_precision"`` — average keyword precision for image queries.
            - ``"text_precision"`` — average keyword precision for text queries.
            - ``"modality_accuracy"`` — fraction of queries whose top result
              matches the expected modality.
            - ``"per_query"`` — list of per-query result dicts.
        """
        per_query: list[dict] = []
        image_precisions: list[float] = []
        text_precisions: list[float] = []
        modality_correct: list[bool] = []

        for query in self._queries:
            try:
                result = self._mm_pipeline.retrieve(query["question"], top_k=top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("evaluate_retrieval: query %s failed: %s", query["query_id"], exc)
                per_query.append({"query_id": query["query_id"], "error": str(exc)})
                continue

            expected_type = query["expected_type"]
            keywords = query["expected_keywords"]

            if expected_type == "image":
                precision = self.image_retrieval_precision(query, result)
                image_precisions.append(precision)
            else:
                precision = self.text_retrieval_precision(query, result)
                text_precisions.append(precision)

            mod_acc = self.modality_accuracy(query, result)
            modality_correct.append(mod_acc >= 0.5)

            per_query.append(
                {
                    "query_id": query["query_id"],
                    "question": query["question"],
                    "expected_type": expected_type,
                    "precision": precision,
                    "modality_accuracy": mod_acc,
                    "n_text_chunks": len(result.text_chunks),
                    "n_images": len(result.images),
                }
            )
            logger.debug(
                "evaluate_retrieval: %s  precision=%.3f  mod_acc=%.3f",
                query["query_id"],
                precision,
                mod_acc,
            )

        total = len(self._queries)
        avg_image = sum(image_precisions) / len(image_precisions) if image_precisions else 0.0
        avg_text = sum(text_precisions) / len(text_precisions) if text_precisions else 0.0
        mod_accuracy = sum(modality_correct) / len(modality_correct) if modality_correct else 0.0

        return {
            "total_queries": total,
            "image_queries": len(image_precisions),
            "text_queries": len(text_precisions),
            "image_precision": round(avg_image, 4),
            "text_precision": round(avg_text, 4),
            "modality_accuracy": round(mod_accuracy, 4),
            "per_query": per_query,
        }

    # ------------------------------------------------------------------
    # Per-query metrics
    # ------------------------------------------------------------------

    def image_retrieval_precision(self, query: dict, result) -> float:
        """Fraction of expected keywords found in retrieved image captions/text.

        Concatenates all captions and text from retrieved images, then counts
        how many expected keywords appear (case-insensitive).

        Args:
            query: A query dict with ``"expected_keywords"`` list.
            result: :class:`~src.retrieval.multimodal_pipeline.RetrievalResult`.

        Returns:
            Float in [0, 1].  ``0.0`` if no keywords or no images retrieved.
        """
        keywords = query.get("expected_keywords", [])
        if not keywords:
            return 0.0

        combined = " ".join(
            (img.get("caption", "") + " " + img.get("text", "")).lower()
            for img in result.images
        )

        hits = sum(1 for kw in keywords if kw.lower() in combined)
        return hits / len(keywords)

    def text_retrieval_precision(self, query: dict, result) -> float:
        """Fraction of expected keywords found in retrieved text chunks.

        Concatenates all chunk text, then counts how many expected keywords
        appear (case-insensitive).

        Args:
            query: A query dict with ``"expected_keywords"`` list.
            result: :class:`~src.retrieval.multimodal_pipeline.RetrievalResult`.

        Returns:
            Float in [0, 1].  ``0.0`` if no keywords or no text chunks retrieved.
        """
        keywords = query.get("expected_keywords", [])
        if not keywords:
            return 0.0

        combined = " ".join(
            (chunk.get("text", "") + " " + chunk.get("chunk_text", "")).lower()
            for chunk in result.text_chunks
        )

        hits = sum(1 for kw in keywords if kw.lower() in combined)
        return hits / len(keywords)

    def modality_accuracy(self, query: dict, result) -> float:
        """Score reflecting whether the result set matches the expected modality.

        For image queries: fraction of total results that are images.
        For text queries: fraction of total results that are text chunks.

        Args:
            query: A query dict with ``"expected_type"`` field (``"image"`` or ``"text"``).
            result: :class:`~src.retrieval.multimodal_pipeline.RetrievalResult`.

        Returns:
            Float in [0, 1].  ``0.0`` if both result lists are empty.
        """
        n_text = len(result.text_chunks)
        n_images = len(result.images)
        total = n_text + n_images
        if total == 0:
            return 0.0

        expected_type = query.get("expected_type", "text")
        if expected_type == "image":
            return n_images / total
        return n_text / total

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_text_only_vs_multimodal(self, top_k: int = 5) -> dict:
        """Compare keyword precision between text-only and multimodal pipelines.

        Runs both pipelines on the text-type queries only (since text-only
        pipelines cannot retrieve images).

        Args:
            top_k: Number of results to request from each pipeline.

        Returns:
            Dict with keys ``"text_only_precision"``, ``"multimodal_precision"``,
            ``"improvement"`` (multimodal − text_only), and ``"per_query"`` list.

        Raises:
            RuntimeError: When no text-only pipeline was provided at construction.
        """
        if self._text_pipeline is None:
            raise RuntimeError(
                "compare_text_only_vs_multimodal requires a text_pipeline at construction."
            )

        text_queries = [q for q in self._queries if q["expected_type"] == "text"]
        mm_precisions: list[float] = []
        to_precisions: list[float] = []
        per_query: list[dict] = []

        for query in text_queries:
            try:
                mm_result = self._mm_pipeline.retrieve(query["question"], top_k=top_k)
                to_chunks = self._text_pipeline.retrieve(
                    query["question"], top_k=top_k, filters=None
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("compare: query %s failed: %s", query["query_id"], exc)
                continue

            mm_prec = self.text_retrieval_precision(query, mm_result)

            # Wrap text-only result in a minimal result-like object
            class _FakeResult:
                text_chunks = to_chunks
                images: list = []

            to_prec = self.text_retrieval_precision(query, _FakeResult())

            mm_precisions.append(mm_prec)
            to_precisions.append(to_prec)
            per_query.append(
                {
                    "query_id": query["query_id"],
                    "text_only_precision": round(to_prec, 4),
                    "multimodal_precision": round(mm_prec, 4),
                }
            )

        avg_mm = sum(mm_precisions) / len(mm_precisions) if mm_precisions else 0.0
        avg_to = sum(to_precisions) / len(to_precisions) if to_precisions else 0.0

        return {
            "text_only_precision": round(avg_to, 4),
            "multimodal_precision": round(avg_mm, 4),
            "improvement": round(avg_mm - avg_to, 4),
            "per_query": per_query,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_report(self, results: dict) -> str:
        """Format evaluation results as a Markdown table.

        Args:
            results: Dict returned by :meth:`evaluate_retrieval`.

        Returns:
            Multi-line Markdown string suitable for console output or saving.
        """
        lines: list[str] = [
            "# Multimodal RAG Evaluation Report",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total queries | {results.get('total_queries', 0)} |",
            f"| Image queries | {results.get('image_queries', 0)} |",
            f"| Text queries | {results.get('text_queries', 0)} |",
            f"| Image retrieval precision | {results.get('image_precision', 0.0):.4f} |",
            f"| Text retrieval precision | {results.get('text_precision', 0.0):.4f} |",
            f"| Modality accuracy | {results.get('modality_accuracy', 0.0):.4f} |",
            "",
            "## Per-Query Results",
            "",
            "| Query ID | Type | Precision | Modality Acc | Text | Images |",
            "|----------|------|-----------|--------------|------|--------|",
        ]

        for pq in results.get("per_query", []):
            if "error" in pq:
                lines.append(
                    f"| {pq.get('query_id', '?')} | — | ERROR | — | — | — |"
                )
                continue
            lines.append(
                f"| {pq['query_id']} "
                f"| {pq['expected_type']} "
                f"| {pq['precision']:.4f} "
                f"| {pq['modality_accuracy']:.4f} "
                f"| {pq['n_text_chunks']} "
                f"| {pq['n_images']} |"
            )

        return "\n".join(lines)
