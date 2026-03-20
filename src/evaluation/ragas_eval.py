"""RAGAS evaluation for the Medical RAG Q&A pipeline.

Runs faithfulness and answer_relevancy metrics over a set of test queries
using Groq (via the OpenAI-compatible SDK) as the judge LLM and
sentence-transformers as the embedding model.

Typical usage
-------------
::

    python -m src.evaluation.ragas_eval

Output files are written to ``reports/figures/``.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_settings
from src.generation.prompt_builder import build_prompt
from src.generation.refusal import should_refuse

if TYPE_CHECKING:
    from src.generation.llm_client import LLMClient
    from src.retrieval.pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_test_queries(path: str | Path) -> list[dict]:
    """Load and return test query entries from a JSON file.

    Args:
        path: Path to a JSON file containing a list of query dicts.  Each
            dict must have keys ``"id"``, ``"category"``, ``"question"``,
            ``"ground_truth"``, and ``"expected_source_keywords"``.

    Returns:
        List of query dicts as loaded from the file.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        queries = json.load(fh)
    logger.info("load_test_queries: loaded %d queries from %s", len(queries), path)
    return queries


def run_rag_pipeline(
    queries: list[dict],
    pipeline: RetrievalPipeline,
    llm_client: LLMClient,
    sleep_between_queries: float = 0.0,
) -> list[dict]:
    """Run each query through retrieve → generate and collect result records.

    For each query in *queries*, retrieves the top-K chunks, checks the
    refusal condition, and (if not refused) builds a prompt and generates an
    answer.  Refused queries are included in the output with the standard
    refusal phrase so that RAGAS can score them (they will receive low scores,
    which is the correct signal for failed retrievals).

    Args:
        queries: List of query dicts as returned by :func:`load_test_queries`.
        pipeline: Retrieval pipeline instance.  Injected for testability.
        llm_client: LLM client instance.  Injected for testability.
        sleep_between_queries: Seconds to sleep between queries.  Set to
            ``1.0`` or higher when using the Groq free tier to avoid rate
            limits.  Defaults to ``0.0`` (no sleep).

    Returns:
        List of result dicts, one per input query, with keys:
        ``"id"``, ``"category"``, ``"question"``, ``"ground_truth"``,
        ``"answer"``, ``"retrieved_contexts"``, ``"refused"``,
        ``"max_score"``.
    """
    results: list[dict] = []

    for i, q in enumerate(queries):
        logger.info(
            "run_rag_pipeline: query %d/%d  id=%s", i + 1, len(queries), q["id"]
        )
        chunks = pipeline.retrieve(q["question"])
        refused = should_refuse(chunks)
        max_score = max((c["score"] for c in chunks), default=0.0)
        retrieved_contexts = [c["chunk_text"] for c in chunks]

        if refused:
            answer = "I cannot answer from the provided documents."
            logger.info(
                "run_rag_pipeline: id=%s refused (max_score=%.4f)",
                q["id"],
                max_score,
            )
        else:
            prompt = build_prompt(q["question"], chunks)
            answer = llm_client.generate(prompt)

        results.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "answer": answer,
                "retrieved_contexts": retrieved_contexts,
                "refused": refused,
                "max_score": max_score,
            }
        )

        if sleep_between_queries > 0:
            time.sleep(sleep_between_queries)

    refused_count = sum(r["refused"] for r in results)
    logger.info(
        "run_rag_pipeline: completed %d queries (%d refused)",
        len(results),
        refused_count,
    )
    return results


def build_eval_dataset(results: list[dict]) -> Any:
    """Convert RAG pipeline results into a RAGAS EvaluationDataset.

    Maps each result record to a :class:`ragas.dataset_schema.SingleTurnSample`
    with the following field mapping:

    - ``question``          → ``user_input``
    - ``answer``            → ``response``
    - ``retrieved_contexts`` → ``retrieved_contexts``
    - ``ground_truth``      → ``reference``

    Args:
        results: List of result dicts as returned by :func:`run_rag_pipeline`.

    Returns:
        :class:`ragas.dataset_schema.EvaluationDataset` ready to be passed
        to :func:`run_ragas`.
    """
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample  # noqa: PLC0415

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["ground_truth"],
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)
    logger.debug("build_eval_dataset: %d samples created", len(samples))
    return dataset


def run_ragas(dataset: Any, ragas_llm: Any, ragas_embeddings: Any) -> pd.DataFrame:
    """Run RAGAS faithfulness and answer_relevancy on *dataset*.

    Uses ``RunConfig(max_retries=10, max_wait=60)`` and ``batch_size=5`` to
    stay within Groq's free-tier rate limits.  ``raise_exceptions=False``
    ensures a single failure returns NaN rather than aborting the entire run.

    Args:
        dataset: :class:`ragas.dataset_schema.EvaluationDataset` as returned
            by :func:`build_eval_dataset`.
        ragas_llm: RAGAS-compatible LLM (``InstructorBaseRagasLLM``) created
            by :func:`create_ragas_llm`.
        ragas_embeddings: RAGAS-compatible embedding model created by
            :func:`create_ragas_embeddings`.

    Returns:
        :class:`pandas.DataFrame` with one row per sample and columns
        ``"faithfulness"`` and ``"answer_relevancy"`` (floats in [0, 1];
        ``NaN`` for samples where scoring failed).
    """
    from ragas import evaluate  # noqa: PLC0415
    from ragas.metrics.collections.answer_relevancy import AnswerRelevancy  # noqa: PLC0415
    from ragas.metrics.collections.faithfulness import Faithfulness  # noqa: PLC0415
    from ragas.run_config import RunConfig  # noqa: PLC0415

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
    ]

    logger.info("run_ragas: starting evaluation with %d metrics", len(metrics))
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_retries=10, max_wait=60, timeout=180),
        batch_size=5,
        raise_exceptions=False,
        show_progress=True,
    )
    df = result.to_pandas()
    logger.info(
        "run_ragas: done  faithfulness=%.4f  answer_relevancy=%.4f",
        df["faithfulness"].mean(),
        df["answer_relevancy"].mean(),
    )
    return df


def save_results(df: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Save RAGAS scores to a timestamped CSV and bar-chart PNG.

    Args:
        df: DataFrame with at least ``"faithfulness"`` and
            ``"answer_relevancy"`` columns (as returned by :func:`run_ragas`
            after attaching ``"id"`` and ``"category"`` columns).
        output_dir: Directory to write output files into.  Created if it does
            not exist.

    Returns:
        ``(csv_path, chart_path)`` — absolute paths of the files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"ragas_results_{ts}.csv"
    chart_path = output_dir / f"ragas_scores_{ts}.png"

    df.to_csv(csv_path, index=False)

    # Bar chart of mean scores
    means = df[["faithfulness", "answer_relevancy"]].mean()
    labels = ["Faithfulness", "Answer Relevancy"]
    colors = ["#2196F3", "#FF9800"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, means.values, color=colors)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean Score")
    ax.set_title("RAGAS Evaluation Scores")

    for bar, val in zip(bars, means.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)

    logger.info("save_results: CSV=%s  chart=%s", csv_path, chart_path)
    return csv_path, chart_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def create_ragas_llm(groq_api_key: str, llm_base_url: str, llm_model: str) -> Any:
    """Create a RAGAS-compatible LLM pointed at the Groq API.

    Args:
        groq_api_key: Groq API key.
        llm_base_url: Base URL for the Groq OpenAI-compatible endpoint.
        llm_model: Model identifier (e.g. ``"llama-3.3-70b-versatile"``).

    Returns:
        ``InstructorBaseRagasLLM`` instance wrapping a Groq-backed OpenAI
        client.
    """
    from openai import OpenAI  # noqa: PLC0415
    from ragas.llms import llm_factory  # noqa: PLC0415

    client = OpenAI(api_key=groq_api_key, base_url=llm_base_url)
    return llm_factory(llm_model, provider="openai", client=client)


def create_ragas_embeddings(embedding_model: str) -> Any:
    """Create a RAGAS-compatible embedding model using HuggingFace.

    Reuses the same ``sentence-transformers`` model already used by the
    retrieval pipeline, so no additional model is downloaded.

    Args:
        embedding_model: HuggingFace model name (e.g.
            ``"sentence-transformers/all-MiniLM-L6-v2"``).

    Returns:
        :class:`ragas.embeddings.HuggingFaceEmbeddings` instance.
    """
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings  # noqa: PLC0415

    return RagasHFEmbeddings(model=embedding_model)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: load queries, run eval pipeline, save outputs.

    Reads ``test_queries.json`` from the same directory as this file.
    Results are written to ``reports/figures/`` relative to the project root.

    Requires a running Qdrant instance and a valid ``GROQ_API_KEY``.
    """
    s = get_settings()

    query_path = Path(__file__).parent / "test_queries.json"
    output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"

    queries = load_test_queries(query_path)

    from src.generation.llm_client import get_llm_client  # noqa: PLC0415
    from src.retrieval.pipeline import get_pipeline  # noqa: PLC0415

    pipeline = get_pipeline()
    llm_client = get_llm_client()

    ragas_llm = create_ragas_llm(s.groq_api_key, s.llm_base_url, s.llm_model)
    ragas_embeddings = create_ragas_embeddings(s.embedding_model)

    results = run_rag_pipeline(queries, pipeline, llm_client, sleep_between_queries=1.0)
    dataset = build_eval_dataset(results)
    df = run_ragas(dataset, ragas_llm, ragas_embeddings)

    # Re-attach per-query identifiers before saving
    df["id"] = [r["id"] for r in results]
    df["category"] = [r["category"] for r in results]

    csv_path, chart_path = save_results(df, output_dir)
    logger.info("Evaluation complete.  Results: %s  Chart: %s", csv_path, chart_path)


if __name__ == "__main__":
    main()
