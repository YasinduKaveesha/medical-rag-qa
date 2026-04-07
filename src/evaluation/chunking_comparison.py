"""Chunking strategy comparison via RAGAS evaluation.

Runs the same set of test queries through three separate Qdrant collections,
each ingested with a different chunking strategy (fixed_size, sentence,
semantic), and compares faithfulness and answer_relevancy scores.

Collection names default to environment variables ``COLLECTION_FIXED``,
``COLLECTION_SENTENCE``, and ``COLLECTION_SEMANTIC`` or to the built-in
defaults listed below.  The encoder and reranker are shared across all three
strategies.

Typical usage
-------------
::

    python -m src.evaluation.chunking_comparison

Output files are written to ``reports/figures/``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.config import get_settings
from src.evaluation.ragas_eval import (
    build_eval_dataset,
    create_ragas_embeddings,
    create_ragas_llm,
    load_test_queries,
    run_rag_pipeline,
    run_ragas,
)

logger = logging.getLogger(__name__)

# Default collection names — override with env vars or pass explicitly
_DEFAULT_COLLECTIONS: dict[str, str] = {
    "fixed_size": "medical_docs_fixed",
    "sentence": "medical_docs_sentence",
    "semantic": "medical_docs_semantic",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_strategy(
    strategy_name: str,
    collection_name: str,
    queries: list[dict],
    llm_client: Any,
    ragas_llm: Any,
    ragas_embeddings: Any,
    sleep_between_queries: float = 0.0,
) -> pd.DataFrame:
    """Run the full RAG + RAGAS eval loop for one chunking strategy.

    Creates a :class:`src.retrieval.vector_store.QdrantStore` pointing at
    *collection_name*, wraps it in a :class:`src.retrieval.pipeline.RetrievalPipeline`
    (sharing the process-level encoder and reranker singletons), and runs the
    evaluation pipeline.

    Args:
        strategy_name: Human-readable label for this strategy
            (e.g. ``"fixed_size"``).  Added as a ``"strategy"`` column in the
            returned DataFrame.
        collection_name: Qdrant collection to query.
        queries: Test queries as returned by
            :func:`src.evaluation.ragas_eval.load_test_queries`.
        llm_client: :class:`src.generation.llm_client.LLMClient` instance.
        ragas_llm: RAGAS LLM instance for scoring.
        ragas_embeddings: RAGAS embeddings instance for scoring.
        sleep_between_queries: Seconds to sleep between pipeline calls.
            Increase to ``1.0`` on Groq free tier.  Defaults to ``0.0``.

    Returns:
        :class:`pandas.DataFrame` with ``"faithfulness"``,
        ``"answer_relevancy"``, and ``"strategy"`` columns, one row per query.
    """
    from src.embeddings.encoder import get_encoder  # noqa: PLC0415
    from src.retrieval.pipeline import RetrievalPipeline  # noqa: PLC0415
    from src.retrieval.reranker import get_reranker  # noqa: PLC0415
    from src.retrieval.vector_store import QdrantStore  # noqa: PLC0415

    s = get_settings()
    logger.info(
        "evaluate_strategy: %s  collection=%s  queries=%d",
        strategy_name,
        collection_name,
        len(queries),
    )

    store = QdrantStore(
        host=s.qdrant_host,
        port=s.qdrant_port,
        collection_name=collection_name,
    )
    pipeline = RetrievalPipeline(
        encoder=get_encoder(),
        store=store,
        reranker=get_reranker(),
    )

    results = run_rag_pipeline(
        queries, pipeline, llm_client, sleep_between_queries=sleep_between_queries
    )
    dataset = build_eval_dataset(results)
    df = run_ragas(dataset, ragas_llm, ragas_embeddings)
    df["strategy"] = strategy_name

    logger.info(
        "evaluate_strategy: %s done  faithfulness=%.4f  answer_relevancy=%.4f",
        strategy_name,
        df["faithfulness"].mean(),
        df["answer_relevancy"].mean(),
    )
    return df


def run_comparison(
    collections: dict[str, str],
    queries: list[dict],
    llm_client: Any,
    ragas_llm: Any,
    ragas_embeddings: Any,
    sleep_between_strategies: float = 0.0,
    sleep_between_queries: float = 0.0,
) -> pd.DataFrame:
    """Evaluate all chunking strategies and return a combined DataFrame.

    Strategies are evaluated **sequentially** to stay within Groq's free-tier
    rate limits.  Optionally sleeps between strategies.

    Args:
        collections: Mapping of strategy name → Qdrant collection name.
        queries: Test queries as returned by
            :func:`src.evaluation.ragas_eval.load_test_queries`.
        llm_client: Shared LLM client instance.
        ragas_llm: Shared RAGAS LLM instance.
        ragas_embeddings: Shared RAGAS embeddings instance.
        sleep_between_strategies: Seconds to sleep between strategies.
            Defaults to ``0.0``.  Set to ``10.0`` on the Groq free tier.
        sleep_between_queries: Seconds to sleep between individual pipeline
            calls within each strategy.  Defaults to ``0.0``.

    Returns:
        :class:`pandas.DataFrame` combining all per-strategy result DataFrames,
        with columns ``"faithfulness"``, ``"answer_relevancy"``, and
        ``"strategy"``.
    """
    dfs: list[pd.DataFrame] = []
    strategy_items = list(collections.items())

    for i, (strategy, collection) in enumerate(strategy_items):
        logger.info(
            "run_comparison: evaluating %s (%d/%d)",
            strategy,
            i + 1,
            len(strategy_items),
        )
        df = evaluate_strategy(
            strategy_name=strategy,
            collection_name=collection,
            queries=queries,
            llm_client=llm_client,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
            sleep_between_queries=sleep_between_queries,
        )
        dfs.append(df)

        if sleep_between_strategies > 0 and i < len(strategy_items) - 1:
            logger.info(
                "run_comparison: sleeping %.1fs before next strategy",
                sleep_between_strategies,
            )
            time.sleep(sleep_between_strategies)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(
        "run_comparison: finished %d strategies  total_rows=%d",
        len(strategy_items),
        len(combined),
    )
    return combined


def save_comparison(df: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    """Save chunking comparison scores to a timestamped CSV and grouped bar chart.

    Computes per-strategy mean and standard deviation of faithfulness and
    answer_relevancy, then saves a summary CSV and a grouped bar chart with
    error bars (± 1 std dev).

    Args:
        df: Combined DataFrame as returned by :func:`run_comparison`.  Must
            have ``"strategy"``, ``"faithfulness"``, and ``"answer_relevancy"``
            columns.
        output_dir: Directory to write output files into.  Created if absent.

    Returns:
        ``(csv_path, chart_path)`` — absolute paths of the files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"chunking_comparison_{ts}.csv"
    chart_path = output_dir / f"chunking_comparison_{ts}.png"

    # Named aggregation — avoids MultiIndex columns
    summary = (
        df.groupby("strategy")
        .agg(
            faithfulness_mean=("faithfulness", "mean"),
            faithfulness_std=("faithfulness", "std"),
            answer_relevancy_mean=("answer_relevancy", "mean"),
            answer_relevancy_std=("answer_relevancy", "std"),
        )
        .fillna(0)
    )

    summary.reset_index().to_csv(csv_path, index=False)

    # Grouped bar chart
    strategies = summary.index.tolist()
    x = list(range(len(strategies)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [i - width / 2 for i in x],
        summary["faithfulness_mean"],
        width,
        yerr=summary["faithfulness_std"],
        label="Faithfulness",
        color="#2196F3",
        capsize=4,
    )
    ax.bar(
        [i + width / 2 for i in x],
        summary["answer_relevancy_mean"],
        width,
        yerr=summary["answer_relevancy_std"],
        label="Answer Relevancy",
        color="#FF9800",
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean RAGAS Score")
    ax.set_title("Chunking Strategy Comparison")
    ax.legend()

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)

    logger.info("save_comparison: CSV=%s  chart=%s", csv_path, chart_path)
    return csv_path, chart_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _get_collections() -> dict[str, str]:
    """Return collection names from env vars, falling back to defaults."""
    return {
        "fixed_size": os.environ.get("COLLECTION_FIXED", _DEFAULT_COLLECTIONS["fixed_size"]),
        "sentence": os.environ.get(
            "COLLECTION_SENTENCE", _DEFAULT_COLLECTIONS["sentence"]
        ),
        "semantic": os.environ.get("COLLECTION_SEMANTIC", _DEFAULT_COLLECTIONS["semantic"]),
    }


def main() -> None:
    """CLI entry point: compare chunking strategies via RAGAS evaluation.

    Reads ``test_queries.json`` from the ``src/evaluation/`` directory.
    Output files are written to ``reports/figures/``.

    Requires a running Qdrant instance (with three pre-ingested collections),
    a valid ``GROQ_API_KEY``, and the ``sentence-transformers`` model cached
    locally.

    Collection names can be overridden via environment variables:
    ``COLLECTION_FIXED``, ``COLLECTION_SENTENCE``, ``COLLECTION_SEMANTIC``.
    """
    s = get_settings()

    query_path = Path(__file__).parent / "test_queries.json"
    output_dir = Path(__file__).parent.parent.parent / "reports" / "figures"

    queries = load_test_queries(query_path)

    from src.generation.llm_client import get_llm_client  # noqa: PLC0415

    llm_client = get_llm_client()
    ragas_llm = create_ragas_llm(s.groq_api_key, s.llm_base_url, s.llm_model)
    ragas_embeddings = create_ragas_embeddings(s.embedding_model)
    collections = _get_collections()

    df = run_comparison(
        collections=collections,
        queries=queries,
        llm_client=llm_client,
        ragas_llm=ragas_llm,
        ragas_embeddings=ragas_embeddings,
        sleep_between_strategies=10.0,
        sleep_between_queries=1.0,
    )

    csv_path, chart_path = save_comparison(df, output_dir)
    logger.info(
        "Comparison complete.  Results: %s  Chart: %s", csv_path, chart_path
    )


if __name__ == "__main__":
    main()
