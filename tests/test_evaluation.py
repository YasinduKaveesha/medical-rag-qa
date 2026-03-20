"""Tests for src.evaluation — ragas_eval and chunking_comparison.

All tests are lightweight: no real Qdrant, no real Groq/RAGAS API calls.
The RAGAS evaluate() function and pipeline components are replaced with
MagicMocks or monkeypatched module-level functions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_QUERY = {
    "id": "q001",
    "category": "dosing",
    "question": "What is the dose of amitriptyline?",
    "ground_truth": "The dose is 10–25 mg at night.",
    "expected_source_keywords": ["amitriptyline", "dose"],
}

_CHUNK = {
    "chunk_text": "Amitriptyline 10–25 mg at night for neuropathic pain.",
    "metadata": {
        "source_document": "WHO-EML-2023.pdf",
        "document_type": "essential_medicines_list",
        "section_title": "2.3 Palliative care",
        "page_number": 3,
        "chunk_id": "cid_0",
        "chunk_index": 0,
        "chunking_strategy": "fixed_size",
    },
    "score": 0.90,
    "reranker_score": 4.5,
}


def _make_chunks(n: int = 2, score: float = 0.90) -> list[dict]:
    return [
        {**_CHUNK, "metadata": {**_CHUNK["metadata"], "chunk_id": f"cid_{i}"}, "score": score}
        for i in range(n)
    ]


def _make_pipeline(chunks: list[dict] | None = None) -> MagicMock:
    m = MagicMock()
    m.retrieve.return_value = chunks if chunks is not None else _make_chunks(2)
    return m


def _make_llm_client(response: str = "The dose is 25 mg [1].") -> MagicMock:
    m = MagicMock()
    m.generate.return_value = response
    return m


def _ragas_scores_df(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {"faithfulness": [0.85] * n, "answer_relevancy": [0.80] * n}
    )


# ---------------------------------------------------------------------------
# autouse fixture — fresh Settings singleton for every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config as cfg

    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    cfg._settings = None
    yield
    cfg._settings = None


# ===========================================================================
# load_test_queries tests
# ===========================================================================

from src.evaluation.ragas_eval import load_test_queries  # noqa: E402


def test_load_test_queries_returns_list() -> None:
    path = Path(__file__).parent.parent / "src" / "evaluation" / "test_queries.json"
    result = load_test_queries(path)
    assert isinstance(result, list)


def test_load_test_queries_has_required_fields() -> None:
    path = Path(__file__).parent.parent / "src" / "evaluation" / "test_queries.json"
    queries = load_test_queries(path)
    required = {"id", "category", "question", "ground_truth", "expected_source_keywords"}
    for q in queries:
        assert required.issubset(q.keys()), f"Missing fields in {q}"


def test_load_test_queries_count() -> None:
    path = Path(__file__).parent.parent / "src" / "evaluation" / "test_queries.json"
    queries = load_test_queries(path)
    assert len(queries) >= 20


# ===========================================================================
# run_rag_pipeline tests
# ===========================================================================

from src.evaluation.ragas_eval import run_rag_pipeline  # noqa: E402


def test_run_rag_pipeline_calls_retrieve() -> None:
    pipeline = _make_pipeline()
    llm = _make_llm_client()
    run_rag_pipeline([_QUERY], pipeline, llm)
    pipeline.retrieve.assert_called_once_with(_QUERY["question"])


def test_run_rag_pipeline_calls_generate() -> None:
    """LLM generate is called for non-refused queries."""
    pipeline = _make_pipeline(_make_chunks(2, score=0.90))
    llm = _make_llm_client()
    run_rag_pipeline([_QUERY], pipeline, llm)
    llm.generate.assert_called_once()


def test_run_rag_pipeline_result_fields() -> None:
    pipeline = _make_pipeline()
    llm = _make_llm_client()
    results = run_rag_pipeline([_QUERY], pipeline, llm)
    assert len(results) == 1
    required = {"id", "category", "question", "ground_truth", "answer", "retrieved_contexts",
                "refused", "max_score"}
    assert required.issubset(results[0].keys())


def test_run_rag_pipeline_refusal_sets_flag() -> None:
    """Empty retrieve result → refused=True, generate not called."""
    pipeline = _make_pipeline([])  # empty → should_refuse fires
    llm = _make_llm_client()
    results = run_rag_pipeline([_QUERY], pipeline, llm)
    assert results[0]["refused"] is True
    llm.generate.assert_not_called()


def test_run_rag_pipeline_refused_answer_text() -> None:
    pipeline = _make_pipeline([])
    llm = _make_llm_client()
    results = run_rag_pipeline([_QUERY], pipeline, llm)
    assert results[0]["answer"] == "I cannot answer from the provided documents."


def test_run_rag_pipeline_retrieved_contexts_are_strings() -> None:
    pipeline = _make_pipeline(_make_chunks(2))
    llm = _make_llm_client()
    results = run_rag_pipeline([_QUERY], pipeline, llm)
    contexts = results[0]["retrieved_contexts"]
    assert isinstance(contexts, list)
    assert all(isinstance(c, str) for c in contexts)


def test_run_rag_pipeline_max_score_populated() -> None:
    pipeline = _make_pipeline(_make_chunks(2, score=0.90))
    llm = _make_llm_client()
    results = run_rag_pipeline([_QUERY], pipeline, llm)
    assert abs(results[0]["max_score"] - 0.90) < 1e-6


# ===========================================================================
# build_eval_dataset tests
# ===========================================================================

from src.evaluation.ragas_eval import build_eval_dataset  # noqa: E402


def _make_results(n: int = 1) -> list[dict]:
    return [
        {
            "id": f"q{i:03d}",
            "category": "dosing",
            "question": "What is the dose?",
            "ground_truth": "25 mg.",
            "answer": "The dose is 25 mg.",
            "retrieved_contexts": ["Context text here."],
            "refused": False,
            "max_score": 0.85,
        }
        for i in range(n)
    ]


def test_build_eval_dataset_type() -> None:
    from ragas.dataset_schema import EvaluationDataset  # noqa: PLC0415

    dataset = build_eval_dataset(_make_results(1))
    assert isinstance(dataset, EvaluationDataset)


def test_build_eval_dataset_sample_count() -> None:
    dataset = build_eval_dataset(_make_results(3))
    assert len(dataset.samples) == 3


def test_build_eval_dataset_user_input() -> None:
    results = _make_results(1)
    dataset = build_eval_dataset(results)
    assert dataset.samples[0].user_input == results[0]["question"]


def test_build_eval_dataset_response() -> None:
    results = _make_results(1)
    dataset = build_eval_dataset(results)
    assert dataset.samples[0].response == results[0]["answer"]


def test_build_eval_dataset_contexts() -> None:
    results = _make_results(1)
    dataset = build_eval_dataset(results)
    assert isinstance(dataset.samples[0].retrieved_contexts, list)
    assert dataset.samples[0].retrieved_contexts == results[0]["retrieved_contexts"]


# ===========================================================================
# save_results tests
# ===========================================================================

from src.evaluation.ragas_eval import save_results  # noqa: E402


def test_save_results_creates_csv(tmp_path: Path) -> None:
    df = _ragas_scores_df(3)
    csv_path, _ = save_results(df, tmp_path)
    assert csv_path.exists()


def test_save_results_creates_chart(tmp_path: Path) -> None:
    df = _ragas_scores_df(3)
    _, chart_path = save_results(df, tmp_path)
    assert chart_path.exists()


def test_save_results_csv_columns(tmp_path: Path) -> None:
    df = _ragas_scores_df(3)
    csv_path, _ = save_results(df, tmp_path)
    saved = pd.read_csv(csv_path)
    assert "faithfulness" in saved.columns
    assert "answer_relevancy" in saved.columns


# ===========================================================================
# chunking_comparison — run_comparison tests
# ===========================================================================

import src.evaluation.chunking_comparison as cc_module  # noqa: E402
from src.evaluation.chunking_comparison import run_comparison  # noqa: E402


def _mock_evaluate_strategy(
    strategy_name: str,
    collection_name: str,
    queries: list[dict],
    llm_client: object,
    ragas_llm: object,
    ragas_embeddings: object,
    sleep_between_queries: float = 0.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {"faithfulness": [0.80], "answer_relevancy": [0.75], "strategy": [strategy_name]}
    )


_THREE_COLLECTIONS = {
    "fixed_size": "medical_docs_fixed",
    "sentence": "medical_docs_sentence",
    "semantic": "medical_docs_semantic",
}


def test_run_comparison_calls_all_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []

    def tracking_evaluate(
        strategy_name: str,
        collection_name: str,
        queries: list[dict],
        llm_client: object,
        ragas_llm: object,
        ragas_embeddings: object,
        sleep_between_queries: float = 0.0,
    ) -> pd.DataFrame:
        called.append(strategy_name)
        return _mock_evaluate_strategy(
            strategy_name, collection_name, queries, llm_client, ragas_llm, ragas_embeddings
        )

    monkeypatch.setattr(cc_module, "evaluate_strategy", tracking_evaluate)

    run_comparison(
        _THREE_COLLECTIONS,
        [],
        MagicMock(),
        MagicMock(),
        MagicMock(),
    )
    assert len(called) == 3
    assert set(called) == {"fixed_size", "sentence", "semantic"}


def test_run_comparison_result_has_strategy_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cc_module, "evaluate_strategy", _mock_evaluate_strategy)
    result = run_comparison(_THREE_COLLECTIONS, [], MagicMock(), MagicMock(), MagicMock())
    assert "strategy" in result.columns


# ===========================================================================
# chunking_comparison — save_comparison tests
# ===========================================================================

from src.evaluation.chunking_comparison import save_comparison  # noqa: E402


def _make_comparison_df() -> pd.DataFrame:
    rows = []
    for strategy in ["fixed_size", "sentence", "semantic"]:
        for _ in range(3):
            rows.append(
                {"strategy": strategy, "faithfulness": 0.80, "answer_relevancy": 0.75}
            )
    return pd.DataFrame(rows)


def test_save_comparison_creates_csv(tmp_path: Path) -> None:
    df = _make_comparison_df()
    csv_path, _ = save_comparison(df, tmp_path)
    assert csv_path.exists()


def test_save_comparison_creates_chart(tmp_path: Path) -> None:
    df = _make_comparison_df()
    _, chart_path = save_comparison(df, tmp_path)
    assert chart_path.exists()


def test_save_comparison_all_strategies_in_csv(tmp_path: Path) -> None:
    df = _make_comparison_df()
    csv_path, _ = save_comparison(df, tmp_path)
    saved = pd.read_csv(csv_path)
    strategies = set(saved["strategy"].tolist())
    assert strategies == {"fixed_size", "sentence", "semantic"}
