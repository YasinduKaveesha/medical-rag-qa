"""Tests for src.frameworks — LangChainPipeline and LlamaIndexPipeline.

All tests are lightweight: no real Qdrant, no real LLM API calls.
LLM and vectorstore/query-engine components are replaced with test doubles.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

# ---------------------------------------------------------------------------
# autouse fixture — reset Settings singleton for every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config as cfg

    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    cfg._settings = None
    yield
    cfg._settings = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DOC = Document(
    page_content="Amitriptyline 10–25 mg at night for neuropathic pain.",
    metadata={
        "source_document": "WHO-EML-2023.pdf",
        "page_number": 3,
    },
)

_ANSWER = "The dose is 10–25 mg at night."


def _make_lc_vectorstore(docs: list[Document] | None = None) -> MagicMock:
    """Return a mock LangChain vectorstore whose retriever returns *docs*."""
    if docs is None:
        docs = [_DOC]
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = docs
    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    return mock_vs


def _make_lc_llm(answer: str = _ANSWER) -> RunnableLambda:
    """Return a LangChain-compatible LLM runnable that always returns *answer*."""
    return RunnableLambda(lambda _: AIMessage(content=answer))


# ===========================================================================
# LangChainPipeline tests
# ===========================================================================

from src.frameworks.langchain_pipeline import LangChainPipeline  # noqa: E402


class TestLangChainPipelineQuery:
    def test_returns_dict_with_required_keys(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose of amitriptyline?")
        assert {"answer", "sources", "latency_ms"}.issubset(result.keys())

    def test_answer_is_string(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose?")
        assert isinstance(result["answer"], str)

    def test_answer_content(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(),
            llm=_make_lc_llm(_ANSWER),
        )
        result = pipeline.query("What is the dose?")
        assert result["answer"] == _ANSWER

    def test_sources_is_list(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose?")
        assert isinstance(result["sources"], list)

    def test_sources_count_matches_docs(self) -> None:
        docs = [_DOC, _DOC]
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(docs),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose?")
        assert len(result["sources"]) == 2

    def test_source_has_required_fields(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose?")
        assert len(result["sources"]) >= 1
        src = result["sources"][0]
        assert {"source_document", "page_number", "chunk_text"}.issubset(src.keys())

    def test_source_document_populated(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose?")
        assert result["sources"][0]["source_document"] == "WHO-EML-2023.pdf"

    def test_source_page_number_populated(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose?")
        assert result["sources"][0]["page_number"] == 3

    def test_latency_ms_is_non_negative_float(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose?")
        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0.0

    def test_empty_docs_returns_empty_sources(self) -> None:
        pipeline = LangChainPipeline(
            vectorstore=_make_lc_vectorstore(docs=[]),
            llm=_make_lc_llm(),
        )
        result = pipeline.query("What is the dose?")
        assert result["sources"] == []

    def test_retriever_invoked_with_question(self) -> None:
        mock_vs = _make_lc_vectorstore()
        pipeline = LangChainPipeline(vectorstore=mock_vs, llm=_make_lc_llm())
        question = "What is the dose of amitriptyline?"
        pipeline.query(question)
        mock_vs.as_retriever().invoke.assert_called_once_with(question)


# ===========================================================================
# LlamaIndexPipeline tests
# ===========================================================================

from src.frameworks.llamaindex_pipeline import LlamaIndexPipeline  # noqa: E402


def _make_li_source_node(text: str = _DOC.page_content) -> MagicMock:
    """Return a mock LlamaIndex NodeWithScore."""
    node = MagicMock()
    node.text = text
    node.metadata = {"source_document": "WHO-EML-2023.pdf", "page_number": 3}
    ns = MagicMock()
    ns.node = node
    ns.score = 0.85
    return ns


def _make_li_response(answer: str = _ANSWER, num_nodes: int = 1) -> MagicMock:
    """Return a mock LlamaIndex Response."""
    resp = MagicMock()
    resp.__str__ = MagicMock(return_value=answer)
    resp.source_nodes = [_make_li_source_node() for _ in range(num_nodes)]
    return resp


def _make_li_query_engine(answer: str = _ANSWER, num_nodes: int = 1) -> MagicMock:
    """Return a mock RetrieverQueryEngine."""
    engine = MagicMock()
    engine.query.return_value = _make_li_response(answer, num_nodes)
    return engine


class TestLlamaIndexPipelineQuery:
    def test_returns_dict_with_required_keys(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine())
        result = pipeline.query("What is the dose of amitriptyline?")
        assert {"answer", "sources", "latency_ms"}.issubset(result.keys())

    def test_answer_is_string(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine())
        result = pipeline.query("What is the dose?")
        assert isinstance(result["answer"], str)

    def test_answer_content(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine(_ANSWER))
        result = pipeline.query("What is the dose?")
        assert result["answer"] == _ANSWER

    def test_sources_is_list(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine())
        result = pipeline.query("What is the dose?")
        assert isinstance(result["sources"], list)

    def test_sources_count_matches_nodes(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine(num_nodes=2))
        result = pipeline.query("What is the dose?")
        assert len(result["sources"]) == 2

    def test_source_has_required_fields(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine())
        result = pipeline.query("What is the dose?")
        src = result["sources"][0]
        assert {"source_document", "page_number", "chunk_text"}.issubset(src.keys())

    def test_source_document_populated(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine())
        result = pipeline.query("What is the dose?")
        assert result["sources"][0]["source_document"] == "WHO-EML-2023.pdf"

    def test_latency_ms_is_non_negative_float(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine())
        result = pipeline.query("What is the dose?")
        assert isinstance(result["latency_ms"], float)
        assert result["latency_ms"] >= 0.0

    def test_query_engine_called_with_question(self) -> None:
        engine = _make_li_query_engine()
        pipeline = LlamaIndexPipeline(_query_engine=engine)
        question = "What is the dose of amitriptyline?"
        pipeline.query(question)
        engine.query.assert_called_once_with(question)

    def test_empty_source_nodes_returns_empty_sources(self) -> None:
        pipeline = LlamaIndexPipeline(_query_engine=_make_li_query_engine(num_nodes=0))
        result = pipeline.query("What is the dose?")
        assert result["sources"] == []
