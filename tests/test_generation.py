"""Tests for src.generation — prompt_builder, llm_client, refusal, citations."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

QUERY = "What is the recommended dose of amitriptyline?"

_META = {
    "source_document": "WHO-MHP-HPS-EML-2023.02-eng.pdf",
    "document_type": "essential_medicines_list",
    "section_title": "2.3 Medicines for palliative care",
    "page_number": 3,
    "chunk_id": "abc123",
    "chunk_index": 0,
    "chunking_strategy": "fixed_size",
}


def _make_chunks(n: int = 2, scores: list[float] | None = None) -> list[dict]:
    if scores is None:
        scores = [0.9 - i * 0.1 for i in range(n)]
    chunks = []
    for i in range(n):
        meta = {**_META, "chunk_id": f"cid_{i}", "page_number": i + 1}
        chunks.append(
            {
                "chunk_text": f"Chunk text number {i}.",
                "metadata": meta,
                "score": scores[i],
                "reranker_score": 5.0 - i,
            }
        )
    return chunks


# ===========================================================================
# prompt_builder tests
# ===========================================================================

from src.generation.prompt_builder import build_prompt  # noqa: E402


def test_build_prompt_returns_string() -> None:
    result = build_prompt(QUERY, _make_chunks(2))
    assert isinstance(result, str)


def test_build_prompt_contains_query() -> None:
    result = build_prompt(QUERY, _make_chunks(1))
    assert QUERY in result


def test_build_prompt_contains_chunk_text() -> None:
    chunks = _make_chunks(2)
    result = build_prompt(QUERY, chunks)
    for chunk in chunks:
        assert chunk["chunk_text"] in result


def test_build_prompt_contains_source_document() -> None:
    chunks = _make_chunks(1)
    result = build_prompt(QUERY, chunks)
    assert chunks[0]["metadata"]["source_document"] in result


def test_build_prompt_contains_page_number() -> None:
    chunks = _make_chunks(1)
    result = build_prompt(QUERY, chunks)
    assert str(chunks[0]["metadata"]["page_number"]) in result


def test_build_prompt_numbered_chunks() -> None:
    result = build_prompt(QUERY, _make_chunks(3))
    assert "[1]" in result
    assert "[2]" in result
    assert "[3]" in result


def test_build_prompt_system_instruction_present() -> None:
    result = build_prompt(QUERY, _make_chunks(1))
    assert "ONLY the information provided in the context" in result


def test_build_prompt_cannot_answer_phrase_present() -> None:
    result = build_prompt(QUERY, _make_chunks(1))
    assert "I cannot answer from the provided documents" in result


def test_build_prompt_empty_chunks() -> None:
    result = build_prompt(QUERY, [])
    assert result == ""


# ===========================================================================
# llm_client tests
# ===========================================================================

import src.generation.llm_client as llm_module  # noqa: E402
from src.generation.llm_client import LLMClient, get_llm_client  # noqa: E402


def _make_mock_openai(response_text: str = "The dose is 25 mg.") -> MagicMock:
    """Build a MagicMock standing in for openai.OpenAI."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = f"  {response_text}  "
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture(autouse=True)
def reset_llm_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config as cfg

    cfg._settings = None
    llm_module._llm_client = None
    yield
    cfg._settings = None
    llm_module._llm_client = None


def test_generate_returns_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    client = LLMClient(_client=_make_mock_openai())
    result = client.generate("some prompt")
    assert isinstance(result, str)


def test_generate_calls_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock = _make_mock_openai()
    client = LLMClient(_client=mock)
    client.generate("prompt")
    mock.chat.completions.create.assert_called_once()


def test_generate_passes_prompt_as_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock = _make_mock_openai()
    client = LLMClient(_client=mock)
    client.generate("my prompt")
    call_kwargs = mock.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert any(m["role"] == "user" and m["content"] == "my prompt" for m in messages)


def test_generate_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock = _make_mock_openai()
    client = LLMClient(model="llama-3.3-70b-versatile", _client=mock)
    client.generate("prompt")
    call_kwargs = mock.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"


def test_generate_temperature_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock = _make_mock_openai()
    client = LLMClient(_client=mock)
    client.generate("prompt")
    call_kwargs = mock.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0


def test_generate_returns_stripped_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    mock = _make_mock_openai("  answer with spaces  ")
    client = LLMClient(_client=mock)
    result = client.generate("prompt")
    assert result == "answer with spaces"


def test_model_name_property(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    client = LLMClient(model="test-model", _client=_make_mock_openai())
    assert client.model == "test-model"


def test_get_llm_client_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(
        llm_module,
        "LLMClient",
        lambda: LLMClient(_client=_make_mock_openai()),
    )
    c1 = get_llm_client()
    c2 = get_llm_client()
    assert c1 is c2


def test_get_llm_client_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setattr(
        llm_module,
        "LLMClient",
        lambda: LLMClient(_client=_make_mock_openai()),
    )
    c1 = get_llm_client()
    llm_module._llm_client = None
    c2 = get_llm_client()
    assert c1 is not c2


# ===========================================================================
# refusal tests
# ===========================================================================

from src.generation.refusal import should_refuse  # noqa: E402


@pytest.fixture(autouse=True)
def reset_settings_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config as cfg

    cfg._settings = None
    yield
    cfg._settings = None


def test_should_refuse_empty_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    assert should_refuse([]) is True


def test_should_refuse_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    chunks = _make_chunks(2, scores=[0.10, 0.20])
    assert should_refuse(chunks) is True


def test_should_not_refuse_above_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    chunks = _make_chunks(2, scores=[0.80, 0.60])
    assert should_refuse(chunks) is False


def test_should_refuse_exactly_at_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """max score == threshold should NOT refuse (boundary: < not <=)."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    chunks = _make_chunks(1, scores=[0.35])
    assert should_refuse(chunks) is False


def test_should_refuse_uses_max_not_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    """One high-score chunk must prevent refusal even if others are very low."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    chunks = _make_chunks(3, scores=[0.05, 0.05, 0.90])
    assert should_refuse(chunks) is False


def test_should_refuse_custom_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threshold is read from Settings.similarity_threshold."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("SIMILARITY_THRESHOLD", "0.70")
    chunks = _make_chunks(1, scores=[0.50])
    # 0.50 < 0.70 -> should refuse
    assert should_refuse(chunks) is True


# ===========================================================================
# citations tests
# ===========================================================================

from src.generation.citations import extract_citations  # noqa: E402


def test_extract_citations_returns_list() -> None:
    result = extract_citations("The dose is 25 mg [1].", _make_chunks(2))
    assert isinstance(result, list)


def test_extract_citations_empty_answer() -> None:
    assert extract_citations("", _make_chunks(2)) == []


def test_extract_citations_empty_chunks() -> None:
    assert extract_citations("Some answer [1].", []) == []


def test_extract_citations_no_markers() -> None:
    result = extract_citations("This answer has no citation markers.", _make_chunks(2))
    assert result == []


def test_extract_citations_single_citation() -> None:
    chunks = _make_chunks(2)
    answer = "The dose is 25 mg [1]."
    result = extract_citations(answer, chunks)
    assert len(result) == 1
    assert result[0]["source_chunk"] == chunks[0]["chunk_text"]
    assert result[0]["page_number"] == chunks[0]["metadata"]["page_number"]


def test_extract_citations_multiple_markers() -> None:
    chunks = _make_chunks(2)
    answer = "Amitriptyline is used for pain [1]. The dose is 25 mg [2]."
    result = extract_citations(answer, chunks)
    assert len(result) == 2


def test_extract_citations_claim_text() -> None:
    chunks = _make_chunks(1)
    sentence = "The recommended dose is 25 mg [1]."
    result = extract_citations(sentence, chunks)
    assert result[0]["claim"] == sentence.strip()


def test_extract_citations_out_of_range_skipped() -> None:
    """[99] with only 2 chunks must not crash and must produce no citation."""
    chunks = _make_chunks(2)
    result = extract_citations("Some answer [99].", chunks)
    assert result == []


def test_extract_citations_source_document_present() -> None:
    chunks = _make_chunks(1)
    result = extract_citations("Answer here [1].", chunks)
    assert result[0]["source_document"] == chunks[0]["metadata"]["source_document"]


def test_extract_citations_zero_index_skipped() -> None:
    """[0] is not a valid 1-based index and must be skipped."""
    chunks = _make_chunks(2)
    result = extract_citations("Answer [0].", chunks)
    assert result == []
