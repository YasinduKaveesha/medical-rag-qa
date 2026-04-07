"""Tests for multimodal generation extensions — Modules 6a, 6b, 6c."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.generation.citations import extract_multimodal_citations
from src.generation.llm_client import LLMClient
from src.generation.prompt_builder import build_multimodal_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_chunk(
    text: str = "Amoxicillin 500mg three times daily",
    source: str = "who_guidelines.pdf",
    page: int = 5,
) -> dict:
    return {"text": text, "source_document": source, "page_number": page}


def _image(
    caption: str = "chest X-ray showing bilateral infiltrates",
    source_pdf: str = "radiology_atlas.pdf",
    page: int = 12,
) -> dict:
    return {"caption": caption, "source_pdf": source_pdf, "page_number": page, "type": "image"}


# ---------------------------------------------------------------------------
# Module 6a — build_multimodal_prompt
# ---------------------------------------------------------------------------


def test_build_multimodal_prompt_includes_text():
    """Prompt contains the text chunk body and its source reference."""
    chunk = _text_chunk(text="Amoxicillin 500mg", source="who.pdf", page=3)
    prompt = build_multimodal_prompt("What is the dose?", [chunk], [])
    assert "Amoxicillin 500mg" in prompt
    assert "who.pdf" in prompt
    assert "Page 3" in prompt


def test_build_multimodal_prompt_includes_images():
    """Prompt contains the image caption and its source reference."""
    img = _image(caption="bilateral lung infiltrates", source_pdf="atlas.pdf", page=7)
    prompt = build_multimodal_prompt("Describe findings", [], [img])
    assert "bilateral lung infiltrates" in prompt
    assert "atlas.pdf" in prompt
    assert "Page 7" in prompt


def test_build_multimodal_prompt_no_images():
    """IMAGE DESCRIPTIONS section is absent when images list is empty."""
    chunk = _text_chunk()
    prompt = build_multimodal_prompt("query", [chunk], [])
    assert "IMAGE DESCRIPTIONS" not in prompt
    assert "TEXT CONTEXT" in prompt


def test_build_multimodal_prompt_no_text():
    """TEXT CONTEXT section is absent when text_chunks list is empty."""
    img = _image()
    prompt = build_multimodal_prompt("query", [], [img])
    assert "TEXT CONTEXT" not in prompt
    assert "IMAGE DESCRIPTIONS" in prompt


def test_build_multimodal_prompt_includes_query():
    """The user query appears in the prompt."""
    prompt = build_multimodal_prompt(
        "What is the recommended dose of amoxicillin?",
        [_text_chunk()],
        [_image()],
    )
    assert "What is the recommended dose of amoxicillin?" in prompt


# ---------------------------------------------------------------------------
# Module 6b — LLMClient.generate_with_vision
# ---------------------------------------------------------------------------


def _make_llm_client(response_text: str = "vision answer") -> LLMClient:
    """Build an LLMClient with a mocked OpenAI client."""
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value.choices[
        0
    ].message.content = response_text
    with patch("src.generation.llm_client.get_settings") as mock_settings:
        mock_settings.return_value.llm_model = "llama-3.3-70b-versatile"
        mock_settings.return_value.vision_llm_model = "llama-3.2-11b-vision-preview"
        mock_settings.return_value.groq_api_key = "test-key"
        mock_settings.return_value.llm_base_url = "https://api.groq.com/openai/v1"
        client = LLMClient(_client=mock_openai)
    # Attach mock_openai so tests can inspect calls
    client._mock_openai = mock_openai
    return client


def test_generate_with_vision_returns_string(tmp_path):
    """generate_with_vision returns a string response."""
    # Create a real PNG file to satisfy Path.read_bytes()
    from PIL import Image
    img = Image.new("RGB", (10, 10))
    img_path = str(tmp_path / "test.png")
    img.save(img_path)

    client = _make_llm_client("vision answer text")
    with patch("src.generation.llm_client.get_settings") as mock_settings:
        mock_settings.return_value.vision_llm_model = "llama-3.2-11b-vision-preview"
        result = client.generate_with_vision("describe this image", [img_path])

    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_vision_limits_images(tmp_path):
    """Only the first max_images paths are sent to the API."""
    from PIL import Image

    paths = []
    for i in range(5):
        img = Image.new("RGB", (10, 10))
        p = str(tmp_path / f"img_{i}.png")
        img.save(p)
        paths.append(p)

    client = _make_llm_client("answer")
    with patch("src.generation.llm_client.get_settings") as mock_settings:
        mock_settings.return_value.vision_llm_model = "llama-3.2-11b-vision-preview"
        client.generate_with_vision("query", paths, max_images=2)

    call_kwargs = client._mock_openai.chat.completions.create.call_args.kwargs
    content = call_kwargs["messages"][0]["content"]
    image_items = [c for c in content if c.get("type") == "image_url"]
    assert len(image_items) == 2


def test_generate_with_vision_fallback_on_error(tmp_path):
    """generate_with_vision returns text-only generate() output on any exception.

    Strategy: provide a real image file so Path.read_bytes() succeeds, but
    make the API call raise so the except branch is exercised.  The fallback
    `generate()` call then hits the second side_effect item.
    """
    from PIL import Image

    img = Image.new("RGB", (10, 10))
    img_path = str(tmp_path / "real.png")
    img.save(img_path)

    mock_openai = MagicMock()
    # First call: vision API raises; second call: text-only fallback succeeds
    mock_openai.chat.completions.create.side_effect = [
        RuntimeError("vision model not available"),
        MagicMock(choices=[MagicMock(message=MagicMock(content="fallback answer"))]),
    ]

    with patch("src.generation.llm_client.get_settings") as mock_settings:
        mock_settings.return_value.llm_model = "llama-3.3-70b-versatile"
        mock_settings.return_value.vision_llm_model = "llama-3.2-11b-vision-preview"
        mock_settings.return_value.groq_api_key = "key"
        mock_settings.return_value.llm_base_url = "https://api.groq.com/openai/v1"
        client = LLMClient(_client=mock_openai)

    with patch("src.generation.llm_client.get_settings") as mock_settings:
        mock_settings.return_value.vision_llm_model = "llama-3.2-11b-vision-preview"
        result = client.generate_with_vision("some prompt", [img_path])

    assert result == "fallback answer"
    assert mock_openai.chat.completions.create.call_count == 2


def test_generate_still_works():
    """Existing generate() method is unmodified and still returns a string."""
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value.choices[
        0
    ].message.content = "  standard answer  "

    with patch("src.generation.llm_client.get_settings") as mock_settings:
        mock_settings.return_value.llm_model = "llama-3.3-70b-versatile"
        mock_settings.return_value.groq_api_key = "key"
        mock_settings.return_value.llm_base_url = "https://api.groq.com/openai/v1"
        client = LLMClient(_client=mock_openai)

    result = client.generate("What is the dose of amoxicillin?")
    assert result == "standard answer"  # strip() applied
    mock_openai.chat.completions.create.assert_called_once()
    call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0


# ---------------------------------------------------------------------------
# Module 6c — extract_multimodal_citations
# ---------------------------------------------------------------------------


def test_extract_multimodal_citations_finds_text():
    """[Source: X, Page Y] in the answer is matched to the correct text chunk."""
    chunk = {
        "text": "Amoxicillin 500mg three times daily for 7 days",
        "source_document": "who_guidelines.pdf",
        "page_number": 5,
    }
    answer = "Treatment is recommended. [Source: who_guidelines.pdf, Page 5]"
    citations = extract_multimodal_citations(answer, [chunk], [])

    assert len(citations) == 1
    c = citations[0]
    assert c["source_type"] == "text"
    assert c["source_document"] == "who_guidelines.pdf"
    assert c["page_number"] == 5
    assert "Amoxicillin" in (c["chunk_text"] or "")
    assert c["image_path"] is None
    assert c["image_caption"] is None


def test_extract_multimodal_citations_finds_image():
    """[Image from: X, Page Y] in the answer is matched to the correct image."""
    img = {
        "caption": "bilateral pulmonary infiltrates on chest X-ray",
        "source_pdf": "radiology_atlas.pdf",
        "page_number": 12,
        "image_path": "data/extracted_images/atlas_p12_x3.png",
        "type": "image",
    }
    answer = "See the imaging findings. [Image from: radiology_atlas.pdf, Page 12]"
    citations = extract_multimodal_citations(answer, [], [img])

    assert len(citations) == 1
    c = citations[0]
    assert c["source_type"] == "image"
    assert c["page_number"] == 12
    assert c["image_path"] == "data/extracted_images/atlas_p12_x3.png"
    assert "infiltrates" in (c["image_caption"] or "")
    assert c["chunk_text"] is None


def test_extract_multimodal_citations_no_refs():
    """Answer with no reference markers returns an empty list."""
    answer = "I cannot answer from the provided documents."
    citations = extract_multimodal_citations(answer, [_text_chunk()], [_image()])
    assert citations == []
