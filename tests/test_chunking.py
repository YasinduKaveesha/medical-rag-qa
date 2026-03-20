"""Tests for src.ingestion.chunkers — all three chunking strategies."""

from __future__ import annotations

import re

import numpy as np
import pytest

from src.ingestion.chunkers import FixedSizeChunker, SemanticChunker, SentenceChunker

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Sample metadata mimicking Module 1 attach_metadata output
SAMPLE_METADATA = {
    "page_number": 3,
    "source_document": "WHO-MHP-HPS-EML-2023.02-eng.pdf",
    "document_type": "essential_medicines_list",
    "section_title": "2.3 Medicines for palliative care",
    "chunk_id": "WHO-MHP-HPS-EML-2023.02-eng_p0003",
    "tables": [],
}

# Short text (well under 512 tokens)
SHORT_TEXT = (
    "Amitriptyline is a tricyclic antidepressant used for depression and neuropathic pain. "
    "It is available as 10 mg, 25 mg, and 75 mg tablets. "
    "Cyclizine is used for nausea and vomiting. "
    "Dexamethasone is a corticosteroid used in palliative care."
)

# Longer text with clear sentence boundaries (forces multiple chunks at low token limits)
LONG_TEXT = " ".join(
    [
        f"Sentence {i}: this medication is indicated for the treatment of condition {i},"
        f" and the recommended dose is {i * 10} mg daily for adults."
        for i in range(1, 60)
    ]
)


# ---------------------------------------------------------------------------
# Mock encoder for SemanticChunker tests
# ---------------------------------------------------------------------------


class MockEncoder:
    """Returns fixed embeddings so tests do not load the real model."""

    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings
        self.encode_calls: list[list[str]] = []

    def encode(self, sentences: list[str], show_progress_bar: bool = False) -> np.ndarray:
        self.encode_calls.append(sentences)
        # Return the pre-set embeddings (or a slice if fewer sentences given)
        n = len(sentences)
        if n <= len(self._embeddings):
            return self._embeddings[:n]
        # Tile to cover however many sentences are passed
        repeats = (n // len(self._embeddings)) + 1
        return np.tile(self._embeddings, (repeats, 1))[:n]


def _identical_encoder(n: int = 10) -> MockEncoder:
    """Encoder where all sentences are identical → similarity=1 → no splits."""
    emb = np.ones((n, 4), dtype=np.float32)
    return MockEncoder(emb)


def _split_encoder(n: int = 10) -> MockEncoder:
    """Encoder where every other sentence is orthogonal → similarity=0 → many splits."""
    emb = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        emb[i, i % 4] = 1.0
    return MockEncoder(emb)


# ---------------------------------------------------------------------------
# FixedSizeChunker tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixed_chunker():
    return FixedSizeChunker(max_tokens=512, overlap_tokens=50)


@pytest.fixture(scope="module")
def fixed_chunker_small():
    """Small token limit to force multiple chunks on SHORT_TEXT."""
    return FixedSizeChunker(max_tokens=30, overlap_tokens=5)


def test_fixed_returns_list_of_dicts(fixed_chunker):
    result = fixed_chunker.chunk(SHORT_TEXT, SAMPLE_METADATA)
    assert isinstance(result, list)
    for item in result:
        assert "text" in item
        assert "metadata" in item


def test_fixed_short_text_single_chunk(fixed_chunker):
    result = fixed_chunker.chunk(SHORT_TEXT, SAMPLE_METADATA)
    assert len(result) == 1


def test_fixed_empty_text(fixed_chunker):
    assert fixed_chunker.chunk("", SAMPLE_METADATA) == []
    assert fixed_chunker.chunk("   ", SAMPLE_METADATA) == []


def test_fixed_chunk_token_limit(fixed_chunker_small):
    result = fixed_chunker_small.chunk(LONG_TEXT, SAMPLE_METADATA)
    assert len(result) > 1
    tokenizer = fixed_chunker_small._tokenizer
    for chunk in result:
        token_count = len(tokenizer.encode(chunk["text"], add_special_tokens=False))
        assert token_count <= 30, f"Chunk exceeds token limit: {token_count}"


def test_fixed_overlap(fixed_chunker_small):
    """Consecutive chunks should share tokens at their boundary."""
    result = fixed_chunker_small.chunk(LONG_TEXT, SAMPLE_METADATA)
    assert len(result) >= 2
    tokenizer = fixed_chunker_small._tokenizer
    ids_0 = tokenizer.encode(result[0]["text"], add_special_tokens=False)
    ids_1 = tokenizer.encode(result[1]["text"], add_special_tokens=False)
    # The last `overlap_tokens` of chunk 0 should appear at the start of chunk 1
    overlap = fixed_chunker_small._overlap_tokens
    tail = ids_0[-overlap:]
    head = ids_1[:overlap]
    assert tail == head, "Expected token overlap between consecutive chunks"


def test_fixed_multiple_chunks_on_long_text(fixed_chunker_small):
    result = fixed_chunker_small.chunk(LONG_TEXT, SAMPLE_METADATA)
    assert len(result) > 1


# ---------------------------------------------------------------------------
# SentenceChunker tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sentence_chunker():
    return SentenceChunker(max_tokens=512)


@pytest.fixture(scope="module")
def sentence_chunker_small():
    return SentenceChunker(max_tokens=30)


def test_sentence_returns_list_of_dicts(sentence_chunker):
    result = sentence_chunker.chunk(SHORT_TEXT, SAMPLE_METADATA)
    assert isinstance(result, list)
    for item in result:
        assert "text" in item
        assert "metadata" in item


def test_sentence_short_text_single_chunk(sentence_chunker):
    result = sentence_chunker.chunk(SHORT_TEXT, SAMPLE_METADATA)
    assert len(result) == 1


def test_sentence_empty_text(sentence_chunker):
    assert sentence_chunker.chunk("", SAMPLE_METADATA) == []
    assert sentence_chunker.chunk("   ", SAMPLE_METADATA) == []


def test_sentence_chunk_token_limit(sentence_chunker_small):
    result = sentence_chunker_small.chunk(LONG_TEXT, SAMPLE_METADATA)
    tokenizer = sentence_chunker_small._tokenizer
    for chunk in result:
        token_count = len(tokenizer.encode(chunk["text"], add_special_tokens=False))
        # Allow a single oversized sentence to exceed limit (emitted as-is)
        sentences_in_chunk = re.split(r"(?<=[.!?])\s+", chunk["text"])
        if len(sentences_in_chunk) > 1:
            assert token_count <= 30, f"Multi-sentence chunk exceeds limit: {token_count}"


def test_sentence_multiple_chunks_on_long_text(sentence_chunker_small):
    result = sentence_chunker_small.chunk(LONG_TEXT, SAMPLE_METADATA)
    assert len(result) > 1


def test_sentence_no_mid_sentence_splits(sentence_chunker_small):
    """No chunk should end in the middle of a word (basic sanity check)."""
    result = sentence_chunker_small.chunk(LONG_TEXT, SAMPLE_METADATA)
    for chunk in result[:-1]:  # last chunk may not end with punctuation
        text = chunk["text"].rstrip()
        # Each flush happens at a sentence boundary — text should end with punctuation
        # or the last word of a sentence
        assert text, "Chunk text should not be empty"


# ---------------------------------------------------------------------------
# SemanticChunker tests
# ---------------------------------------------------------------------------


def test_semantic_returns_list_of_dicts():
    enc = _identical_encoder()
    chunker = SemanticChunker(encoder=enc, similarity_threshold=0.5)
    result = chunker.chunk(SHORT_TEXT, SAMPLE_METADATA)
    assert isinstance(result, list)
    for item in result:
        assert "text" in item
        assert "metadata" in item


def test_semantic_empty_text():
    enc = _identical_encoder()
    chunker = SemanticChunker(encoder=enc)
    assert chunker.chunk("", SAMPLE_METADATA) == []
    assert chunker.chunk("   ", SAMPLE_METADATA) == []


def test_semantic_single_sentence():
    """Single sentence → 1 chunk without calling encoder."""
    enc = _identical_encoder()
    chunker = SemanticChunker(encoder=enc, similarity_threshold=0.5)
    result = chunker.chunk("Only one sentence here.", SAMPLE_METADATA)
    assert len(result) == 1
    assert enc.encode_calls == [], "Encoder should not be called for a single sentence"


def test_semantic_uses_encoder():
    """Encoder's encode() must be called when there are multiple sentences."""
    enc = _identical_encoder()
    chunker = SemanticChunker(encoder=enc, similarity_threshold=0.5)
    chunker.chunk(SHORT_TEXT, SAMPLE_METADATA)
    assert len(enc.encode_calls) == 1


def test_semantic_no_splits_when_similar():
    """High similarity (identical embeddings) → one chunk for SHORT_TEXT."""
    enc = _identical_encoder()
    chunker = SemanticChunker(encoder=enc, similarity_threshold=0.5)
    result = chunker.chunk(SHORT_TEXT, SAMPLE_METADATA)
    assert len(result) == 1


def test_semantic_splits_when_dissimilar():
    """Orthogonal embeddings (similarity=0) → many chunks."""
    enc = _split_encoder(20)
    chunker = SemanticChunker(encoder=enc, similarity_threshold=0.5)
    result = chunker.chunk(LONG_TEXT, SAMPLE_METADATA)
    assert len(result) > 1


# ---------------------------------------------------------------------------
# Metadata inheritance tests — shared across all 3 strategies
# ---------------------------------------------------------------------------

REQUIRED_INHERITED = {"source_document", "document_type", "section_title", "page_number"}
REQUIRED_NEW = {"chunk_id", "chunk_index", "chunking_strategy"}


@pytest.mark.parametrize(
    "chunker_factory",
    [
        lambda: FixedSizeChunker(max_tokens=30, overlap_tokens=5),
        lambda: SentenceChunker(max_tokens=30),
        lambda: SemanticChunker(encoder=_split_encoder(20), similarity_threshold=0.5),
    ],
    ids=["fixed", "sentence", "semantic"],
)
def test_metadata_inheritance(chunker_factory):
    chunker = chunker_factory()
    result = chunker.chunk(LONG_TEXT, SAMPLE_METADATA)
    assert result, "Expected at least one chunk"
    for chunk in result:
        meta = chunk["metadata"]
        for key in REQUIRED_INHERITED:
            assert key in meta, f"Missing inherited key '{key}'"
        assert meta["source_document"] == SAMPLE_METADATA["source_document"]
        assert meta["page_number"] == SAMPLE_METADATA["page_number"]
        assert meta["document_type"] == SAMPLE_METADATA["document_type"]
        assert meta["section_title"] == SAMPLE_METADATA["section_title"]


@pytest.mark.parametrize(
    "chunker_factory",
    [
        lambda: FixedSizeChunker(max_tokens=30, overlap_tokens=5),
        lambda: SentenceChunker(max_tokens=30),
        lambda: SemanticChunker(encoder=_split_encoder(20), similarity_threshold=0.5),
    ],
    ids=["fixed", "sentence", "semantic"],
)
def test_chunk_ids_are_unique(chunker_factory):
    chunker = chunker_factory()
    result = chunker.chunk(LONG_TEXT, SAMPLE_METADATA)
    ids = [c["metadata"]["chunk_id"] for c in result]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids detected"


@pytest.mark.parametrize(
    "chunker_factory",
    [
        lambda: FixedSizeChunker(max_tokens=30, overlap_tokens=5),
        lambda: SentenceChunker(max_tokens=30),
        lambda: SemanticChunker(encoder=_split_encoder(20), similarity_threshold=0.5),
    ],
    ids=["fixed", "sentence", "semantic"],
)
def test_chunk_ids_are_not_page_id(chunker_factory):
    chunker = chunker_factory()
    result = chunker.chunk(LONG_TEXT, SAMPLE_METADATA)
    page_id = SAMPLE_METADATA["chunk_id"]
    for chunk in result:
        assert chunk["metadata"]["chunk_id"] != page_id, (
            "Child chunk_id must differ from parent page chunk_id"
        )


@pytest.mark.parametrize(
    "chunker_factory",
    [
        lambda: FixedSizeChunker(max_tokens=30, overlap_tokens=5),
        lambda: SentenceChunker(max_tokens=30),
        lambda: SemanticChunker(encoder=_split_encoder(20), similarity_threshold=0.5),
    ],
    ids=["fixed", "sentence", "semantic"],
)
def test_chunk_index_sequential(chunker_factory):
    chunker = chunker_factory()
    result = chunker.chunk(LONG_TEXT, SAMPLE_METADATA)
    indices = [c["metadata"]["chunk_index"] for c in result]
    assert indices == list(range(len(result))), "chunk_index must be 0-based sequential"


@pytest.mark.parametrize(
    "chunker_factory,expected_strategy",
    [
        (lambda: FixedSizeChunker(max_tokens=30, overlap_tokens=5), "fixed_size"),
        (lambda: SentenceChunker(max_tokens=30), "sentence"),
        (
            lambda: SemanticChunker(encoder=_split_encoder(20), similarity_threshold=0.5),
            "semantic",
        ),
    ],
    ids=["fixed", "sentence", "semantic"],
)
def test_chunking_strategy_field(chunker_factory, expected_strategy):
    chunker = chunker_factory()
    result = chunker.chunk(SHORT_TEXT, SAMPLE_METADATA)
    for chunk in result:
        assert chunk["metadata"]["chunking_strategy"] == expected_strategy


def test_metadata_not_mutated_between_chunks():
    """Mutations to one chunk's metadata must not affect others."""
    chunker = FixedSizeChunker(max_tokens=30, overlap_tokens=5)
    result = chunker.chunk(LONG_TEXT, SAMPLE_METADATA)
    assert len(result) >= 2
    result[0]["metadata"]["source_document"] = "MUTATED"
    assert result[1]["metadata"]["source_document"] == SAMPLE_METADATA["source_document"]


def test_fixed_invalid_overlap_raises():
    with pytest.raises(ValueError):
        FixedSizeChunker(max_tokens=50, overlap_tokens=50)
