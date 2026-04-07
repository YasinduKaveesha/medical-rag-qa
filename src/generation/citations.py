"""Citation extraction for the Medical RAG Q&A generation step.

Maps sentences in the LLM's answer back to the source chunks referenced by
``[N]`` index markers.  The prompt built by
:mod:`src.generation.prompt_builder` numbers context chunks ``[1]``,
``[2]``, … so the LLM can cite them inline, and this module resolves those
markers back to concrete source metadata.

Typical usage
-------------
::

    from src.generation.citations import extract_citations

    citations = extract_citations(answer, chunks)
    # citations[i]["claim"]           — sentence that made the claim
    # citations[i]["source_chunk"]    — chunk text supporting it
    # citations[i]["page_number"]     — page in the source document
    # citations[i]["source_document"] — filename of the source document
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Sentence boundary: split after . ! ? followed by whitespace
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# Citation marker: [N] where N is one or more digits
_MARKER_RE = re.compile(r"\[(\d+)\]")


def extract_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Map answer sentences to the source chunks they cite.

    Splits *answer* into sentences, finds ``[N]`` citation markers within
    each sentence, and looks up the corresponding chunk by 1-based index.
    Out-of-range indices (e.g. the LLM hallucinated ``[99]`` when only 3
    chunks were provided) are silently skipped with a warning log.

    Args:
        answer: The LLM-generated answer string.  May contain ``[N]``
            citation markers inserted by the model in response to the
            system prompt instruction.
        chunks: List of retrieval result dicts as returned by
            :meth:`src.retrieval.pipeline.RetrievalPipeline.retrieve`.
            Each dict must have keys ``"chunk_text"`` and ``"metadata"``
            (with sub-keys ``"page_number"`` and ``"source_document"``).

    Returns:
        List of citation dicts.  Each dict contains:

        - ``"claim"`` — the sentence from *answer* that contains the marker.
        - ``"source_chunk"`` — the ``chunk_text`` of the referenced chunk.
        - ``"page_number"`` — integer page number from the chunk's metadata.
        - ``"source_document"`` — filename from the chunk's metadata.

        Returns ``[]`` when *answer* is empty, *chunks* is empty, or the
        answer contains no ``[N]`` markers.
    """
    if not answer or not chunks:
        return []

    sentences = [s.strip() for s in _SENTENCE_RE.split(answer) if s.strip()]
    citations: list[dict] = []

    for sentence in sentences:
        markers = _MARKER_RE.findall(sentence)
        for marker_str in markers:
            idx = int(marker_str)  # 1-based
            if idx < 1 or idx > len(chunks):
                logger.warning(
                    "extract_citations: marker [%d] is out of range (chunks=%d) — skipping",
                    idx,
                    len(chunks),
                )
                continue

            chunk = chunks[idx - 1]
            meta = chunk.get("metadata", {})
            citations.append(
                {
                    "claim": sentence,
                    "source_chunk": chunk.get("chunk_text", ""),
                    "page_number": meta.get("page_number"),
                    "source_document": meta.get("source_document", ""),
                }
            )

    logger.debug(
        "extract_citations: %d sentences -> %d citations",
        len(sentences),
        len(citations),
    )
    return citations


# ---------------------------------------------------------------------------
# Phase 2 — Multimodal citation extraction
# ---------------------------------------------------------------------------

# [Source: some_file.pdf, Page 5]
_SOURCE_REF_RE = re.compile(
    r"\[Source:\s*(?P<doc>[^,\]]+),\s*Page\s*(?P<page>\d+)\]", re.IGNORECASE
)
# [Image from: some_file.pdf, Page 12]
_IMAGE_REF_RE = re.compile(
    r"\[Image from:\s*(?P<doc>[^,\]]+),\s*Page\s*(?P<page>\d+)\]", re.IGNORECASE
)


def extract_multimodal_citations(
    answer: str,
    text_chunks: list[dict],
    images: list[dict],
) -> list[dict]:
    """Map multimodal answer text back to source text chunks and images.

    Scans *answer* for ``[Source: X, Page Y]`` and ``[Image from: X, Page Y]``
    reference markers produced by :func:`src.generation.prompt_builder.build_multimodal_prompt`.
    Each reference is matched to the corresponding chunk or image by document
    name and page number.

    Args:
        answer: LLM-generated answer string.
        text_chunks: Flat text chunk dicts (keys: ``text``/``chunk_text``,
            ``source_document``, ``page_number``).
        images: Flat image dicts (keys: ``caption``, ``source_pdf`` or
            ``source_document``, ``page_number``, ``image_path``).

    Returns:
        List of citation dicts, each with:

        - ``"claim"`` — sentence containing the reference.
        - ``"source_type"`` — ``"text"`` or ``"image"``.
        - ``"source_document"`` — document filename.
        - ``"page_number"`` — integer page number.
        - ``"chunk_text"`` — chunk body text (``None`` for image citations).
        - ``"image_path"`` — path to image file (``None`` for text citations).
        - ``"image_caption"`` — caption string (``None`` for text citations).

        Returns ``[]`` when *answer* contains no recognised reference markers.
    """
    if not answer:
        return []

    sentences = [s.strip() for s in _SENTENCE_RE.split(answer) if s.strip()]
    # Also treat the whole answer as one block in case it isn't sentence-split
    if not sentences:
        sentences = [answer.strip()]

    citations: list[dict] = []

    for sentence in sentences:
        # Text source references
        for m in _SOURCE_REF_RE.finditer(sentence):
            doc = m.group("doc").strip()
            page = int(m.group("page"))
            matched = _match_text_chunk(doc, page, text_chunks)
            citations.append(
                {
                    "claim": sentence,
                    "source_type": "text",
                    "source_document": doc,
                    "page_number": page,
                    "chunk_text": matched.get("text") or matched.get("chunk_text")
                    if matched
                    else None,
                    "image_path": None,
                    "image_caption": None,
                }
            )

        # Image source references
        for m in _IMAGE_REF_RE.finditer(sentence):
            doc = m.group("doc").strip()
            page = int(m.group("page"))
            matched = _match_image(doc, page, images)
            citations.append(
                {
                    "claim": sentence,
                    "source_type": "image",
                    "source_document": doc,
                    "page_number": page,
                    "chunk_text": None,
                    "image_path": matched.get("image_path") if matched else None,
                    "image_caption": matched.get("caption") if matched else None,
                }
            )

    logger.debug(
        "extract_multimodal_citations: %d sentences -> %d citations",
        len(sentences),
        len(citations),
    )
    return citations


def _match_text_chunk(doc: str, page: int, chunks: list[dict]) -> dict | None:
    """Return the first chunk whose source_document and page_number match."""
    for chunk in chunks:
        chunk_doc = chunk.get("source_document", "")
        chunk_page = chunk.get("page_number")
        if doc in chunk_doc or chunk_doc in doc:
            if chunk_page == page:
                return chunk
    return None


def _match_image(doc: str, page: int, images: list[dict]) -> dict | None:
    """Return the first image whose source (source_pdf/source_document) and page match."""
    for img in images:
        img_doc = img.get("source_pdf", img.get("source_document", ""))
        img_page = img.get("page_number")
        if doc in img_doc or img_doc in doc:
            if img_page == page:
                return img
    return None
