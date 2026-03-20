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
