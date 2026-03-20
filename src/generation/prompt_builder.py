"""Prompt assembly for the Medical RAG Q&A generation step.

Builds a single string containing a strict system instruction, numbered
context chunks, and the user question.  The numbered ``[N]`` markers allow
the LLM to cite specific chunks by index, which :mod:`src.generation.citations`
maps back to source metadata.

Typical usage
-------------
::

    from src.generation.prompt_builder import build_prompt

    prompt = build_prompt(query, chunks)
    # Pass prompt to LLMClient.generate()
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SYSTEM_INSTRUCTION = """\
You are a medical information assistant. Answer the user's question using \
ONLY the information provided in the context below. For each claim in your \
answer, cite the source document and page number in the format \
[source_document, p.PAGE_NUMBER]. If the answer cannot be determined from \
the provided context, respond with exactly:
"I cannot answer from the provided documents."
Do not use any prior knowledge outside the provided context."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Assemble a system prompt, numbered context blocks, and the user query.

    Each context block is numbered ``[1]``, ``[2]``, ... in the order the
    chunks are provided.  The LLM is instructed to cite chunks using these
    indices.  :func:`src.generation.citations.extract_citations` maps the
    indices back to the original source metadata.

    Args:
        query: Raw natural-language question string.
        chunks: List of retrieval result dicts as returned by
            :meth:`src.retrieval.pipeline.RetrievalPipeline.retrieve`.
            Each dict must have keys ``"chunk_text"`` and ``"metadata"``
            (with sub-keys ``"source_document"``, ``"page_number"``, and
            ``"section_title"``).

    Returns:
        A formatted prompt string ready to be passed to
        :meth:`src.generation.llm_client.LLMClient.generate`.
        Returns ``""`` immediately when *chunks* is empty — the caller
        (typically :func:`src.generation.refusal.should_refuse`) is
        responsible for deciding whether to proceed without context.
    """
    if not chunks:
        logger.debug("build_prompt called with empty chunks — returning empty string")
        return ""

    lines: list[str] = []

    lines.append("SYSTEM:")
    lines.append(_SYSTEM_INSTRUCTION)
    lines.append("")
    lines.append("CONTEXT:")

    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source_document", "unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section_title", "")
        text = chunk.get("chunk_text", "")

        header = f"[{i}] Source: {source} | Page: {page}"
        if section:
            header += f" | Section: {section}"
        lines.append(header)
        lines.append(text)
        lines.append("")

    lines.append("QUESTION:")
    lines.append(query)

    prompt = "\n".join(lines)
    logger.debug(
        "build_prompt: assembled prompt with %d chunks (%d chars)", len(chunks), len(prompt)
    )
    return prompt
