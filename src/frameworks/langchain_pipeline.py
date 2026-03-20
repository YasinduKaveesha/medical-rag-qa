"""LangChain LCEL pipeline for the Medical RAG Q&A system.

Provides a drop-in alternative to the custom retrieval pipeline, using
LangChain's LCEL chain with a Groq-backed ChatOpenAI LLM and a Qdrant
vector-store retriever.

Typical usage
-------------
::

    from src.frameworks.langchain_pipeline import LangChainPipeline, get_langchain_pipeline

    pipeline = get_langchain_pipeline()
    result = pipeline.query("What is the dose of amitriptyline?")
    print(result["answer"])
    print(result["sources"])
    print(result["latency_ms"])
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a clinical pharmacology assistant.  Answer the question using ONLY the
context provided below.  Cite the source document and page number inline using
square brackets, e.g. [WHO-EML-2023.pdf, p.3].  If the context is insufficient,
say "I cannot answer from the provided documents."

Context:
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class LangChainPipeline:
    """RAG pipeline implemented with LangChain LCEL.

    The vectorstore retriever and LLM are injected at construction time so
    they can be replaced with test doubles in unit tests.

    Args:
        vectorstore: LangChain ``VectorStore`` instance with an
            ``as_retriever()`` method.  Defaults to a Qdrant vectorstore
            pointing at the configured collection when ``None``.
        llm: A LangChain ``Runnable`` LLM (e.g. ``ChatOpenAI``).  Defaults
            to a Groq-backed ``ChatOpenAI`` instance when ``None``.
    """

    def __init__(
        self,
        vectorstore: Any | None = None,
        llm: Runnable | None = None,
    ) -> None:
        from src.config import get_settings  # noqa: PLC0415

        s = get_settings()

        if vectorstore is None:
            vectorstore = _build_vectorstore(s)
        if llm is None:
            llm = _build_llm(s)

        self._vectorstore = vectorstore
        self._llm = llm
        self._retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": s.top_k_retrieval},
        )
        self._prompt = _PROMPT
        self._parser = StrOutputParser()

        logger.debug("LangChainPipeline: initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """Run the RAG chain and return answer + sources.

        Retrieves relevant documents, formats a prompt, invokes the LLM,
        and returns a standardised result dict.

        Args:
            question: The user's clinical question.

        Returns:
            Dict with keys:

            - ``"answer"`` (``str``): LLM-generated answer.
            - ``"sources"`` (``list[dict]``): One entry per retrieved
              document with keys ``"source_document"``, ``"page_number"``,
              ``"chunk_text"``.
            - ``"latency_ms"`` (``float``): Wall-clock latency in
              milliseconds.
        """
        t0 = time.perf_counter()

        docs: list[Document] = self._retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        chain = self._prompt | self._llm | self._parser
        answer: str = chain.invoke({"context": context, "question": question})

        latency_ms = (time.perf_counter() - t0) * 1000.0

        sources = [
            {
                "source_document": d.metadata.get("source_document", ""),
                "page_number": d.metadata.get("page_number"),
                "chunk_text": d.page_content,
            }
            for d in docs
        ]

        logger.info(
            "LangChainPipeline.query: docs=%d  latency_ms=%.1f",
            len(docs),
            latency_ms,
        )
        return {"answer": answer, "sources": sources, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_vectorstore(s: Any) -> Any:
    """Build a LangChain Qdrant vectorstore from settings."""
    from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: PLC0415
    from langchain_community.vectorstores import Qdrant as LCQdrant  # noqa: PLC0415
    from qdrant_client import QdrantClient  # noqa: PLC0415

    lc_embeddings = HuggingFaceEmbeddings(model_name=s.embedding_model)
    client = QdrantClient(host=s.qdrant_host, port=s.qdrant_port)
    return LCQdrant(
        client=client,
        collection_name=s.collection_name,
        embeddings=lc_embeddings,
    )


def _build_llm(s: Any) -> Any:
    """Build a Groq-backed ChatOpenAI LLM from settings."""
    from langchain_openai import ChatOpenAI  # noqa: PLC0415

    return ChatOpenAI(
        model=s.llm_model,
        openai_api_key=s.groq_api_key,
        openai_api_base=s.llm_base_url,
        temperature=0,
    )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_pipeline: LangChainPipeline | None = None


def get_langchain_pipeline() -> LangChainPipeline:
    """Return a process-level singleton ``LangChainPipeline``.

    The pipeline is initialised once on first call using the current
    :class:`src.config.Settings` and reused on subsequent calls.

    Returns:
        Singleton :class:`LangChainPipeline` instance.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = LangChainPipeline()
        logger.info("get_langchain_pipeline: pipeline created")
    return _pipeline
