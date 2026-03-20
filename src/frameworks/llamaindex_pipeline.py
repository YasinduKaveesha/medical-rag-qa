"""LlamaIndex pipeline for the Medical RAG Q&A system.

Provides a drop-in alternative to the custom retrieval pipeline using
LlamaIndex's ``RetrieverQueryEngine``.  Because the optional
``llama-index-vector-stores-qdrant`` package is not a project dependency, the
existing :class:`src.retrieval.vector_store.QdrantStore` is bridged into
LlamaIndex via a custom :class:`_QdrantBridgeRetriever`.

Typical usage
-------------
::

    from src.frameworks.llamaindex_pipeline import LlamaIndexPipeline, get_llamaindex_pipeline

    pipeline = get_llamaindex_pipeline()
    result = pipeline.query("What is the dose of amitriptyline?")
    print(result["answer"])
    print(result["sources"])
    print(result["latency_ms"])
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Qdrant bridge retriever (LlamaIndex BaseRetriever subclass)
# ---------------------------------------------------------------------------


class _QdrantBridgeRetriever:
    """Bridge between :class:`src.retrieval.vector_store.QdrantStore` and LlamaIndex.

    Subclasses :class:`llama_index.core.retrievers.BaseRetriever` so that it can
    be used inside :class:`llama_index.core.query_engine.RetrieverQueryEngine`.
    Uses :meth:`src.embeddings.encoder.EmbeddingEncoder.encode` for query
    vectorisation, then delegates retrieval to ``QdrantStore.search``.

    Args:
        store: :class:`src.retrieval.vector_store.QdrantStore` instance.
        encoder: :class:`src.embeddings.encoder.EmbeddingEncoder` instance.
        top_k: Number of results to retrieve.
    """

    def __new__(cls, store: Any, encoder: Any, top_k: int = 5) -> "_QdrantBridgeRetriever":
        from llama_index.core.retrievers import BaseRetriever  # noqa: PLC0415

        # Dynamically subclass BaseRetriever so we do not need to subclass at
        # class-definition time (avoids import errors when llama_index is absent).
        bases = (BaseRetriever,)
        DynamicClass = type("_QdrantBridgeRetriever", bases, dict(cls.__dict__))

        instance = object.__new__(DynamicClass)
        object.__setattr__(instance, "_store", store)
        object.__setattr__(instance, "_encoder", encoder)
        object.__setattr__(instance, "_top_k", top_k)
        return instance  # type: ignore[return-value]

    def _retrieve(self, query_bundle: Any) -> list[Any]:  # noqa: ANN401
        """Retrieve chunks for *query_bundle* and return LlamaIndex NodeWithScore list.

        Args:
            query_bundle: LlamaIndex ``QueryBundle`` with a ``query_str`` field.

        Returns:
            List of :class:`llama_index.core.schema.NodeWithScore` objects.
        """
        from llama_index.core.schema import NodeWithScore, TextNode  # noqa: PLC0415

        store = object.__getattribute__(self, "_store")
        encoder = object.__getattribute__(self, "_encoder")
        top_k = object.__getattribute__(self, "_top_k")

        query_vector = encoder.encode(query_bundle.query_str)
        chunks = store.search(query_vector, top_k=top_k)

        nodes = []
        for chunk in chunks:
            node = TextNode(
                text=chunk["chunk_text"],
                metadata=chunk.get("metadata", {}),
            )
            nodes.append(NodeWithScore(node=node, score=chunk.get("score", 0.0)))
        return nodes


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class LlamaIndexPipeline:
    """RAG pipeline implemented with LlamaIndex ``RetrieverQueryEngine``.

    The query engine is injected at construction time to allow test doubles.

    Args:
        _query_engine: Pre-built LlamaIndex ``RetrieverQueryEngine``.  When
            ``None`` (the default), one is built from the current
            :class:`src.config.Settings`.
    """

    def __init__(self, _query_engine: Any | None = None) -> None:
        if _query_engine is None:
            _query_engine = _build_query_engine()
        self._query_engine = _query_engine
        logger.debug("LlamaIndexPipeline: initialised")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict:
        """Run the LlamaIndex query engine and return answer + sources.

        Args:
            question: The user's clinical question.

        Returns:
            Dict with keys:

            - ``"answer"`` (``str``): LLM-generated answer.
            - ``"sources"`` (``list[dict]``): One entry per source node with
              keys ``"source_document"``, ``"page_number"``, ``"chunk_text"``.
            - ``"latency_ms"`` (``float``): Wall-clock latency in
              milliseconds.
        """
        t0 = time.perf_counter()

        response = self._query_engine.query(question)
        answer: str = str(response)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        sources = []
        for node_with_score in getattr(response, "source_nodes", []):
            node = node_with_score.node
            metadata = getattr(node, "metadata", {})
            sources.append(
                {
                    "source_document": metadata.get("source_document", ""),
                    "page_number": metadata.get("page_number"),
                    "chunk_text": getattr(node, "text", ""),
                }
            )

        logger.info(
            "LlamaIndexPipeline.query: sources=%d  latency_ms=%.1f",
            len(sources),
            latency_ms,
        )
        return {"answer": answer, "sources": sources, "latency_ms": latency_ms}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_query_engine() -> Any:
    """Build a ``RetrieverQueryEngine`` from current settings."""
    from llama_index.core import Settings as LISettings  # noqa: PLC0415
    from llama_index.core.query_engine import RetrieverQueryEngine  # noqa: PLC0415
    from llama_index.llms.openai import OpenAI as LlamaOpenAI  # noqa: PLC0415

    from src.config import get_settings  # noqa: PLC0415
    from src.embeddings.encoder import get_encoder  # noqa: PLC0415
    from src.retrieval.vector_store import QdrantStore  # noqa: PLC0415

    s = get_settings()

    llm = LlamaOpenAI(
        model=s.llm_model,
        api_key=s.groq_api_key,
        api_base=s.llm_base_url,
        temperature=0,
    )
    LISettings.llm = llm

    store = QdrantStore(
        host=s.qdrant_host,
        port=s.qdrant_port,
        collection_name=s.collection_name,
    )
    retriever = _QdrantBridgeRetriever(
        store=store,
        encoder=get_encoder(),
        top_k=s.top_k_retrieval,
    )

    return RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_pipeline: LlamaIndexPipeline | None = None


def get_llamaindex_pipeline() -> LlamaIndexPipeline:
    """Return a process-level singleton ``LlamaIndexPipeline``.

    The pipeline is initialised once on first call using the current
    :class:`src.config.Settings` and reused on subsequent calls.

    Returns:
        Singleton :class:`LlamaIndexPipeline` instance.
    """
    global _pipeline
    if _pipeline is None:
        _pipeline = LlamaIndexPipeline()
        logger.info("get_llamaindex_pipeline: pipeline created")
    return _pipeline
