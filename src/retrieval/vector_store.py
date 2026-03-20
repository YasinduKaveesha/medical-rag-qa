"""Qdrant vector store wrapper for the Medical RAG Q&A pipeline.

Provides a singleton :class:`QdrantStore` that wraps ``QdrantClient`` and
exposes collection management, chunk upsert, and similarity search with
optional metadata filtering.

Typical usage
-------------
::

    from src.retrieval.vector_store import get_store

    store = get_store()
    store.create_collection()
    n = store.upsert_chunks(chunks, embeddings)
    results = store.search(query_vector, top_k=20)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)


class QdrantStore:
    """Wrapper around :class:`qdrant_client.QdrantClient` for vector storage.

    All five public methods wrap their Qdrant calls and re-raise connectivity
    failures as :class:`ConnectionError` with a human-readable message so the
    caller does not need to handle Qdrant-specific exceptions.

    Args:
        host: Qdrant server hostname.  Defaults to ``"localhost"``.
        port: Qdrant server port.  Defaults to ``6333``.
        collection_name: Name of the Qdrant collection to operate on.
        _client: Optional pre-constructed client object.  When provided it is
            used directly, bypassing ``QdrantClient(host, port)`` construction.
            Intended for testing only — pass a :class:`unittest.mock.MagicMock`
            to avoid requiring a running Qdrant instance.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "medical_docs",
        _client: Any | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name

        if _client is not None:
            self._client = _client
            logger.debug("QdrantStore using injected client (test mode)")
        else:
            from qdrant_client import QdrantClient  # noqa: PLC0415

            logger.info("Connecting to Qdrant at %s:%d", host, port)
            self._client = QdrantClient(host=host, port=port)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection(self, vector_size: int = 384) -> None:
        """Create the Qdrant collection with cosine distance and payload indexes.

        Creates the collection if it does not already exist, then adds keyword
        payload indexes on ``document_type`` and ``section_title`` for fast
        filtered retrieval.  This method is idempotent — if the collection
        already exists the call is a no-op.

        Args:
            vector_size: Dimensionality of the embedding vectors.  Defaults to
                ``384`` (``all-MiniLM-L6-v2``).

        Raises:
            ConnectionError: If Qdrant is unreachable.
        """
        from qdrant_client.models import (  # noqa: PLC0415
            Distance,
            PayloadSchemaType,
            VectorParams,
        )

        try:
            existing = [c.name for c in self._client.get_collections().collections]
            if self._collection_name not in existing:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info(
                    "Created collection '%s' (dim=%d, distance=COSINE)",
                    self._collection_name,
                    vector_size,
                )
            else:
                logger.debug(
                    "Collection '%s' already exists — skipping create", self._collection_name
                )

            for field in ("document_type", "section_title"):
                self._client.create_payload_index(
                    collection_name=self._collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                logger.debug("Payload index ensured: %s", field)

        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Qdrant at {self._host}:{self._port}. "
                "Is the Docker container running? "
                f"Original error: {exc}"
            ) from exc

    def delete_collection(self) -> None:
        """Delete the collection and all its data.

        Raises:
            ConnectionError: If Qdrant is unreachable.
        """
        try:
            self._client.delete_collection(collection_name=self._collection_name)
            logger.info("Deleted collection '%s'", self._collection_name)
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Qdrant at {self._host}:{self._port}. "
                f"Original error: {exc}"
            ) from exc

    def get_collection_info(self) -> dict:
        """Return a plain dict with basic collection statistics.

        Returns:
            Dict with keys ``"vectors_count"``, ``"points_count"``, and
            ``"status"`` (``"green"`` / ``"yellow"`` / ``"red"``).

        Raises:
            ConnectionError: If Qdrant is unreachable.
        """
        try:
            info = self._client.get_collection(collection_name=self._collection_name)
            return {
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Qdrant at {self._host}:{self._port}. "
                f"Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: list[dict], embeddings: list[np.ndarray]) -> int:
        """Upsert chunk text + metadata as Qdrant points.

        Each point is keyed by the chunk's ``chunk_id`` UUID hex string.  The
        full metadata dict plus the chunk ``text`` are stored in the payload so
        a search hit carries everything needed for generation — no secondary
        lookup required.

        Args:
            chunks: List of chunk dicts, each with keys ``"text"`` and
                ``"metadata"``.  ``metadata`` must contain a ``"chunk_id"``
                key (UUID hex string).
            embeddings: List of 1-D ``np.float32`` arrays of shape
                ``(vector_size,)``, one per chunk.  Must be the same length as
                *chunks*.

        Returns:
            Number of points upserted.  Returns ``0`` immediately for empty
            input without calling the Qdrant client.

        Raises:
            ConnectionError: If Qdrant is unreachable.
        """
        if not chunks:
            return 0

        from qdrant_client.models import PointStruct  # noqa: PLC0415

        try:
            points = [
                PointStruct(
                    id=chunk["metadata"]["chunk_id"],
                    vector=emb.tolist(),
                    payload={"text": chunk["text"], **chunk["metadata"]},
                )
                for chunk, emb in zip(chunks, embeddings)
            ]
            self._client.upsert(collection_name=self._collection_name, points=points)
            logger.info("Upserted %d points into '%s'", len(points), self._collection_name)
            return len(points)
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Qdrant at {self._host}:{self._port}. "
                f"Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search for the most similar chunks to *query_vector*.

        Args:
            query_vector: 1-D ``np.float32`` array of shape ``(vector_size,)``.
            top_k: Maximum number of results to return.  Defaults to ``20``.
            filters: Optional dict for metadata pre-filtering.  Supported keys:
                ``"document_type"`` and ``"section_title"``.  Unknown keys are
                silently ignored.  Pass ``None`` to search without filtering.

        Returns:
            List of dicts, each with keys:

            - ``"chunk"`` — the full payload dict (``text`` + all metadata keys).
            - ``"score"`` — cosine similarity score (``float`` in ``[0, 1]``).

            Results are ordered by score descending (Qdrant default).

        Raises:
            ConnectionError: If Qdrant is unreachable.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415

        qdrant_filter: Filter | None = None
        if filters:
            must = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
                if k in {"document_type", "section_title"}
            ]
            if must:
                qdrant_filter = Filter(must=must)

        try:
            hits = self._client.search(
                collection_name=self._collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
            return [{"chunk": hit.payload, "score": hit.score} for hit in hits]
        except Exception as exc:
            raise ConnectionError(
                f"Cannot connect to Qdrant at {self._host}:{self._port}. "
                f"Original error: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_store: QdrantStore | None = None


def get_store() -> QdrantStore:
    """Return the process-level singleton :class:`QdrantStore`.

    On the first call, reads host / port / collection name from
    :func:`src.config.get_settings` and constructs the client.  Subsequent
    calls return the cached instance.

    Returns:
        The singleton :class:`QdrantStore` instance.
    """
    global _store
    if _store is None:
        s = get_settings()
        logger.info(
            "Initialising singleton QdrantStore: %s:%d / %s",
            s.qdrant_host,
            s.qdrant_port,
            s.collection_name,
        )
        _store = QdrantStore(
            host=s.qdrant_host,
            port=s.qdrant_port,
            collection_name=s.collection_name,
        )
    return _store
