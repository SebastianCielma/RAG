"""Qdrant vector database storage for RAG application."""

import logging
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rag.core.config import get_settings
from rag.core.exceptions import VectorDBError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """Get or create singleton Qdrant client with connection pooling."""
    settings = get_settings()
    try:
        logger.info("Creating Qdrant client connection to %s", settings.qdrant_url)
        client = QdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout,
        )
        logger.info("Qdrant client connected successfully")
        return client
    except Exception as exc:
        logger.exception("Failed to connect to Qdrant")
        msg = f"Failed to connect to Qdrant: {settings.qdrant_url}"
        raise VectorDBError(msg) from exc


class QdrantStorage:
    """Handles vector storage and retrieval using Qdrant."""

    def __init__(
        self,
        collection: str | None = None,
        dim: int | None = None,
    ) -> None:
        """Initialize Qdrant storage with collection creation if needed."""
        settings = get_settings()
        self.collection = collection or settings.qdrant_collection
        self._dim = dim or settings.embed_dim
        self.client = get_qdrant_client()
        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            if not self.client.collection_exists(self.collection):
                logger.info(
                    "Creating collection '%s' with dimension %d",
                    self.collection,
                    self._dim,
                )
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(
                        size=self._dim, distance=Distance.COSINE
                    ),
                )
        except Exception as exc:
            logger.exception("Failed to ensure collection exists")
            msg = f"Failed to create collection '{self.collection}'"
            raise VectorDBError(msg) from exc

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> int:
        """Upsert vectors with their payloads into the collection."""
        if not (len(ids) == len(vectors) == len(payloads)):
            raise ValueError(
                f"Mismatched lengths: ids={len(ids)}, vectors={len(vectors)}, "
                f"payloads={len(payloads)}"
            )

        if not ids:
            logger.warning("Empty upsert request, skipping")
            return 0

        points = [
            PointStruct(id=id_, vector=vec, payload=pay)
            for id_, vec, pay in zip(ids, vectors, payloads, strict=True)
        ]

        try:
            self.client.upsert(self.collection, points=points)
            logger.info(
                "Upserted %d points to collection '%s'", len(points), self.collection
            )
            return len(points)
        except Exception as exc:
            logger.exception("Failed to upsert %d points", len(points))
            raise VectorDBError(f"Failed to upsert {len(points)} points") from exc

    def list_sources(self) -> list[str]:
        """Get list of unique document sources in the collection."""
        sources: set[str] = set()
        offset: str | int | None = None

        try:
            while True:
                results, offset = self.client.scroll(
                    collection_name=self.collection,
                    limit=100,
                    offset=offset,
                    with_payload=["source"],
                    with_vectors=False,
                )

                for point in results:
                    payload = getattr(point, "payload", None) or {}
                    if source := payload.get("source", ""):
                        sources.add(source)

                if offset is None:
                    break

            logger.debug("Found %d unique sources", len(sources))
            return sorted(sources)
        except Exception as exc:
            logger.exception("Failed to list sources")
            raise VectorDBError("Failed to list document sources") from exc

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> dict[str, list[str]]:
        """Search for similar vectors with optional source filtering."""
        query_filter: Filter | None = None
        if source_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_filter),
                    )
                ]
            )

        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                query_filter=query_filter,
                with_payload=True,
                limit=top_k,
            )
        except Exception as exc:
            logger.exception("Search failed")
            raise VectorDBError("Vector search failed") from exc

        contexts: list[str] = []
        sources: list[str] = []

        for result in results:
            payload = getattr(result, "payload", None) or {}
            if text := payload.get("text", ""):
                contexts.append(text)
                sources.append(payload.get("source", ""))

        logger.debug(
            "Search returned %d results (filter=%s)",
            len(contexts),
            source_filter or "none",
        )
        return {"contexts": contexts, "sources": sources}

    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection)
            logger.info("Deleted collection '%s'", self.collection)
            return True
        except Exception as exc:
            logger.exception("Failed to delete collection '%s'", self.collection)
            msg = f"Failed to delete collection '{self.collection}'"
            raise VectorDBError(msg) from exc

    def count(self) -> int:
        """Get the number of points in the collection."""
        info = self.client.get_collection(self.collection)
        return info.points_count or 0

    def delete_by_source(self, source_id: str) -> int:
        """Delete all points with a specific source ID."""
        try:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=source_id),
                    )
                ]
            )

            self.client.delete(
                collection_name=self.collection,
                points_selector=filter_condition,
            )

            logger.info("Deleted points for source '%s'", source_id)
            return 1

        except Exception as exc:
            logger.exception("Failed to delete source '%s'", source_id)
            msg = f"Failed to delete source '{source_id}'"
            raise VectorDBError(msg) from exc


@lru_cache(maxsize=1)
def get_storage() -> QdrantStorage:
    """Get singleton QdrantStorage instance."""
    return QdrantStorage()
