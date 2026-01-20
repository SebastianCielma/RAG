"""Embedding service for generating text embeddings."""

import hashlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from sentence_transformers import SentenceTransformer

from rag.core.config import get_settings
from rag.core.exceptions import EmbeddingError

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Embedding cache for repeated texts (max 1000 entries)
_embedding_cache: dict[str, list[float]] = {}
_EMBEDDING_CACHE_MAX_SIZE = 1000


def _get_text_hash(text: str) -> str:
    """Generate a short hash for cache key."""
    return hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Get or create the embedding model singleton."""
    settings = get_settings()
    try:
        logger.info("Loading embedding model: %s", settings.embed_model)
        model = SentenceTransformer(settings.embed_model)
        logger.info("Embedding model loaded successfully")
        return model
    except Exception as exc:
        logger.exception("Failed to load embedding model")
        raise EmbeddingError(f"Failed to load model '{settings.embed_model}'") from exc


def warmup() -> None:
    """Pre-load the embedding model. Call at application startup."""
    logger.info("Warming up embedding model...")
    model = get_embedding_model()
    _ = model.encode(["warmup"], convert_to_numpy=True)
    logger.info("Warmup complete")


def embed_texts(texts: list[str], use_cache: bool = True) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    if not texts:
        raise ValueError("Cannot embed empty list of texts")

    if use_cache:
        cached_results: list[list[float] | None] = []
        texts_to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            text_hash = _get_text_hash(text)
            if text_hash in _embedding_cache:
                cached_results.append(_embedding_cache[text_hash])
            else:
                cached_results.append(None)
                texts_to_embed.append((i, text))

        if not texts_to_embed:
            logger.debug("All %d embeddings served from cache", len(texts))
            return [emb for emb in cached_results if emb is not None]

        try:
            model = get_embedding_model()
            uncached_texts = [t for _, t in texts_to_embed]
            new_embeddings: NDArray[np.float32] = model.encode(
                uncached_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            for (orig_idx, text), embedding in zip(
                texts_to_embed, new_embeddings.tolist(), strict=True
            ):
                text_hash = _get_text_hash(text)
                if len(_embedding_cache) >= _EMBEDDING_CACHE_MAX_SIZE:
                    oldest_key = next(iter(_embedding_cache))
                    del _embedding_cache[oldest_key]
                _embedding_cache[text_hash] = embedding
                cached_results[orig_idx] = embedding

            logger.debug(
                "Embedded %d texts (%d from cache)",
                len(texts),
                len(texts) - len(texts_to_embed),
            )
            return [emb for emb in cached_results if emb is not None]

        except Exception as exc:
            logger.exception("Failed to generate embeddings")
            raise EmbeddingError(f"Failed to embed {len(texts)} texts") from exc

    try:
        model = get_embedding_model()
        embeddings: NDArray[np.float32] = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    except Exception as exc:
        logger.exception("Failed to generate embeddings")
        raise EmbeddingError(f"Failed to embed {len(texts)} texts") from exc


def clear_embedding_cache() -> int:
    """Clear the embedding cache."""
    global _embedding_cache
    count = len(_embedding_cache)
    _embedding_cache = {}
    logger.info("Cleared %d entries from embedding cache", count)
    return count
