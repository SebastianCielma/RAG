"""Tests for embeddings service."""


import pytest

from rag.services.embeddings import embed_texts, get_embedding_model


class TestGetEmbeddingModel:
    """Tests for get_embedding_model function."""

    def test_returns_sentence_transformer(self) -> None:
        """Should return a SentenceTransformer instance."""
        model = get_embedding_model()
        assert model is not None
        assert hasattr(model, "encode")

    def test_returns_same_instance_on_multiple_calls(self) -> None:
        """Should return cached instance (singleton pattern)."""
        model1 = get_embedding_model()
        model2 = get_embedding_model()
        assert model1 is model2


class TestEmbedTexts:
    """Tests for embed_texts function."""

    def test_returns_list_of_embeddings(self) -> None:
        """Should return embeddings for each input text."""
        texts = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI.",
        ]
        embeddings = embed_texts(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)

    def test_embedding_dimension(self) -> None:
        """Should return embeddings with correct dimension (384)."""
        embeddings = embed_texts(["Test text"])

        for emb in embeddings:
            assert len(emb) == 384

    def test_raises_value_error_on_empty_list(self) -> None:
        """Should raise ValueError for empty input list."""
        with pytest.raises(ValueError, match="empty list"):
            embed_texts([])
