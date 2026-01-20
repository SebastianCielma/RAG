"""Tests for schemas module."""

import pytest
from pydantic import ValidationError

from rag.models.schemas import (
    DocumentPayload,
    LLMModel,
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
)


class TestLLMModel:
    """Tests for LLMModel enum."""

    def test_default_returns_llama_70b(self) -> None:
        """Default model should be Llama 3.3 70B."""
        assert LLMModel.default() == LLMModel.LLAMA_3_3_70B

    def test_enum_values_are_valid_model_ids(self) -> None:
        """All enum values should be valid Groq model identifiers."""
        for model in LLMModel:
            assert isinstance(model.value, str)
            assert len(model.value) > 0

    def test_can_create_from_string_value(self) -> None:
        """Should be able to create enum from string value."""
        model = LLMModel("llama-3.3-70b-versatile")
        assert model == LLMModel.LLAMA_3_3_70B


class TestDocumentPayload:
    """Tests for DocumentPayload model."""

    def test_create_valid_payload(self) -> None:
        """Should create payload with source and text."""
        payload = DocumentPayload(source="test.pdf", text="Sample text")

        assert payload.source == "test.pdf"
        assert payload.text == "Sample text"

    def test_is_immutable(self) -> None:
        """Should be immutable (frozen)."""
        payload = DocumentPayload(source="test.pdf", text="Sample text")

        with pytest.raises(ValidationError):
            payload.source = "new.pdf"  # type: ignore[misc]


class TestRAGChunkAndSrc:
    """Tests for RAGChunkAndSrc model."""

    def test_create_with_chunks(self) -> None:
        """Should create with chunks and source_id."""
        result = RAGChunkAndSrc(
            chunks=["chunk1", "chunk2"],
            source_id="test.pdf",
        )

        assert len(result.chunks) == 2
        assert result.source_id == "test.pdf"

    def test_filters_empty_chunks(self) -> None:
        """Should filter out empty chunks via validator."""
        result = RAGChunkAndSrc(
            chunks=["valid", "", "  ", "another valid"],
            source_id="test.pdf",
        )

        assert result.chunks == ["valid", "another valid"]


class TestRAGUpsertResult:
    """Tests for RAGUpsertResult model."""

    def test_create_with_count(self) -> None:
        """Should create with ingested count."""
        result = RAGUpsertResult(ingested=10)
        assert result.ingested == 10

    def test_rejects_negative_count(self) -> None:
        """Should reject negative ingested count."""
        with pytest.raises(ValidationError):
            RAGUpsertResult(ingested=-1)


class TestRAGSearchResult:
    """Tests for RAGSearchResult model."""

    def test_is_empty_when_no_contexts(self) -> None:
        """is_empty should be True when no contexts."""
        result = RAGSearchResult(contexts=[], sources=[])
        assert result.is_empty is True

    def test_is_not_empty_when_has_contexts(self) -> None:
        """is_empty should be False when has contexts."""
        result = RAGSearchResult(
            contexts=["Some context"],
            sources=["test.pdf"],
        )
        assert result.is_empty is False


class TestRAGQueryResult:
    """Tests for RAGQueryResult model."""

    def test_unique_sources_deduplicates(self) -> None:
        """unique_sources should return deduplicated list."""
        result = RAGQueryResult(
            answer="Test answer",
            sources=["a.pdf", "b.pdf", "a.pdf", "c.pdf", "b.pdf"],
            num_contexts=5,
        )

        assert result.unique_sources == ["a.pdf", "b.pdf", "c.pdf"]

    def test_preserves_source_order(self) -> None:
        """unique_sources should preserve first occurrence order."""
        result = RAGQueryResult(
            answer="Test",
            sources=["c.pdf", "a.pdf", "b.pdf", "a.pdf"],
            num_contexts=4,
        )

        assert result.unique_sources == ["c.pdf", "a.pdf", "b.pdf"]
