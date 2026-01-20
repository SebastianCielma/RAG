"""Pydantic schemas and data models for RAG application."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "LLMModel",
    "DocumentPayload",
    "RAGChunkAndSrc",
    "RAGUpsertResult",
    "RAGSearchResult",
    "RAGQueryResult",
]


class LLMModel(StrEnum):
    """Supported LLM models available via Groq API."""

    LLAMA_3_3_70B = "llama-3.3-70b-versatile"
    LLAMA_3_1_8B = "llama-3.1-8b-instant"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    DEEPSEEK_R1_70B = "deepseek-r1-distill-llama-70b"
    QWEN_QWQ_32B = "qwen-qwq-32b"

    @classmethod
    def default(cls) -> Self:
        """Get the default model."""
        return cls.LLAMA_3_3_70B


class DocumentPayload(BaseModel):
    """Payload structure for documents stored in vector database."""

    model_config = ConfigDict(frozen=True)

    source: str
    text: str


class RAGChunkAndSrc(BaseModel):
    """Chunks extracted from a PDF with source identifier."""

    model_config = ConfigDict(frozen=True)

    chunks: list[str] = Field(default_factory=list)
    source_id: str | None = None

    @field_validator("chunks")
    @classmethod
    def validate_chunks(cls, v: list[str]) -> list[str]:
        """Ensure chunks list contains only non-empty strings."""
        return [chunk for chunk in v if chunk.strip()]


class RAGUpsertResult(BaseModel):
    """Result of upserting chunks to vector database."""

    model_config = ConfigDict(frozen=True)

    ingested: int = Field(ge=0)


class RAGSearchResult(BaseModel):
    """Result of vector similarity search."""

    model_config = ConfigDict(frozen=True)

    contexts: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Check if the search returned no results."""
        return len(self.contexts) == 0


class RAGQueryResult(BaseModel):
    """Complete query result with answer and metadata."""

    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[str] = Field(default_factory=list)
    num_contexts: int = Field(ge=0)

    @property
    def unique_sources(self) -> list[str]:
        """Get deduplicated list of sources."""
        return list(dict.fromkeys(self.sources))
