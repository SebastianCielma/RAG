"""Centralized configuration for RAG application using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        groq_api_key: API key for Groq LLM service.
        qdrant_url: URL for Qdrant vector database.
        qdrant_collection: Name of the Qdrant collection.
        inngest_api_base: Base URL for Inngest API.
        embed_model: Name of the Sentence Transformers model.
        embed_dim: Dimension of the embedding vectors.
        chunk_size: Size of text chunks for splitting.
        chunk_overlap: Overlap between consecutive chunks.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    groq_api_key: str = ""

    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"
    qdrant_timeout: int = 30

    # Inngest Configuration
    inngest_api_base: str = "http://127.0.0.1:8288/v1"
    inngest_app_id: str = "rag_app"

    # Embedding Configuration
    embed_model: str = "all-MiniLM-L6-v2"
    embed_dim: int = 384

    # Text Splitting Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # LLM Configuration
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached settings instance.

    Returns:
        Singleton Settings instance loaded from environment.
    """
    return Settings()
