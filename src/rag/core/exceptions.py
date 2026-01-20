"""Custom exceptions for RAG application.

This module defines a hierarchy of exceptions for proper error handling
throughout the application, enabling specific error catching and reporting.
"""


class RAGError(Exception):
    """Base exception for all RAG application errors."""

    def __init__(self, message: str, *args: object) -> None:
        """Initialize RAGError with a message."""
        self.message = message
        super().__init__(message, *args)


class PDFLoadError(RAGError):
    """Raised when PDF loading or parsing fails."""


class EmbeddingError(RAGError):
    """Raised when text embedding fails."""


class VectorDBError(RAGError):
    """Raised when vector database operations fail."""


class LLMError(RAGError):
    """Raised when LLM inference fails."""


class ConfigurationError(RAGError):
    """Raised when configuration is invalid or missing."""
