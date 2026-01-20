"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing."""
    return [
        "This is a test document about artificial intelligence.",
        "Machine learning is a subset of AI.",
        "Python is a great programming language.",
    ]


@pytest.fixture
def sample_chunks() -> list[str]:
    """Sample chunks for testing vector operations."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Hello world, this is a test chunk for RAG system.",
    ]
