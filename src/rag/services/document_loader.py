"""Document loading and validation service."""

import logging
from functools import lru_cache
from pathlib import Path

import docx
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from rag.core.config import get_settings
from rag.core.exceptions import PDFLoadError

logger = logging.getLogger(__name__)


class DocumentLoadError(Exception):
    """Raised when document loading fails."""


@lru_cache(maxsize=1)
def get_text_splitter() -> SentenceSplitter:
    """Get or create the text splitter singleton."""
    settings = get_settings()
    return SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def load_pdf(path: Path) -> list[str]:
    """Load and parse PDF file."""
    try:
        logger.info("Loading PDF: %s", path.name)
        docs = PDFReader().load_data(file=path)
        return [
            doc.text
            for doc in docs
            if getattr(doc, "text", None) and doc.text.strip()
        ]
    except Exception as exc:
        raise PDFLoadError(f"Failed to load PDF: {path}") from exc


def load_docx(path: Path) -> list[str]:
    """Load and parse DOCX file."""
    try:
        logger.info("Loading DOCX: %s", path.name)
        doc = docx.Document(path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        return ["\n".join(full_text)] if full_text else []
    except Exception as exc:
        raise DocumentLoadError(f"Failed to load DOCX: {path}") from exc


def load_text(path: Path) -> list[str]:
    """Load and parse TXT or Markdown file."""
    try:
        logger.info("Loading Text file: %s", path.name)
        text = path.read_text(encoding="utf-8")
        return [text] if text.strip() else []
    except Exception as exc:
        raise DocumentLoadError(f"Failed to load text file: {path}") from exc


def load_and_chunk_document(path: str | Path) -> list[str]:
    """Load document based on extension and split into chunks."""
    doc_path = Path(path) if isinstance(path, str) else path

    if not doc_path.exists():
        raise DocumentLoadError(f"File not found: {doc_path}")

    ext = doc_path.suffix.lower()
    texts: list[str] = []

    try:
        if ext == ".pdf":
            texts = load_pdf(doc_path)
        elif ext == ".docx":
            texts = load_docx(doc_path)
        elif ext in (".txt", ".md"):
            texts = load_text(doc_path)
        else:
            raise DocumentLoadError(f"Unsupported file format: {ext}")

    except Exception as exc:
        logger.exception("Failed to load document: %s", doc_path)
        raise DocumentLoadError(f"Failed to process {doc_path}") from exc

    if not texts:
        logger.warning("No text extracted from: %s", doc_path.name)
        return []

    splitter = get_text_splitter()
    chunks: list[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))

    logger.info(
        "Extracted %d chunks from %s (format: %s)",
        len(chunks),
        doc_path.name,
        ext,
    )
    return chunks
