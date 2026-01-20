"""RAG Application entry point with FastAPI."""

import json
import logging
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from typing import Any

import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from rag.core.config import get_settings
from rag.core.exceptions import RAGError
from rag.db.qdrant import get_storage
from rag.services.embeddings import embed_texts
from rag.services.embeddings import warmup as warmup_embeddings
from rag.services.llm import stream_chat
from rag.workflows.inngest import (
    inngest_client,
    rag_delete_document,
    rag_ingest_pdf,
    rag_list_documents,
    rag_query_pdf_ai,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("Starting application warmup...")
    try:
        warmup_embeddings()
        _ = get_storage()
        logger.info("Application warmup complete")
    except Exception:
        logger.exception("Warmup failed, continuing anyway")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="RAG Application",
    description="Retrieval-Augmented Generation API with Inngest workflows",
    version="1.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    question: str
    model: str
    top_k: int = 5
    source_filter: str | None = None


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": "1.1.0",
        "service": "rag-api",
    }


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest) -> StreamingResponse:
    """Stream chat response with RAG and citations."""
    logger.info("Received chat request: %s", request.question[:50])

    # 1. Embed query
    query_vec = embed_texts([request.question])[0]

    # 2. Search in Vector DB
    storage = get_storage()
    found = storage.search(
        query_vec,
        request.top_k,
        source_filter=request.source_filter,
    )
    contexts = found["contexts"]
    sources = found["sources"]

    # 3. Build Context Block with Citations indices
    # Format: [1] Source: filename.pdf\nContext text...
    context_parts = []
    unique_sources_map = {}
    source_index = 1

    for i, (ctx, src) in enumerate(zip(contexts, sources, strict=True), 1):
        context_parts.append(f"[{i}] Source: {src}\n{ctx}")
        if src not in unique_sources_map:
            unique_sources_map[src] = source_index
            source_index += 1

    context_block = "\n\n".join(context_parts)

    system_prompt = (
        "You are an intelligent assistant answering questions based on the provided documents.\n"
        "Strictly follow these rules:\n"
        "1. Use ONLY the provided context to answer.\n"
        "2. If the answer is not in the context, say so.\n"
        "3. Be concise and professional."
    )

    user_content = (
        f"Context:\n{context_block}\n\n"
        f"Question: {request.question}\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    async def response_generator() -> AsyncGenerator[str, None]:
        # Yield metadata first (sources/contexts) as a special JSON line
        metadata = {
            "sources": sources,
            "contexts": contexts,
        }
        yield json.dumps(metadata) + "\n"

        # Stream tokens
        async for chunk in stream_chat(
            messages=messages,
            model_name=request.model,
            temperature=settings.llm_temperature,
        ):
            yield chunk

    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream",
    )


@app.exception_handler(RAGError)
async def rag_error_handler(request: Any, exc: RAGError) -> JSONResponse:
    """Handle RAG-specific errors with proper HTTP responses."""
    logger.error("RAG error: %s", exc.message)
    return JSONResponse(
        status_code=500,
        content={"error": exc.message, "type": type(exc).__name__},
    )


# Register Inngest functions
inngest.fast_api.serve(
    app,
    inngest_client,
    [rag_ingest_pdf, rag_query_pdf_ai, rag_list_documents, rag_delete_document],
)
