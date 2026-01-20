"""Inngest workflow functions for RAG application."""

import logging
import uuid
from typing import Any

import inngest
from inngest.experimental import ai

from rag.core.config import get_settings
from rag.db.qdrant import get_storage
from rag.models.schemas import (
    LLMModel,
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
)
from rag.services.embeddings import embed_texts
from rag.services.document_loader import load_and_chunk_document

logger = logging.getLogger(__name__)
settings = get_settings()

inngest_client = inngest.Inngest(
    app_id=settings.inngest_app_id,
    logger=logging.getLogger("inngest"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context) -> dict[str, Any]:
    """Ingest a PDF file: load, chunk, embed, and store in vector database."""

    def _load() -> RAGChunkAndSrc:
        file_path: str = ctx.event.data["file_path"]
        source_id: str = ctx.event.data.get("source_id", file_path)
        chunks = load_and_chunk_document(file_path)
        logger.info("Loaded %d chunks from %s", len(chunks), source_id)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        if not chunks:
            logger.warning("No chunks to upsert for %s", source_id)
            return RAGUpsertResult(ingested=0)

        vectors = embed_texts(chunks)
        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]
        payloads = [{"source": source_id, "text": chunk} for chunk in chunks]

        storage = get_storage()
        storage.upsert(ids, vectors, payloads)
        logger.info("Upserted %d chunks for %s", len(chunks), source_id)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        _load,
        output_type=RAGChunkAndSrc,
    )
    result = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )
    return result.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> dict[str, Any]:
    """Query PDFs using RAG: search for relevant chunks and generate answer."""

    def _search(
        question: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        storage = get_storage()
        found = storage.search(query_vec, top_k, source_filter=source_filter)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question: str = ctx.event.data["question"]
    top_k: int = int(ctx.event.data.get("top_k", 5))
    model_name: str = ctx.event.data.get("model", LLMModel.default().value)
    source_filter: str | None = ctx.event.data.get("source_filter")

    try:
        selected_model = LLMModel(model_name)
    except ValueError:
        logger.warning("Invalid model '%s', falling back to default", model_name)
        selected_model = LLMModel.default()

    logger.info("Processing query with model %s", selected_model.value)

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k, source_filter),
        output_type=RAGSearchResult,
    )

    if found.is_empty:
        logger.warning("No relevant context found for query")
        return RAGQueryResult(
            answer="No relevant information found in the documents.",
            sources=[],
            num_contexts=0,
        ).model_dump()

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=settings.groq_api_key,
        model=selected_model.value,
        base_url="https://api.groq.com/openai/v1",
    )

    response = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer questions using only the provided context.",
                },
                {"role": "user", "content": user_content},
            ],
        },
    )

    answer: str = response["choices"][0]["message"]["content"].strip()
    result = RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    )
    logger.info("Generated answer with %d contexts", result.num_contexts)
    output = result.model_dump()
    output["contexts"] = found.contexts
    return output


@inngest_client.create_function(
    fn_id="RAG: List Documents",
    trigger=inngest.TriggerEvent(event="rag/list_documents"),
)
async def rag_list_documents(ctx: inngest.Context) -> dict[str, Any]:
    """List all available documents in the vector database."""

    def _list_sources() -> list[str]:
        storage = get_storage()
        return storage.list_sources()

    sources = await ctx.step.run("list-sources", _list_sources)
    logger.info("Found %d documents", len(sources))
    return {"documents": sources}


@inngest_client.create_function(
    fn_id="RAG: Delete Document",
    trigger=inngest.TriggerEvent(event="rag/delete_document"),
)
async def rag_delete_document(ctx: inngest.Context) -> dict[str, Any]:
    """Delete a document from the vector database."""
    source_id: str = ctx.event.data["source_id"]

    def _delete() -> int:
        storage = get_storage()
        return storage.delete_by_source(source_id)

    result = await ctx.step.run("delete-source", _delete)
    logger.info("Deleted document: %s", source_id)
    return {"deleted": result > 0, "source_id": source_id}
