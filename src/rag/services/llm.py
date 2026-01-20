"""LLM interaction service with streaming support."""

import json
from collections.abc import AsyncGenerator, Generator
from typing import Any

from groq import AsyncGroq

from rag.core.config import get_settings
from rag.models.schemas import LLMModel

settings = get_settings()


async def stream_chat(
    messages: list[dict[str, str]],
    model_name: str,
    temperature: float = 0.2,
) -> AsyncGenerator[str, None]:
    """Stream chat response from Groq LLM."""
    client = AsyncGroq(api_key=settings.groq_api_key)

    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=settings.llm_max_tokens,
            stream=True,
        )

        async for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as exc:
        yield f"\n\n[Error generating response: {exc}]"
