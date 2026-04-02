"""Embedding access — delegates to the active MCP provider.

Each provider uses its own best embedding model:
  - OpenAI  → text-embedding-3-small
  - Gemini  → text-embedding-004
  - Ollama  → Qwen3-Embedding-0.6B
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


def get_provider_embeddings(provider: str | None = None) -> Embeddings:
    """Return the embedding model for the given (or default) provider."""
    from mcp.base import get_embeddings

    return get_embeddings(provider=provider)
