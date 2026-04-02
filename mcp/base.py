"""Abstract base for Model Control Plane (MCP) — flexible multi-provider switching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


class ModelProvider(ABC):
    """Abstract interface that every LLM provider must implement."""

    name: str

    @abstractmethod
    def get_chat_model(self, model: str | None = None, **kwargs: Any) -> BaseChatModel:
        """Return a LangChain ChatModel instance."""

    @abstractmethod
    def get_embeddings(self, model: str | None = None, **kwargs: Any) -> Embeddings:
        """Return a LangChain Embeddings instance."""


# ---------------------------------------------------------------------------
# Convenience factory functions — import these in the rest of the codebase
# ---------------------------------------------------------------------------

def get_llm(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Get a chat model from the specified (or default) provider."""
    from mcp.registry import provider_registry

    return provider_registry.get_llm(provider, model, **kwargs)


def get_embeddings(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Get an embeddings model from the specified (or default) provider."""
    from mcp.registry import provider_registry

    return provider_registry.get_embeddings(provider, model, **kwargs)
