"""Central registry that maps provider names → provider instances."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import settings
from mcp.base import ModelProvider


class ProviderRegistry:
    """Singleton registry for all model providers."""

    def __init__(self) -> None:
        self._providers: dict[str, ModelProvider] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        from mcp.openai_provider import OpenAIProvider
        from mcp.gemini_provider import GeminiProvider
        from mcp.ollama_provider import OllamaProvider

        for cls in (OpenAIProvider, GeminiProvider, OllamaProvider):
            p = cls()
            self._providers[p.name] = p

    def register(self, provider: ModelProvider) -> None:
        self._providers[provider.name] = provider

    def get_provider(self, name: str | None = None) -> ModelProvider:
        name = name or settings.default_llm_provider
        if name not in self._providers:
            available = ", ".join(self._providers)
            raise ValueError(
                f"Unknown provider '{name}'. Available: {available}"
            )
        return self._providers[name]

    def get_llm(
        self,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> BaseChatModel:
        return self.get_provider(provider).get_chat_model(model, **kwargs)

    def get_embeddings(
        self,
        provider: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> Embeddings:
        return self.get_provider(provider).get_embeddings(model, **kwargs)

    @property
    def available_providers(self) -> list[str]:
        return list(self._providers.keys())


provider_registry = ProviderRegistry()
