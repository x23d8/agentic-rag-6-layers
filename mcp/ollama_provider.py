"""Ollama (local models) provider implementation."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import settings
from mcp.base import ModelProvider


class OllamaProvider(ModelProvider):
    name = "ollama"

    def get_chat_model(self, model: str | None = None, **kwargs: Any) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=kwargs.pop("temperature", 0),
            **kwargs,
        )

    def get_embeddings(self, model: str | None = None, **kwargs: Any) -> Embeddings:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=model or settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
            **kwargs,
        )
