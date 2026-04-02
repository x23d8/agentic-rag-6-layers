"""OpenAI provider implementation."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import settings
from mcp.base import ModelProvider


class OpenAIProvider(ModelProvider):
    name = "openai"

    def get_chat_model(self, model: str | None = None, **kwargs: Any) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=kwargs.pop("temperature", 0),
            **kwargs,
        )

    def get_embeddings(self, model: str | None = None, **kwargs: Any) -> Embeddings:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=model or settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            **kwargs,
        )
