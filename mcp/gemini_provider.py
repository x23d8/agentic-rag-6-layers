"""Google Gemini provider implementation."""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from config import settings
from mcp.base import ModelProvider


class GeminiProvider(ModelProvider):
    name = "gemini"

    def get_chat_model(self, model: str | None = None, **kwargs: Any) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=kwargs.pop("temperature", 0),
            **kwargs,
        )

    def get_embeddings(self, model: str | None = None, **kwargs: Any) -> Embeddings:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=model or settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
            **kwargs,
        )
