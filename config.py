from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Provider selection
    default_llm_provider: str = "openai"
    default_embedding_provider: str = "openai"

    # API keys
    openai_api_key: str = os.getenv('OPENAI_API_KEY')
    google_api_key: str = os.getenv('GOOGLE_API_KEY')
    ollama_base_url: str = "http://localhost:11434"

    # Model names
    openai_model: str = "gpt-4o"
    gemini_model: str = "gemini-2.0-flash"
    ollama_model: str = "llama3.1"

    # Embedding models (per-provider)
    openai_embedding_model: str = "text-embedding-3-small"
    gemini_embedding_model: str = "models/text-embedding-004"
    ollama_embedding_model: str = "Qwen3-Embedding-0.6B"

    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval params
    retrieval_top_k: int = 20
    rerank_top_k: int = 5
    bm25_top_k: int = 20

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "rag_docs"

    # Agent loop
    max_retries: int = 2

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
