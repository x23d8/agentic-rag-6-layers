"""Token-aware recursive chunking for documents.

Uses a provider-appropriate tokenizer as the length function so chunk
sizes are measured in actual tokens, not characters.
"""

from __future__ import annotations

from typing import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# Cache keyed by provider name so switching providers rebuilds the counter.
_cache: dict[str, Callable[[str], int]] = {}


def _build_token_counter(provider: str) -> Callable[[str], int]:
    """Return a token-counting function matched to the embedding provider."""
    if provider == "ollama":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True,
        )
        return lambda text: len(tokenizer.encode(text, add_special_tokens=False))

    # OpenAI and Gemini both use BPE-style tokenizers;
    # cl100k_base is exact for OpenAI, close enough for Gemini.
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(enc.encode(text))


def _get_token_length() -> Callable[[str], int]:
    provider = settings.default_embedding_provider
    if provider not in _cache:
        _cache[provider] = _build_token_counter(provider)
    return _cache[provider]


def recursive_chunk(
    documents: list[Document],
    chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> list[Document]:
    """Split documents into token-measured, semantically coherent chunks.

    Uses RecursiveCharacterTextSplitter with a provider-matched tokenizer
    as the length function.  Separators are tuned for Vietnamese + English
    mixed content.
    """
    token_len = _get_token_length()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",   # paragraph breaks
            "\n",     # line breaks
            ". ",     # sentence boundaries (EN / VI)
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
        length_function=token_len,
    )

    chunks: list[Document] = []
    for doc in documents:
        splits = splitter.split_documents([doc])
        for i, split in enumerate(splits):
            split.metadata["chunk_index"] = i
            split.metadata["source"] = doc.metadata.get("source", "unknown")
            split.metadata["token_count"] = token_len(split.page_content)
        chunks.extend(splits)

    return chunks
