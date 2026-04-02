"""Vector store (ChromaDB) + BM25 sparse index management."""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from config import settings


class DocumentStore:
    """Manages both dense (ChromaDB) and sparse (BM25) indexes.

    Embeddings come from the active provider via the MCP registry,
    so each provider uses its own best embedding model:
      - OpenAI  → text-embedding-3-small
      - Gemini  → text-embedding-004
      - Ollama  → Qwen3-Embedding-0.6B
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embeddings: Embeddings | None = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection

        if embeddings is not None:
            self.embeddings = embeddings
        else:
            from mcp.base import get_embeddings

            provider = settings.default_embedding_provider
            self.embeddings = get_embeddings(provider=provider)

        self._vectorstore = None
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []

    # -- ChromaDB dense store --------------------------------------------------

    @property
    def vectorstore(self) -> Any:
        if self._vectorstore is None:
            from langchain_community.vectorstores import Chroma

            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._vectorstore

    # -- BM25 sparse index -----------------------------------------------------

    def _build_bm25(self, documents: list[Document]) -> None:
        """Build BM25 index from document page_content."""
        tokenized = [doc.page_content.lower().split() for doc in documents]
        self._bm25 = BM25Okapi(tokenized)
        self._docs = list(documents)

    # -- Public API ------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Index documents into both dense and sparse stores."""
        if not documents:
            return

        # Deduplicate by content hash
        seen: set[str] = set()
        unique: list[Document] = []
        for doc in documents:
            h = hashlib.md5(doc.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(doc)

        # Dense index
        self.vectorstore.add_documents(unique)

        # Sparse index — rebuild from all docs
        existing = self._docs[:]
        existing.extend(unique)
        self._build_bm25(existing)

    def dense_search(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve top-k docs via dense embedding similarity."""
        k = k or settings.retrieval_top_k
        return self.vectorstore.similarity_search(query, k=k)

    def sparse_search(self, query: str, k: int | None = None) -> list[Document]:
        """Retrieve top-k docs via BM25."""
        k = k or settings.bm25_top_k
        if self._bm25 is None or not self._docs:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._docs[i] for i in top_indices if scores[i] > 0]

    def hybrid_search(
        self,
        query: str,
        dense_k: int | None = None,
        sparse_k: int | None = None,
    ) -> tuple[list[Document], list[Document]]:
        """Return (dense_results, sparse_results) for RRF fusion downstream."""
        dense = self.dense_search(query, k=dense_k)
        sparse = self.sparse_search(query, k=sparse_k)
        return dense, sparse
