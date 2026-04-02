"""Layer 3 — Hybrid Retrieval: Dense + BM25 → RRF → Cross-encoder reranker."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from config import settings
from pipeline.state import PipelineState


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """Fuse multiple ranked lists using RRF.

    Score(d) = sum over lists of 1 / (k + rank(d))
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            doc_id = doc.page_content[:200]  # use content prefix as key
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
                scores[doc_id] = 0.0
            scores[doc_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [doc_map[did] for did in sorted_ids]


# ---------------------------------------------------------------------------
# Cross-encoder Reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker:
    """Rerank documents using a cross-encoder model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.reranker_model
        self._model = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        """Score and rerank documents by relevance to query."""
        top_k = top_k or settings.rerank_top_k
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)
        return results


# Singleton reranker
_reranker: CrossEncoderReranker | None = None


def _get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def hybrid_retrieve(state: PipelineState, store: Any) -> dict:
    """Run dense + sparse search and return raw results."""
    query = state.rewritten_query or state.original_query

    # If we have sub-questions, retrieve for each and merge
    queries = [query]
    if state.sub_questions:
        queries.extend(state.sub_questions)

    # Also use HyDE passage for dense retrieval
    if state.hyde_passage:
        queries.append(state.hyde_passage)

    all_dense: list[Document] = []
    all_sparse: list[Document] = []

    for q in queries:
        dense, sparse = store.hybrid_search(q)
        all_dense.extend(dense)
        all_sparse.extend(sparse)

    return {
        "dense_results": all_dense,
        "sparse_results": all_sparse,
    }


def fuse_results(state: PipelineState) -> dict:
    """Apply Reciprocal Rank Fusion on dense + sparse results."""
    fused = reciprocal_rank_fusion([state.dense_results, state.sparse_results])
    return {"fused_results": fused}


def rerank_results(state: PipelineState) -> dict:
    """Rerank fused results with cross-encoder."""
    query = state.rewritten_query or state.original_query
    reranker = _get_reranker()
    reranked = reranker.rerank(query, state.fused_results)
    return {"reranked_results": reranked}
