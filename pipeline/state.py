"""LangGraph state definition for the RAG pipeline."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    """Shared state flowing through every node of the LangGraph."""

    # -- Input --
    original_query: str = ""
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    # -- Layer 1: Input Intelligence --
    rewritten_query: str = ""
    hyde_passage: str = ""
    sub_questions: list[str] = Field(default_factory=list)

    # -- Layer 2: Lazy Router --
    needs_retrieval: bool = True

    # -- Layer 3: Hybrid Retrieval --
    dense_results: list[Document] = Field(default_factory=list)
    sparse_results: list[Document] = Field(default_factory=list)
    fused_results: list[Document] = Field(default_factory=list)
    reranked_results: list[Document] = Field(default_factory=list)

    # -- Layer 4: Context Distillation --
    distilled_context: str = ""

    # -- Layer 5: Agent Loop --
    critique_sufficiency: bool = False
    critique_relevance: bool = False
    critique_conflict: bool = False
    critique_passed: bool = False
    retry_count: int = 0
    should_abstain: bool = False

    # -- Layer 6: Output --
    answer: str = ""
    citations: list[dict[str, Any]] = Field(default_factory=list)
    ragas_scores: dict[str, float] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True
