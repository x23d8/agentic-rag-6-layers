"""LangGraph StateGraph — wires all 6 layers into a single compiled graph."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, StateGraph

from indexing.store import DocumentStore
from pipeline.state import PipelineState
from pipeline.input_intelligence import query_rewrite, decompose_query
from pipeline.lazy_router import lazy_route
from pipeline.hybrid_retrieval import hybrid_retrieve, fuse_results, rerank_results
from pipeline.context_distillation import distill_context
from pipeline.agent_loop import (
    critique_check,
    decide_after_critique,
    generate_answer,
    increment_retry,
    mark_abstain,
)
from pipeline.output import format_output
from evaluation.ragas_eval import ragas_evaluate


def build_rag_graph(store: DocumentStore, enable_ragas: bool = True) -> Any:
    """Build and compile the full 6-layer RAG pipeline graph.

    Graph structure:
    ┌─────────────────────────────────────────────────────────┐
    │ Layer 1: query_rewrite → decompose_query                │
    │ Layer 2: lazy_route ─┬─ yes → retrieval path            │
    │                      └─ no  → generate directly         │
    │ Layer 3: hybrid_retrieve → fuse → rerank                │
    │ Layer 4: distill_context                                │
    │ Layer 5: critique ─┬─ pass → generate                   │
    │                    ├─ retry → increment → hybrid_retrieve│
    │                    └─ abstain → mark_abstain → generate  │
    │ Layer 6: generate → ragas_eval → format_output → END    │
    └─────────────────────────────────────────────────────────┘
    """

    # Bind store to hybrid_retrieve
    def _hybrid_retrieve(state: PipelineState) -> dict:
        return hybrid_retrieve(state, store)

    # Build the graph
    graph = StateGraph(PipelineState)

    # ── Layer 1: Input Intelligence ──
    graph.add_node("query_rewrite", query_rewrite)
    graph.add_node("decompose_query", decompose_query)

    # ── Layer 2: Lazy Router ──
    graph.add_node("lazy_route", lazy_route)

    # ── Layer 3: Hybrid Retrieval ──
    graph.add_node("hybrid_retrieve", _hybrid_retrieve)
    graph.add_node("fuse_results", fuse_results)
    graph.add_node("rerank_results", rerank_results)

    # ── Layer 4: Context Distillation ──
    graph.add_node("distill_context", distill_context)

    # ── Layer 5: Agent Loop ──
    graph.add_node("critique_check", critique_check)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("mark_abstain", mark_abstain)

    # ── Layer 6: Output ──
    graph.add_node("generate_answer", generate_answer)
    if enable_ragas:
        graph.add_node("ragas_evaluate", ragas_evaluate)
    graph.add_node("format_output", format_output)

    # ═══════════════════════════════════════════════════════
    # EDGES — wire the flow
    # ═══════════════════════════════════════════════════════

    # Entry: start → query_rewrite
    graph.set_entry_point("query_rewrite")

    # Layer 1 flow
    graph.add_edge("query_rewrite", "decompose_query")
    graph.add_edge("decompose_query", "lazy_route")

    # Layer 2: conditional routing
    graph.add_conditional_edges(
        "lazy_route",
        lambda state: "retrieve" if state.needs_retrieval else "skip",
        {
            "retrieve": "hybrid_retrieve",
            "skip": "generate_answer",
        },
    )

    # Layer 3 flow
    graph.add_edge("hybrid_retrieve", "fuse_results")
    graph.add_edge("fuse_results", "rerank_results")

    # Layer 3 → Layer 4
    graph.add_edge("rerank_results", "distill_context")

    # Layer 4 → Layer 5
    graph.add_edge("distill_context", "critique_check")

    # Layer 5: conditional after critique
    graph.add_conditional_edges(
        "critique_check",
        decide_after_critique,
        {
            "generate": "generate_answer",
            "retry": "increment_retry",
            "abstain": "mark_abstain",
        },
    )

    # Retry loop: increment → back to retrieval
    graph.add_edge("increment_retry", "hybrid_retrieve")

    # Abstain → generate (will output abstain message)
    graph.add_edge("mark_abstain", "generate_answer")

    # Layer 6 flow
    if enable_ragas:
        graph.add_edge("generate_answer", "ragas_evaluate")
        graph.add_edge("ragas_evaluate", "format_output")
    else:
        graph.add_edge("generate_answer", "format_output")

    graph.add_edge("format_output", END)

    return graph.compile()
