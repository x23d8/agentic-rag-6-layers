"""Layer 4 — Context Distillation: compress chunks, prune noise."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from mcp.base import get_llm
from pipeline.state import PipelineState

DISTILL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a context distiller. Given a user query and retrieved text chunks, "
        "extract and compress ONLY the parts that are directly relevant to answering "
        "the query. Remove noise, redundancy, and off-topic content.\n\n"
        "Rules:\n"
        "- Keep factual details, numbers, names that help answer the query\n"
        "- Remove boilerplate, repetitive content, and irrelevant sections\n"
        "- Preserve source attribution markers [Source N] for citation\n"
        "- Output in the same language as the chunks\n"
        "- If a chunk is completely irrelevant, skip it entirely\n"
        "- Keep the output concise but complete",
    ),
    (
        "human",
        "Query: {query}\n\n"
        "Retrieved chunks:\n{chunks}\n\n"
        "Distilled context:",
    ),
])


def distill_context(state: PipelineState) -> dict:
    """Compress retrieved chunks into focused context."""
    if not state.reranked_results:
        return {"distilled_context": ""}

    # Format chunks with source markers
    chunk_texts = []
    for i, doc in enumerate(state.reranked_results):
        source = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("rerank_score", "N/A")
        chunk_texts.append(
            f"[Source {i + 1}] (file: {source}, relevance: {score})\n"
            f"{doc.page_content}"
        )

    chunks_str = "\n\n---\n\n".join(chunk_texts)
    query = state.rewritten_query or state.original_query

    llm = get_llm()
    chain = DISTILL_PROMPT | llm
    result = chain.invoke({"query": query, "chunks": chunks_str})

    return {"distilled_context": result.content}
