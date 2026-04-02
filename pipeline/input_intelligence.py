"""Layer 1 — Input Intelligence: HyDE rewriting + sub-question decomposition."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from mcp.base import get_llm
from pipeline.state import PipelineState

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Given a user question, write a short "
        "hypothetical passage (3-5 sentences) that would perfectly answer the "
        "question. This passage will be used for semantic retrieval, so make it "
        "detailed and factual. Write in the same language as the question.",
    ),
    ("human", "{query}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query rewriter. Rewrite the user's question to be more "
        "specific, self-contained, and optimized for search retrieval. "
        "Resolve pronouns using conversation history if available. "
        "Return ONLY the rewritten query, nothing else. "
        "Write in the same language as the original question.",
    ),
    ("human", "Conversation history:\n{history}\n\nCurrent question: {query}"),
])

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a question decomposer. Given a complex question, break it "
        "down into 2-4 simpler sub-questions that, when answered together, "
        "would fully answer the original question.\n"
        "Return ONLY the sub-questions, one per line, prefixed with numbers.\n"
        "If the question is already simple enough, return just the original question.\n"
        "Write in the same language as the question.",
    ),
    ("human", "{query}"),
])


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def query_rewrite(state: PipelineState) -> dict:
    """Rewrite the user query using conversation context + HyDE."""
    llm = get_llm()

    # Format conversation history
    history_str = ""
    if state.conversation_history:
        history_str = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.conversation_history[-6:]
        )

    # Rewrite query
    rewrite_chain = REWRITE_PROMPT | llm
    rewritten = rewrite_chain.invoke({
        "query": state.original_query,
        "history": history_str or "(no prior conversation)",
    })

    # Generate HyDE passage
    hyde_chain = HYDE_PROMPT | llm
    hyde = hyde_chain.invoke({"query": rewritten.content})

    return {
        "rewritten_query": rewritten.content,
        "hyde_passage": hyde.content,
    }


def decompose_query(state: PipelineState) -> dict:
    """Decompose complex query into sub-questions."""
    llm = get_llm()
    chain = DECOMPOSE_PROMPT | llm

    result = chain.invoke({"query": state.rewritten_query or state.original_query})
    lines = [
        line.strip().lstrip("0123456789.)- ")
        for line in result.content.strip().split("\n")
        if line.strip()
    ]

    # If only 1 sub-question that's basically the same, keep empty
    if len(lines) <= 1:
        return {"sub_questions": []}

    return {"sub_questions": lines}
