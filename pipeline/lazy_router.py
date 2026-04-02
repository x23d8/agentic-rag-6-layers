"""Layer 2 — Lazy Router: decide if retrieval is needed."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from mcp.base import get_llm
from pipeline.state import PipelineState

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a retrieval router. Analyze whether the user's question "
        "requires retrieving information from a knowledge base, or if it can "
        "be answered directly from the conversation history.\n\n"
        "Rules:\n"
        "- If the question is a simple follow-up (e.g., 'yes', 'tell me more', "
        "'can you explain that?') AND the conversation history already contains "
        "enough context → respond NO\n"
        "- If the question asks for new factual information, specific data, or "
        "references → respond YES\n"
        "- When in doubt → respond YES (safer to retrieve than to miss)\n\n"
        "Respond with ONLY 'YES' or 'NO'.",
    ),
    (
        "human",
        "Conversation history:\n{history}\n\n"
        "Current question: {query}\n\n"
        "Does this question require retrieval? (YES/NO)",
    ),
])


def lazy_route(state: PipelineState) -> dict:
    """Decide whether to retrieve or skip straight to generation."""
    # First query in session → always retrieve
    if not state.conversation_history:
        return {"needs_retrieval": True}

    llm = get_llm()
    chain = ROUTER_PROMPT | llm

    history_str = "\n".join(
        f"{m['role']}: {m['content']}" for m in state.conversation_history[-6:]
    )

    result = chain.invoke({
        "query": state.original_query,
        "history": history_str,
    })

    needs = "YES" in result.content.upper()
    return {"needs_retrieval": needs}
