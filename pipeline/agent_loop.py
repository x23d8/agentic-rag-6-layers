"""Layer 5 — Agent Loop: Orchestrator + Critique node with 3 checks."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from mcp.base import get_llm
from pipeline.state import PipelineState

# ---------------------------------------------------------------------------
# Critique Prompts
# ---------------------------------------------------------------------------

SUFFICIENCY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a context sufficiency checker. Determine if the provided "
        "context contains enough information to fully answer the user's query.\n\n"
        "Respond with ONLY:\n"
        "- 'SUFFICIENT' if the context has enough info\n"
        "- 'INSUFFICIENT' if critical information is missing\n\n"
        "Be strict: if the query asks for specific details and the context is vague, "
        "mark as INSUFFICIENT.",
    ),
    (
        "human",
        "Query: {query}\n\nContext:\n{context}\n\nVerdict:",
    ),
])

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a relevance checker. Determine if the retrieved context is "
        "on-topic and relevant to the user's query.\n\n"
        "Respond with ONLY:\n"
        "- 'RELEVANT' if the context is on-topic\n"
        "- 'IRRELEVANT' if the context is mostly off-topic or unrelated\n\n"
        "A context is relevant if at least 50% of it relates to the query.",
    ),
    (
        "human",
        "Query: {query}\n\nContext:\n{context}\n\nVerdict:",
    ),
])

CONFLICT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a conflict detector. Check if the provided context chunks "
        "contain contradictory or conflicting information.\n\n"
        "Respond with ONLY:\n"
        "- 'NO_CONFLICT' if information is consistent\n"
        "- 'CONFLICT: <brief description>' if contradictions exist\n\n"
        "Focus on factual contradictions, not style differences.",
    ),
    (
        "human",
        "Query: {query}\n\nContext:\n{context}\n\nVerdict:",
    ),
])

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a knowledgeable assistant. "
        "CRITICAL INSTRUCTION: You MUST write your entire answer in {language}. "
        "Do NOT use any other language, even if the source documents are in a different language.\n\n"
        "Answer the user's question using ONLY the provided context. Rules:\n"
        "1. Base your answer ONLY on the provided context\n"
        "2. Include inline citations using [Source N] format\n"
        "3. If the context doesn't contain enough information, say so clearly (in {language})\n"
        "4. Be concise but thorough\n"
        "5. Structure your answer with clear sections if the question is complex",
    ),
    (
        "human",
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer in {language} (with citations):",
    ),
])

ABSTAIN_RESPONSE = (
    "Xin lỗi, tôi không có đủ thông tin trong cơ sở dữ liệu để trả lời "
    "câu hỏi này một cách chính xác. Tôi không muốn đưa ra thông tin có thể "
    "không đúng. Vui lòng hỏi một câu hỏi khác trong phạm vi tài liệu đã cung cấp."
)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def critique_check(state: PipelineState) -> dict:
    """Run 3 critique checks: sufficiency, relevance, conflict."""
    llm = get_llm()
    query = state.rewritten_query or state.original_query
    context = state.distilled_context

    if not context.strip():
        return {
            "critique_sufficiency": False,
            "critique_relevance": False,
            "critique_conflict": False,
            "critique_passed": False,
        }

    # Check 1: Sufficiency
    suff_chain = SUFFICIENCY_PROMPT | llm
    suff_result = suff_chain.invoke({"query": query, "context": context})
    is_sufficient = "SUFFICIENT" in suff_result.content.upper() and \
                    "INSUFFICIENT" not in suff_result.content.upper()

    # Check 2: Relevance
    rel_chain = RELEVANCE_PROMPT | llm
    rel_result = rel_chain.invoke({"query": query, "context": context})
    is_relevant = "RELEVANT" in rel_result.content.upper() and \
                  "IRRELEVANT" not in rel_result.content.upper()

    # Check 3: Conflict
    conf_chain = CONFLICT_PROMPT | llm
    conf_result = conf_chain.invoke({"query": query, "context": context})
    no_conflict = "NO_CONFLICT" in conf_result.content.upper()

    passed = is_sufficient and is_relevant and no_conflict

    return {
        "critique_sufficiency": is_sufficient,
        "critique_relevance": is_relevant,
        "critique_conflict": no_conflict,
        "critique_passed": passed,
    }


def _detect_language(text: str) -> str:
    """Return a plain-English language name for the given text.

    Uses langdetect if available, otherwise falls back to a simple
    heuristic based on common Vietnamese diacritics.
    """
    try:
        from langdetect import detect
        code = detect(text)
        _LANG_NAMES = {
            "en": "English", "vi": "Vietnamese", "fr": "French",
            "de": "German",  "es": "Spanish",    "ja": "Japanese",
            "ko": "Korean",  "zh-cn": "Chinese",  "zh-tw": "Chinese",
            "th": "Thai",    "id": "Indonesian",  "pt": "Portuguese",
        }
        return _LANG_NAMES.get(code, "English")
    except Exception:
        # Fallback: check for Vietnamese diacritics
        vi_chars = set("àáâãèéêìíòóôõùúýăđơư")
        if any(c in vi_chars for c in text.lower()):
            return "Vietnamese"
        return "English"


def generate_answer(state: PipelineState) -> dict:
    """Generate grounded answer with citations, in the user's language."""
    # If should abstain, return abstain message
    if state.should_abstain:
        return {
            "answer": ABSTAIN_RESPONSE,
            "citations": [],
        }

    llm = get_llm()
    # Always use the *original* query for language detection so we match
    # what the user literally typed, not the rewritten version.
    original = state.original_query
    query    = state.rewritten_query or original
    context  = state.distilled_context
    language = _detect_language(original)

    # If no context (lazy router skipped retrieval), use conversation history
    if not context.strip() and state.conversation_history:
        context = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.conversation_history[-6:]
        )

    chain = GENERATE_PROMPT | llm
    result = chain.invoke({"query": query, "context": context, "language": language})

    # Extract citations from reranked results
    citations = []
    if state.reranked_results:
        for i, doc in enumerate(state.reranked_results):
            citations.append({
                "source_id": i + 1,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", -1),
                "rerank_score": doc.metadata.get("rerank_score", 0.0),
                "snippet": doc.page_content[:200],
            })

    return {
        "answer": result.content,
        "citations": citations,
    }


def decide_after_critique(state: PipelineState) -> str:
    """Conditional edge: decide next step after critique.

    Returns:
        'generate' — critique passed, proceed to answer
        'retry'    — critique failed, retry retrieval (max 2x)
        'abstain'  — max retries exceeded, abstain
    """
    if state.critique_passed:
        return "generate"

    if state.retry_count < 2:
        return "retry"

    return "abstain"


def increment_retry(state: PipelineState) -> dict:
    """Increment retry counter before re-retrieval."""
    return {"retry_count": state.retry_count + 1}


def mark_abstain(state: PipelineState) -> dict:
    """Mark state as abstain — system refuses to answer."""
    return {"should_abstain": True}
