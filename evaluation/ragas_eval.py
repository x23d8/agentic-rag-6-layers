"""Ragas post-evaluation: faithfulness, answer_relevancy, context_precision."""

from __future__ import annotations

import logging
from typing import Any

from pipeline.state import PipelineState

logger = logging.getLogger(__name__)


def ragas_evaluate(state: PipelineState) -> dict:
    """Run Ragas evaluation on the generated answer.

    Metrics:
    - faithfulness: is the answer grounded in context?
    - answer_relevancy: does the answer address the query?
    - context_precision: is the retrieved context relevant?
    """
    if state.should_abstain:
        return {
            "ragas_scores": {
                "faithfulness": 1.0,  # abstaining is perfectly faithful
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "abstained": True,
            }
        }

    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )
        from datasets import Dataset

        # Build evaluation dataset
        contexts = []
        if state.reranked_results:
            contexts = [doc.page_content for doc in state.reranked_results]

        eval_data = {
            "question": [state.rewritten_query or state.original_query],
            "answer": [state.answer],
            "contexts": [contexts],
        }

        dataset = Dataset.from_dict(eval_data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
        }

    except Exception as e:
        logger.warning(f"Ragas evaluation failed: {e}. Using LLM-based fallback.")
        scores = _llm_fallback_eval(state)

    return {"ragas_scores": scores}


def _llm_fallback_eval(state: PipelineState) -> dict[str, float]:
    """Fallback evaluation using LLM when Ragas is unavailable."""
    from langchain_core.prompts import ChatPromptTemplate
    from mcp.base import get_llm

    EVAL_PROMPT = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an evaluation judge. Score the following on a scale of 0.0 to 1.0:\n"
            "1. faithfulness: Is the answer fully grounded in the context? (no hallucination)\n"
            "2. answer_relevancy: Does the answer directly address the question?\n"
            "3. context_precision: Is the context relevant to the question?\n\n"
            "Respond in EXACTLY this format:\n"
            "faithfulness: 0.X\n"
            "answer_relevancy: 0.X\n"
            "context_precision: 0.X",
        ),
        (
            "human",
            "Question: {query}\n\n"
            "Context: {context}\n\n"
            "Answer: {answer}\n\n"
            "Scores:",
        ),
    ])

    llm = get_llm()
    chain = EVAL_PROMPT | llm
    result = chain.invoke({
        "query": state.rewritten_query or state.original_query,
        "context": state.distilled_context[:2000],
        "answer": state.answer[:2000],
    })

    scores = {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5}
    for line in result.content.strip().split("\n"):
        for metric in scores:
            if metric in line.lower():
                try:
                    val = float(line.split(":")[-1].strip())
                    scores[metric] = max(0.0, min(1.0, val))
                except ValueError:
                    pass
    return scores
