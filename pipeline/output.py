"""Layer 6 — Output formatting and display."""

from __future__ import annotations

from pipeline.state import PipelineState


def format_output(state: PipelineState) -> dict:
    """Return the clean answer only.

    Sources and Ragas scores are already returned as structured fields
    in the API response — no need to embed them in the answer text.
    """
    return {"answer": state.answer}
