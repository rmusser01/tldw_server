from __future__ import annotations

from typing import Any


class ChatWorkflowQuestionRenderer:
    """Minimal question renderer for chat workflow steps."""

    async def render_question(
        self,
        *,
        base_question: str,
        phrasing_instructions: str | None,
        prior_answers: list[dict[str, Any]],
        context_snapshot: list[dict[str, Any]] | list[Any],
        model: str | None = None,
    ) -> dict[str, Any]:
        return {
            "displayed_question": base_question,
            "question_generation_meta": {
                "mode": "fallback",
                "reason": "renderer_unavailable",
                "model": model,
            },
            "fallback_used": True,
        }
