"""Pure prompt builders for Explainer generation jobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.Explainer.models import ExplainerNode, ExplainerSession
from tldw_Server_API.app.core.Explainer.retrieval import ExplainerSourceContext

PROMPT_TEMPLATE_VERSION = "explainer_node_expansion_v1"


@dataclass(frozen=True)
class ExplainerPrompt:
    prompt_template_version: str
    system: str
    user: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def build_node_expansion_prompt(
    *,
    session: ExplainerSession,
    node: ExplainerNode,
    source_context: ExplainerSourceContext,
    intent: str,
    grounding: str,
) -> ExplainerPrompt:
    """Build a deterministic node-expansion prompt without side effects."""

    system = (
        "You expand a persisted research explainer tree. Return structured output "
        "with child nodes, citations, outside-knowledge flags, and generation metadata."
    )
    source_block = _format_source_block(source_context)
    user = "\n".join(
        [
            f"Session title: {session.title}",
            f"Depth preset: {session.depth_preset}",
            f"Requested intent: {intent}",
            f"Grounding mode: {grounding}",
            f"Node title: {node.title}",
            f"Node body: {node.body or ''}",
            f"Selected answer: {node.selected_option_id or node.selected_custom_answer or ''}",
            "Source context:",
            source_block,
        ]
    )
    return ExplainerPrompt(
        prompt_template_version=PROMPT_TEMPLATE_VERSION,
        system=system,
        user=user,
        metadata={
            "sessionId": session.id,
            "nodeId": node.id,
            "intent": intent,
            "grounding": grounding,
            "sourceExcerptCount": len(source_context.normalized_excerpts()),
        },
    )


def _format_source_block(source_context: ExplainerSourceContext) -> str:
    excerpts = source_context.normalized_excerpts()
    if not excerpts:
        return "(no selected-source excerpts available)"
    lines: list[str] = []
    for index, excerpt in enumerate(excerpts, start=1):
        location = f" {excerpt.location_label}" if excerpt.location_label else ""
        lines.append(
            f"[{index}] {excerpt.title}{location}\n"
            f"source_id={excerpt.source_id} source_type={excerpt.source_type}\n"
            f"{excerpt.excerpt}"
        )
    return "\n\n".join(lines)


__all__ = [
    "PROMPT_TEMPLATE_VERSION",
    "ExplainerPrompt",
    "build_node_expansion_prompt",
]
