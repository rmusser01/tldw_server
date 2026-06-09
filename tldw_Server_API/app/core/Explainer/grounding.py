"""Grounding validation for Explainer generation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.Explainer.models import ExplainerEvidenceState, ExplainerGrounding
from tldw_Server_API.app.core.Explainer.retrieval import ExplainerSourceContext


@dataclass(frozen=True)
class GroundedExplainerChild:
    title: str
    body: str | None
    kind: str
    intent: str
    evidence_state: str
    outside_knowledge_used: bool
    citations: list[dict[str, Any]] = field(default_factory=list)


def resolve_grounded_children(
    *,
    generation: dict[str, Any] | None,
    grounding: str,
    source_context: ExplainerSourceContext,
    intent: str,
) -> list[GroundedExplainerChild]:
    """Normalize generated children and enforce Explainer grounding rules."""

    if _source_only_is_insufficient(grounding=grounding, source_context=source_context):
        return [_insufficient_child(intent=intent)]

    children = _extract_children(generation)
    if not children:
        return [_insufficient_child(intent=intent)]

    grounded_children = [_coerce_child(child, fallback_intent=intent) for child in children]
    if grounding == ExplainerGrounding.SOURCE_ONLY.value:
        if any(child.outside_knowledge_used or not child.citations for child in grounded_children):
            return [_insufficient_child(intent=intent)]
        return [
            GroundedExplainerChild(
                title=child.title,
                body=child.body,
                kind=child.kind,
                intent=child.intent,
                evidence_state=ExplainerEvidenceState.SUPPORTED.value,
                outside_knowledge_used=False,
                citations=child.citations,
            )
            for child in grounded_children
        ]
    return grounded_children


def _source_only_is_insufficient(
    *,
    grounding: str,
    source_context: ExplainerSourceContext,
) -> bool:
    return (
        grounding == ExplainerGrounding.SOURCE_ONLY.value
        and (source_context.insufficient or not source_context.normalized_excerpts())
    )


def _extract_children(generation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(generation, dict):
        return []
    children = generation.get("children") or generation.get("nodes") or []
    if not isinstance(children, list):
        return []
    return [child for child in children if isinstance(child, dict)]


def _coerce_child(child: dict[str, Any], *, fallback_intent: str) -> GroundedExplainerChild:
    citations = _coerce_citations(child.get("citations") or [])
    outside_knowledge_used = bool(
        child.get("outside_knowledge_used")
        if "outside_knowledge_used" in child
        else child.get("outsideKnowledgeUsed")
    )
    evidence_state = _coerce_evidence_state(
        child.get("evidence_state") or child.get("evidenceState"),
        citations=citations,
        outside_knowledge_used=outside_knowledge_used,
    )
    return GroundedExplainerChild(
        title=_text_or_default(child.get("title"), "Generated explanation"),
        body=str(child.get("body")) if child.get("body") is not None else None,
        kind=_text_or_default(child.get("kind"), "explanation"),
        intent=_text_or_default(child.get("intent"), fallback_intent),
        evidence_state=evidence_state,
        outside_knowledge_used=outside_knowledge_used,
        citations=citations,
    )


def _coerce_citations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    citations: list[dict[str, Any]] = []
    for citation in value:
        if not isinstance(citation, dict):
            continue
        source_id = citation.get("source_id") or citation.get("sourceId")
        source_type = citation.get("source_type") or citation.get("sourceType")
        title = citation.get("title")
        excerpt = citation.get("excerpt")
        if not (source_id and source_type and title and excerpt):
            continue
        citations.append(
            {
                "source_id": str(source_id),
                "source_type": str(source_type),
                "title": str(title),
                "excerpt": str(excerpt),
                "location_label": citation.get("location_label") or citation.get("locationLabel"),
                "start_offset": _get_alias(citation, "start_offset", "startOffset"),
                "end_offset": _get_alias(citation, "end_offset", "endOffset"),
                "url": citation.get("url"),
                "snapshot_hash": citation.get("snapshot_hash") or citation.get("snapshotHash"),
            }
        )
    return citations


def _coerce_evidence_state(
    explicit_value: Any,
    *,
    citations: list[dict[str, Any]],
    outside_knowledge_used: bool,
) -> str:
    allowed = {item.value for item in ExplainerEvidenceState}
    explicit = str(explicit_value or "").strip()
    if explicit in allowed:
        return explicit
    if citations and outside_knowledge_used:
        return ExplainerEvidenceState.PARTIALLY_SUPPORTED.value
    if citations:
        return ExplainerEvidenceState.SUPPORTED.value
    if outside_knowledge_used:
        return ExplainerEvidenceState.UNCITED.value
    return ExplainerEvidenceState.UNCITED.value


def _insufficient_child(*, intent: str) -> GroundedExplainerChild:
    return GroundedExplainerChild(
        title="Insufficient source evidence",
        body=(
            "The selected sources do not provide enough cited evidence for this expansion. "
            "Select more sources, switch grounding mode, or ask a narrower question."
        ),
        kind="summary",
        intent=intent,
        evidence_state=ExplainerEvidenceState.INSUFFICIENT.value,
        outside_knowledge_used=False,
        citations=[],
    )


def _text_or_default(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _get_alias(value: dict[str, Any], snake_key: str, camel_key: str) -> Any:
    return value[snake_key] if snake_key in value else value.get(camel_key)


__all__ = [
    "GroundedExplainerChild",
    "resolve_grounded_children",
]
