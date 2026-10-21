"""Shared runtime helpers for persona exemplar retrieval and prompt metadata."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.Persona.exemplar_prompt_assembly import (
    PersonaExemplarPromptAssembly,
    assemble_persona_exemplar_prompt,
)
from tldw_Server_API.app.core.Persona.exemplar_turn_classifier import PersonaTurnClassification


@dataclass(frozen=True)
class PersonaExemplarRuntimeContext:
    """Resolved exemplar prompt assembly plus compact selection metadata for one turn."""

    assembly: PersonaExemplarPromptAssembly
    selection_metadata: dict[str, Any]


def append_persona_exemplar_sections(base_text: str, sections: list[tuple[str, str, int]] | None) -> str:
    """Append non-empty exemplar prompt sections to base text with blank-line separation."""

    contents = [str(base_text or "").strip()]
    contents.extend(str(content or "").strip() for _, content, _ in list(sections or []))
    return "\n\n".join(content for content in contents if content)


def build_persona_rag_why_text(
    *,
    has_state_context: bool,
    has_memory_context: bool,
    has_exemplar_guidance: bool,
) -> str:
    """Build a readable explanation of which persona guidance sources shaped a RAG query."""

    applied_contexts: list[str] = []
    if has_state_context:
        applied_contexts.append("persistent persona state")
    if has_memory_context:
        applied_contexts.append("personalization memories")
    if has_exemplar_guidance:
        applied_contexts.append("persona exemplar guidance")
    if not applied_contexts:
        return "Input appears to be a knowledge query."
    if len(applied_contexts) == 1:
        suffix = applied_contexts[0]
    elif len(applied_contexts) == 2:
        suffix = " and ".join(applied_contexts)
    else:
        suffix = f"{', '.join(applied_contexts[:-1])}, and {applied_contexts[-1]}"
    return f"Input appears to be a knowledge query with applied {suffix}."


def build_persona_exemplar_selection_metadata(
    *,
    assembly: PersonaExemplarPromptAssembly,
    classifier: PersonaTurnClassification,
    error_reason: str | None = None,
) -> dict[str, Any]:
    """Serialize compact exemplar selection metadata for persisted persona turns."""

    selected_ids = [
        str(item.get("id") or "")
        for item in list(assembly.selected_exemplars or [])
        if str(item.get("id") or "").strip()
    ]
    rejected = [
        {
            "id": str(item.get("id") or ""),
            "reason": str(item.get("reason") or ""),
        }
        for item in list(assembly.rejected_exemplars or [])
        if str(item.get("id") or "").strip() or str(item.get("reason") or "").strip()
    ]
    return {
        "applied": bool(assembly.sections),
        "selected_ids": selected_ids,
        "selected_count": len(selected_ids),
        "rejected": rejected,
        "rejected_count": len(rejected),
        "error_reason": str(error_reason or "").strip() or None,
        "classifier": {
            "scenario_tags": list(getattr(classifier, "scenario_tags", []) or []),
            "tone": str(getattr(classifier, "tone", "neutral") or "neutral"),
            "risk_tags": list(getattr(classifier, "risk_tags", []) or []),
            "capability_tags": [],
        },
    }


async def resolve_persona_exemplar_runtime_context(
    *,
    persona_scope_db: Any | None,
    user_id: str,
    persona_id: str,
    classifier: PersonaTurnClassification,
    current_turn_text: str,
    lookup_limit: int = 50,
) -> PersonaExemplarRuntimeContext:
    """Load, assemble, and summarize persona exemplar guidance for one live turn."""

    persona_exemplars: list[dict[str, Any]] = []
    error_reason: str | None = None

    if persona_scope_db is not None:
        try:
            persona_exemplars = await asyncio.to_thread(
                # Join the caller's ChaCha operation, or own this read outside one (a WebSocket turn).
                chacha_operation()(persona_scope_db.list_persona_exemplars),
                user_id=user_id,
                persona_id=persona_id,
                include_disabled=False,
                include_deleted=False,
                limit=lookup_limit,
                offset=0,
            )
        except Exception as exc:
            error_reason = "lookup_failed"
            logger.warning(
                "Persona websocket exemplar lookup failed for persona {} user {}: {}",
                persona_id,
                user_id,
                exc,
            )

    assembly = PersonaExemplarPromptAssembly()
    if error_reason is None:
        try:
            assembly = assemble_persona_exemplar_prompt(
                persona_id=persona_id,
                exemplars=persona_exemplars,
                requested_scenario_tags=list(classifier.scenario_tags),
                requested_tone=classifier.tone,
                current_turn_text=current_turn_text,
            )
        except Exception as exc:
            error_reason = "assembly_failed"
            logger.warning(
                "Persona websocket exemplar assembly failed for persona {} user {}: {}",
                persona_id,
                user_id,
                exc,
            )

    return PersonaExemplarRuntimeContext(
        assembly=assembly,
        selection_metadata=build_persona_exemplar_selection_metadata(
            assembly=assembly,
            classifier=classifier,
            error_reason=error_reason,
        ),
    )


__all__ = [
    "PersonaExemplarRuntimeContext",
    "append_persona_exemplar_sections",
    "build_persona_exemplar_selection_metadata",
    "build_persona_rag_why_text",
    "resolve_persona_exemplar_runtime_context",
]
