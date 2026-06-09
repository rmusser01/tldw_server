"""Jobs helpers and worker handler for Explainer node expansion."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from tldw_Server_API.app.core.Explainer.grounding import GroundedExplainerChild, resolve_grounded_children
from tldw_Server_API.app.core.Explainer.models import (
    ExplainerEvidenceState,
    ExplainerGrounding,
    ExplainerNode,
    ExplainerNodeStatus,
    ExplainerOutputIntent,
    ExplainerSession,
)
from tldw_Server_API.app.core.Explainer.prompting import ExplainerPrompt, build_node_expansion_prompt
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository
from tldw_Server_API.app.core.Explainer.retrieval import (
    ExplainerRetriever,
    ExplainerSourceContext,
    coerce_source_context,
    retrieve_selected_source_context,
    validate_source_context_ownership,
)

EXPLAINER_DOMAIN = "explainer"
EXPLAINER_JOB_TYPE = "node_expansion"
EXPLAINER_QUEUE = "default"

ExplainerGenerator = Callable[[ExplainerPrompt], Any]
_ENQUEUEABLE_JOB_STATUSES = {"queued", "pending", "scheduled"}


class ExplainerGenerationNotConfiguredError(RuntimeError):
    """Raised when an expansion needs generation but no generator is configured."""


class ExplainerTerminalJobError(ValueError):
    """Raised when Jobs idempotency returns a non-enqueueable job row."""


@dataclass(frozen=True)
class ExplainerJobAccepted:
    job_id: str
    session_id: str
    node_id: str
    status: str


def enqueue_explainer_node_expansion_job(
    *,
    jm: Any,
    session: ExplainerSession,
    node: ExplainerNode,
    owner_user_id: str,
    intent: str,
    priority: int = 5,
) -> ExplainerJobAccepted:
    """Create a Jobs row for an Explainer node expansion."""

    answer_revision = current_answer_revision(node)
    payload = {
        "session_id": session.id,
        "node_id": node.id,
        "intent": intent,
        "answer_revision": answer_revision,
    }
    job = jm.create_job(
        domain=EXPLAINER_DOMAIN,
        queue=EXPLAINER_QUEUE,
        job_type=EXPLAINER_JOB_TYPE,
        payload=payload,
        owner_user_id=str(owner_user_id),
        priority=priority,
        max_retries=2,
        idempotency_key=f"explainer:{session.id}:{node.id}:{intent}:{answer_revision}",
    )
    status = str(job.get("status") or "queued").strip().lower()
    if status not in _ENQUEUEABLE_JOB_STATUSES:
        raise ExplainerTerminalJobError(
            f"Explainer expansion job was not queued; current job status is {status or 'unknown'}"
        )
    return ExplainerJobAccepted(
        job_id=str(job.get("id")),
        session_id=session.id,
        node_id=node.id,
        status=status,
    )


def is_explainer_generation_configured() -> bool:
    """Return True when Explainer generation has been explicitly enabled."""

    return (
        _env_truthy(os.getenv("EXPLAINER_GENERATOR_ENABLED"))
        and bool((os.getenv("EXPLAINER_GENERATOR_PROVIDER") or "").strip())
        and bool((os.getenv("EXPLAINER_GENERATOR_MODEL") or "").strip())
    )


def make_configured_explainer_generator() -> ExplainerGenerator:
    """Build a lazy LLM-adapter backed generator from explicit Explainer env config."""

    if not is_explainer_generation_configured():
        raise ExplainerGenerationNotConfiguredError(
            "Explainer generation is not configured. Set EXPLAINER_GENERATOR_ENABLED=1, "
            "EXPLAINER_GENERATOR_PROVIDER, and EXPLAINER_GENERATOR_MODEL."
        )

    provider = (os.getenv("EXPLAINER_GENERATOR_PROVIDER") or "").strip()
    model = (os.getenv("EXPLAINER_GENERATOR_MODEL") or "").strip()
    temperature = _optional_float(os.getenv("EXPLAINER_GENERATOR_TEMPERATURE"))
    max_tokens = _optional_int(os.getenv("EXPLAINER_GENERATOR_MAX_TOKENS"))
    timeout = _optional_float(os.getenv("EXPLAINER_GENERATOR_TIMEOUT_SECONDS"))

    async def _generator(prompt: ExplainerPrompt) -> dict[str, Any]:
        from tldw_Server_API.app.core.LLM_Calls import adapter_utils

        app_config = adapter_utils.ensure_app_config()
        adapter = adapter_utils.get_adapter_or_raise(provider)
        messages = prompt.as_messages()
        system_message = None
        if hasattr(adapter_utils, "split_system_message"):
            system_message, messages = adapter_utils.split_system_message(messages)
        request: dict[str, Any] = {
            "messages": messages,
            "system_message": system_message,
            "model": model,
            "api_key": adapter_utils.resolve_provider_api_key_from_config(provider, app_config),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "app_config": app_config,
        }
        response = adapter.chat(request, timeout=timeout)
        if inspect.isawaitable(response):
            response = await response
        parsed = _parse_generation_response(response)
        metadata = parsed.get("generation_metadata") or parsed.get("generationMetadata") or {}
        parsed["generation_metadata"] = dict(metadata) if isinstance(metadata, dict) else {}
        parsed["generation_metadata"].setdefault("provider", provider)
        parsed["generation_metadata"].setdefault("model", model)
        return parsed

    return _generator


async def handle_explainer_node_expansion_job(
    job: dict[str, Any],
    *,
    repo: ExplainerRepository,
    generator: ExplainerGenerator | None = None,
    retriever: ExplainerRetriever | None = None,
) -> dict[str, Any]:
    """Run one Explainer node expansion job and persist the generated children."""

    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        raise ValueError("explainer job payload must be an object")
    job_type = str(job.get("job_type") or payload.get("job_type") or EXPLAINER_JOB_TYPE).strip()
    if job_type != EXPLAINER_JOB_TYPE:
        raise ValueError(f"unsupported explainer job type: {job_type}")

    session_id = _required_text(payload.get("session_id"), "session_id")
    node_id = _required_text(payload.get("node_id"), "node_id")
    owner_user_id = _required_text(job.get("owner_user_id") or payload.get("owner_user_id"), "owner_user_id")
    intent = _normalize_intent(str(payload.get("intent") or ""))
    job_id = str(job.get("id") or "")

    session = repo.get_session(session_id, owner_user_id=owner_user_id)
    if session is None or node_id not in session.nodes:
        raise KeyError("Explainer node not found")
    node = session.nodes[node_id]
    effective_intent = intent or node.intent or session.output_intent
    payload_revision = str(payload.get("answer_revision") or "").strip()
    current_revision = current_answer_revision(node)
    batch_key = _expansion_batch_key(
        session_id=session.id,
        node_id=node.id,
        intent=effective_intent,
        answer_revision=payload_revision or current_revision,
    )

    existing_batch_child_ids = _existing_batch_child_ids(session=session, node=node, batch_key=batch_key)
    if existing_batch_child_ids:
        return {
            "session_id": session.id,
            "node_id": node.id,
            "status": "skipped",
            "reason": "duplicate_expansion_batch",
            "children_created": 0,
            "child_node_ids": existing_batch_child_ids,
        }

    if payload_revision and payload_revision != current_revision:
        return {
            "session_id": session.id,
            "node_id": node.id,
            "status": "skipped",
            "reason": "stale_answer_revision",
            "children_created": 0,
            "child_node_ids": [],
        }

    repo.update_node(
        session_id,
        node_id,
        owner_user_id=owner_user_id,
        status=ExplainerNodeStatus.GENERATING.value,
    )

    try:
        source_context = await _load_source_context(
            session=session,
            owner_user_id=owner_user_id,
            retriever=retriever,
        )
        prompt = build_node_expansion_prompt(
            session=session,
            node=node,
            source_context=source_context,
            intent=effective_intent,
            grounding=session.grounding,
        )
        generation: dict[str, Any] | None = None
        if not _source_only_without_evidence(session=session, source_context=source_context):
            generation = await _call_generator(generator, prompt)
        children = resolve_grounded_children(
            generation=generation,
            grounding=session.grounding,
            source_context=source_context,
            intent=effective_intent,
        )
        children = _enforce_selected_citations(
            session=session,
            source_context=source_context,
            children=children,
        )
        metadata = _build_generation_metadata(
            generation=generation,
            prompt=prompt,
            session=session,
            source_context=source_context,
            job_id=job_id,
            batch_key=batch_key,
            answer_revision=current_revision,
        )
        created_child_ids = _persist_children(
            repo=repo,
            session=session,
            node=node,
            owner_user_id=owner_user_id,
            children=children,
            metadata=metadata,
        )
        repo.update_node(
            session.id,
            node.id,
            owner_user_id=owner_user_id,
            status=ExplainerNodeStatus.COMPLETE.value,
            generation_metadata=metadata,
        )
        return {
            "session_id": session.id,
            "node_id": node.id,
            "children_created": len(created_child_ids),
            "child_node_ids": created_child_ids,
        }
    except Exception:
        repo.update_node(
            session_id,
            node_id,
            owner_user_id=owner_user_id,
            status=ExplainerNodeStatus.ERROR.value,
            generation_metadata=_build_error_metadata(
                job_id=job_id,
                grounding=session.grounding,
            ),
        )
        raise


async def _load_source_context(
    *,
    session: ExplainerSession,
    owner_user_id: str,
    retriever: ExplainerRetriever | None,
) -> ExplainerSourceContext:
    if retriever is None:
        return retrieve_selected_source_context(session=session, owner_user_id=owner_user_id)
    raw_context = retriever(session=session, owner_user_id=owner_user_id)
    if inspect.isawaitable(raw_context):
        raw_context = await raw_context
    return validate_source_context_ownership(
        session=session,
        owner_user_id=owner_user_id,
        source_context=coerce_source_context(raw_context),
    )


async def _call_generator(
    generator: ExplainerGenerator | None,
    prompt: ExplainerPrompt,
) -> dict[str, Any]:
    if generator is None:
        raise ExplainerGenerationNotConfiguredError("Explainer generator is not configured")
    result = generator(prompt)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, dict):
        raise TypeError("Explainer generator must return a dict")
    return result


def _persist_children(
    *,
    repo: ExplainerRepository,
    session: ExplainerSession,
    node: ExplainerNode,
    owner_user_id: str,
    children: list[GroundedExplainerChild],
    metadata: dict[str, Any],
) -> list[str]:
    created_child_ids: list[str] = []
    persisted_child_ids: list[str] = []
    try:
        for child in children:
            created = repo.create_node(
                session.id,
                owner_user_id=owner_user_id,
                parent_id=node.id,
                title=child.title,
                body=child.body,
                kind=child.kind,
                intent=child.intent,
                status=ExplainerNodeStatus.COMPLETE.value,
                evidence_state=child.evidence_state,
                outside_knowledge_used=child.outside_knowledge_used,
                citations=child.citations,
            )
            if created is None:
                raise KeyError("Explainer session not found while writing child node")
            created_child_ids.append(created.id)
            updated = repo.update_node(
                session.id,
                created.id,
                owner_user_id=owner_user_id,
                generation_metadata=metadata,
            )
            if updated is None:
                raise KeyError("Explainer child not found while writing generation metadata")
            persisted_child_ids.append(updated.id)
        return persisted_child_ids
    except Exception:
        for child_id in reversed(created_child_ids):
            try:
                repo.delete_node(session.id, child_id, owner_user_id=owner_user_id)
            except Exception as rollback_error:
                _ = rollback_error
        raise


def _enforce_selected_citations(
    *,
    session: ExplainerSession,
    source_context: ExplainerSourceContext,
    children: list[GroundedExplainerChild],
) -> list[GroundedExplainerChild]:
    selected = {
        (source.source_type, source.source_id)
        for source in session.selected_sources
    }
    if not selected:
        return [
            GroundedExplainerChild(
                title=child.title,
                body=child.body,
                kind=child.kind,
                intent=child.intent,
                evidence_state=(
                    ExplainerEvidenceState.UNCITED.value
                    if child.citations
                    else child.evidence_state
                ),
                outside_knowledge_used=child.outside_knowledge_used,
                citations=[],
            )
            for child in children
        ]
    filtered_children: list[GroundedExplainerChild] = []
    for child in children:
        citations = [
            citation
            for citation in child.citations
            if (str(citation.get("source_type")), str(citation.get("source_id"))) in selected
        ]
        if session.grounding == ExplainerGrounding.SOURCE_ONLY.value:
            if len(citations) != len(child.citations) or not _citations_match_authoritative_context(
                citations=citations,
                source_context=source_context,
            ):
                return [_insufficient_source_child(intent=child.intent)]
        evidence_state = child.evidence_state
        if child.citations and not citations:
            evidence_state = ExplainerEvidenceState.UNCITED.value
        filtered_children.append(
            GroundedExplainerChild(
                title=child.title,
                body=child.body,
                kind=child.kind,
                intent=child.intent,
                evidence_state=evidence_state,
                outside_knowledge_used=child.outside_knowledge_used,
                citations=citations,
            )
        )
    return filtered_children


def _build_generation_metadata(
    *,
    generation: dict[str, Any] | None,
    prompt: ExplainerPrompt,
    session: ExplainerSession,
    source_context: ExplainerSourceContext,
    job_id: str,
    batch_key: str,
    answer_revision: str,
) -> dict[str, Any]:
    raw_metadata = (generation or {}).get("generation_metadata") or (generation or {}).get("generationMetadata") or {}
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata.setdefault("provider", "unknown")
    metadata.setdefault("model", "unknown")
    metadata["promptTemplateVersion"] = prompt.prompt_template_version
    metadata["grounding"] = session.grounding
    metadata["retrievalSettings"] = dict(source_context.retrieval_metadata or {})
    metadata["jobId"] = job_id
    metadata["answerRevision"] = answer_revision
    metadata["explainerExpansionBatchKey"] = batch_key
    metadata["generatedAt"] = _utc_now()
    return metadata


def _build_error_metadata(*, job_id: str, grounding: str) -> dict[str, Any]:
    return {
        "jobId": job_id,
        "grounding": grounding,
        "error": "generation_failed",
        "generatedAt": _utc_now(),
    }


def _source_only_without_evidence(
    *,
    session: ExplainerSession,
    source_context: ExplainerSourceContext,
) -> bool:
    return (
        session.grounding == ExplainerGrounding.SOURCE_ONLY.value
        and (source_context.insufficient or not source_context.normalized_excerpts())
    )


def current_answer_revision(node: ExplainerNode) -> str:
    raw = "|".join(
        [
            node.updated_at,
            node.selected_option_id or "",
            node.selected_custom_answer or "",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _expansion_batch_key(
    *,
    session_id: str,
    node_id: str,
    intent: str,
    answer_revision: str,
) -> str:
    return f"explainer:{session_id}:{node_id}:{intent}:{answer_revision}"


def _existing_batch_child_ids(
    *,
    session: ExplainerSession,
    node: ExplainerNode,
    batch_key: str,
) -> list[str]:
    child_ids: list[str] = []
    for child_id in node.child_node_ids:
        child = session.nodes.get(child_id)
        metadata = child.generation_metadata if child is not None else None
        if isinstance(metadata, dict) and metadata.get("explainerExpansionBatchKey") == batch_key:
            child_ids.append(child_id)
    return child_ids


def _citations_match_authoritative_context(
    *,
    citations: list[dict[str, Any]],
    source_context: ExplainerSourceContext,
) -> bool:
    authoritative_excerpts = source_context.normalized_excerpts()
    if not citations or not authoritative_excerpts:
        return False
    return all(
        any(_citation_matches_excerpt(citation, excerpt.to_citation_payload()) for excerpt in authoritative_excerpts)
        for citation in citations
    )


def _citation_matches_excerpt(citation: dict[str, Any], excerpt: dict[str, Any]) -> bool:
    if str(citation.get("source_id")) != str(excerpt.get("source_id")):
        return False
    if str(citation.get("source_type")) != str(excerpt.get("source_type")):
        return False
    if _normalize_text(citation.get("excerpt")) != _normalize_text(excerpt.get("excerpt")):
        return False
    citation_hash = citation.get("snapshot_hash")
    excerpt_hash = excerpt.get("snapshot_hash")
    if excerpt_hash is not None and str(citation_hash or "") != str(excerpt_hash):
        return False
    for field_name in ("start_offset", "end_offset"):
        citation_offset = citation.get(field_name)
        excerpt_offset = excerpt.get(field_name)
        if excerpt_offset is not None and (
            citation_offset is None or int(citation_offset) != int(excerpt_offset)
        ):
            return False
    citation_location = citation.get("location_label")
    excerpt_location = excerpt.get("location_label")
    if citation_location and excerpt_location and str(citation_location) != str(excerpt_location):
        return False
    return True


def _insufficient_source_child(*, intent: str) -> GroundedExplainerChild:
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


def _parse_generation_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict) and isinstance(response.get("children"), list):
        return response
    content = _extract_response_content(response)
    if not content:
        raise ValueError("Explainer generator response did not contain content")
    parsed = json.loads(_extract_json_object(content))
    if not isinstance(parsed, dict):
        raise TypeError("Explainer generator response JSON must be an object")
    return parsed


def _extract_response_content(response: Any) -> str:
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        return str(response or "")
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and message.get("content") is not None:
                return str(message.get("content") or "")
            if first_choice.get("text") is not None:
                return str(first_choice.get("text") or "")
    for key in ("content", "text", "response", "output"):
        if response.get(key) is not None:
            return str(response.get(key) or "")
    return ""


def _extract_json_object(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Explainer generator response did not contain a JSON object")
    return text[start : end + 1]


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(value: str | None) -> float | None:
    text = str(value or "").strip()
    return float(text) if text else None


def _optional_int(value: str | None) -> int | None:
    text = str(value or "").strip()
    return int(text) if text else None


def _normalize_intent(value: str) -> str:
    if not value:
        return ""
    allowed = {item.value for item in ExplainerOutputIntent}
    if value not in allowed:
        raise ValueError(f"intent must be one of {sorted(allowed)}")
    return value


def _required_text(value: Any, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is required")
    return text


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "EXPLAINER_DOMAIN",
    "EXPLAINER_JOB_TYPE",
    "EXPLAINER_QUEUE",
    "ExplainerGenerationNotConfiguredError",
    "ExplainerGenerator",
    "ExplainerJobAccepted",
    "ExplainerTerminalJobError",
    "current_answer_revision",
    "enqueue_explainer_node_expansion_job",
    "handle_explainer_node_expansion_job",
    "is_explainer_generation_configured",
    "make_configured_explainer_generator",
]
