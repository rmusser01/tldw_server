"""Jobs helpers and worker handler for Explainer node expansion."""

from __future__ import annotations

import hashlib
import inspect
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

    answer_revision = _answer_revision(node)
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
    return ExplainerJobAccepted(
        job_id=str(job.get("id")),
        session_id=session.id,
        node_id=node.id,
        status=str(job.get("status") or "queued"),
    )


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
        children = _enforce_selected_citations(session=session, children=children)
        metadata = _build_generation_metadata(
            generation=generation,
            prompt=prompt,
            session=session,
            source_context=source_context,
            job_id=job_id,
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
        raise RuntimeError("Explainer generator is not configured")
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
    child_ids: list[str] = []
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
        updated = repo.update_node(
            session.id,
            created.id,
            owner_user_id=owner_user_id,
            generation_metadata=metadata,
        )
        child_ids.append((updated or created).id)
    return child_ids


def _enforce_selected_citations(
    *,
    session: ExplainerSession,
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
        if session.grounding == ExplainerGrounding.SOURCE_ONLY.value and len(citations) != len(child.citations):
            return [
                GroundedExplainerChild(
                    title="Insufficient source evidence",
                    body=(
                        "The selected sources do not provide enough cited evidence for this expansion. "
                        "Select more sources, switch grounding mode, or ask a narrower question."
                    ),
                    kind="summary",
                    intent=child.intent,
                    evidence_state=ExplainerEvidenceState.INSUFFICIENT.value,
                    outside_knowledge_used=False,
                    citations=[],
                )
            ]
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
) -> dict[str, Any]:
    raw_metadata = (generation or {}).get("generation_metadata") or (generation or {}).get("generationMetadata") or {}
    metadata = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    metadata.setdefault("provider", "unknown")
    metadata.setdefault("model", "unknown")
    metadata["promptTemplateVersion"] = prompt.prompt_template_version
    metadata["grounding"] = session.grounding
    metadata["retrievalSettings"] = dict(source_context.retrieval_metadata or {})
    metadata["jobId"] = job_id
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


def _answer_revision(node: ExplainerNode) -> str:
    raw = "|".join(
        [
            node.updated_at,
            node.selected_option_id or "",
            node.selected_custom_answer or "",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


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
    "ExplainerGenerator",
    "ExplainerJobAccepted",
    "enqueue_explainer_node_expansion_job",
    "handle_explainer_node_expansion_job",
]
