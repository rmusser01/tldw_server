"""Chatbook serialization and restoration helpers for Explainer sessions."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Explainer.models import (
    ExplainerCitation,
    ExplainerDepthPreset,
    ExplainerEvidenceState,
    ExplainerGrounding,
    ExplainerMode,
    ExplainerNode,
    ExplainerNodeKind,
    ExplainerNodeStatus,
    ExplainerOutputIntent,
    ExplainerSelectedSource,
    ExplainerSession,
)
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository

EXPLAINER_CHATBOOK_FORMAT = "tldw.explainer_session.v1"
EXPLAINER_CHATBOOK_TYPE = "explainer_session"

_SENSITIVE_METADATA_KEY_PARTS = (
    "api_key",
    "api_token",
    "apikey",
    "access_token",
    "authorization",
    "auth_token",
    "bearer",
    "credential",
    "password",
    "raw_prompt",
    "refresh_token",
    "secret",
    "session_token",
    "system_prompt",
)


def build_explainer_chatbook_payload(
    *,
    repo: ExplainerRepository,
    session_id: str,
    owner_user_id: str,
    exported_at: str | None = None,
) -> dict[str, Any]:
    """Serialize an ownership-checked Explainer session as one Chatbook content item."""
    session = repo.get_session(session_id, owner_user_id=owner_user_id)
    if session is None:
        raise LookupError("Explainer session not found")

    ordered_nodes = _ordered_nodes(session)
    structured_nodes = [_node_payload(node) for node in ordered_nodes]
    structured_citations: list[dict[str, Any]] = []
    for node in ordered_nodes:
        structured_citations.extend(
            _citation_payload(citation, node_id=node.id)
            for citation in node.citations
        )

    exported_at_value = exported_at or _utcnow_iso()
    payload = {
        "format": EXPLAINER_CHATBOOK_FORMAT,
        "type": EXPLAINER_CHATBOOK_TYPE,
        "structured": {
            "session": _session_payload(session),
            "selectedSources": [
                _selected_source_payload(source)
                for source in session.selected_sources
            ],
            "nodes": structured_nodes,
            "citations": structured_citations,
        },
        "rendered": {
            "markdown": _render_markdown(session=session, ordered_nodes=ordered_nodes),
        },
        "metadata": {
            "schemaVersion": 1,
            "exportedAt": exported_at_value,
            "sourceBundling": "references_only",
        },
    }
    return payload


def restore_explainer_chatbook_payload(
    *,
    repo: ExplainerRepository,
    payload: dict[str, Any],
    owner_user_id: str,
    prefix_imported: bool = False,
) -> ExplainerSession:
    """Restore an Explainer Chatbook payload for the importing user."""
    explainer_payload = _unwrap_explainer_payload(payload)
    structured = explainer_payload.get("structured")
    if not isinstance(structured, dict):
        raise ValueError("Explainer payload missing structured content")

    session_payload = structured.get("session")
    if not isinstance(session_payload, dict):
        raise ValueError("Explainer payload missing session")

    nodes_payload = structured.get("nodes")
    if not isinstance(nodes_payload, list) or not nodes_payload:
        raise ValueError("Explainer payload missing nodes")

    title = _text(session_payload.get("title"), "Imported Explainer session")
    if prefix_imported:
        title = f"[Imported] {title}"
    settings = session_payload.get("settings") if isinstance(session_payload.get("settings"), dict) else {}
    selected_sources = [
        _restore_selected_source(source)
        for source in structured.get("selectedSources") or []
        if isinstance(source, dict)
    ]

    first_root_payload = _first_root_payload(nodes_payload)
    root_prompt = _text(first_root_payload.get("title"), title)
    restored = repo.create_session(
        owner_user_id=owner_user_id,
        title=title,
        mode=_enum_value(settings.get("mode") or session_payload.get("mode"), ExplainerMode, ExplainerMode.GOAL.value),
        output_intent=_enum_value(
            settings.get("outputIntent") or session_payload.get("outputIntent"),
            ExplainerOutputIntent,
            ExplainerOutputIntent.EXPLAIN.value,
        ),
        grounding=_enum_value(
            settings.get("grounding") or session_payload.get("grounding"),
            ExplainerGrounding,
            ExplainerGrounding.OPEN.value,
        ),
        depth_preset=_enum_value(
            settings.get("depthPreset") or session_payload.get("depthPreset"),
            ExplainerDepthPreset,
            ExplainerDepthPreset.STANDARD.value,
        ),
        selected_sources=selected_sources,
        root_prompt=root_prompt,
    )

    try:
        _restore_session_nodes(
            repo=repo,
            restored=restored,
            owner_user_id=owner_user_id,
            nodes_payload=nodes_payload,
            first_root_payload=first_root_payload,
        )
        loaded = repo.get_session(restored.id, owner_user_id=owner_user_id)
        if loaded is None:
            raise ValueError("Restored Explainer session could not be reloaded")
        return loaded
    except Exception:
        # A failed restore must not leave a partially imported session behind.
        try:
            repo.delete_session(restored.id, owner_user_id=owner_user_id)
        except Exception:
            logger.warning(
                "Failed to clean up partially restored Explainer session {}", restored.id
            )
        raise


def _restore_session_nodes(
    *,
    repo: ExplainerRepository,
    restored: ExplainerSession,
    owner_user_id: str,
    nodes_payload: list[Any],
    first_root_payload: dict[str, Any],
) -> None:
    original_to_new: dict[str, str] = {}
    root_node_id = restored.root_node_ids[0]
    original_root_id = _text(first_root_payload.get("id"), "")
    if original_root_id:
        original_to_new[original_root_id] = root_node_id
    _apply_node_payload(
        repo=repo,
        session_id=restored.id,
        owner_user_id=owner_user_id,
        node_id=root_node_id,
        node_payload=first_root_payload,
    )

    remaining = [
        node for node in nodes_payload
        if isinstance(node, dict) and node is not first_root_payload
    ]
    while remaining:
        progressed = False
        next_remaining: list[dict[str, Any]] = []
        for node_payload in remaining:
            original_id = _text(node_payload.get("id"), "")
            original_parent_id = node_payload.get("parentId")
            if original_parent_id in ("", None):
                parent_id = None
            elif str(original_parent_id) in original_to_new:
                parent_id = original_to_new[str(original_parent_id)]
            else:
                next_remaining.append(node_payload)
                continue

            created = repo.create_node(
                restored.id,
                owner_user_id=owner_user_id,
                parent_id=parent_id,
                title=_text(node_payload.get("title"), "Untitled"),
                body=node_payload.get("body") if isinstance(node_payload.get("body"), str) else None,
                kind=_enum_value(node_payload.get("kind"), ExplainerNodeKind, ExplainerNodeKind.EXPLANATION.value),
                intent=_enum_value(node_payload.get("intent"), ExplainerOutputIntent, restored.output_intent),
                status=_enum_value(node_payload.get("status"), ExplainerNodeStatus, ExplainerNodeStatus.IDLE.value),
                evidence_state=_enum_value(
                    node_payload.get("evidenceState"),
                    ExplainerEvidenceState,
                    ExplainerEvidenceState.UNCITED.value,
                ),
                outside_knowledge_used=bool(node_payload.get("outsideKnowledgeUsed")),
                citations=[],
            )
            if created is None:
                raise ValueError("Failed to restore Explainer node")
            if original_id:
                original_to_new[original_id] = created.id
            _apply_node_payload(
                repo=repo,
                session_id=restored.id,
                owner_user_id=owner_user_id,
                node_id=created.id,
                node_payload=node_payload,
            )
            progressed = True
        if not progressed:
            raise ValueError("Explainer payload node parent references could not be resolved")
        remaining = next_remaining


def _unwrap_explainer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    payload_type = str(payload.get("type") or "").strip()
    if payload_type == EXPLAINER_CHATBOOK_TYPE:
        return payload

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if payload_type == "generated_document" and metadata.get("subtype") == EXPLAINER_CHATBOOK_TYPE:
        content = (
            payload.get("content")
            if "content" in payload
            else payload.get("payload")
        )
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as exc:
                raise ValueError("Generated document Explainer payload is not valid JSON") from exc
        if isinstance(content, dict):
            if str(content.get("type") or "") == EXPLAINER_CHATBOOK_TYPE:
                return content
            if "structured" in content:
                fallback = copy.deepcopy(content)
                fallback.setdefault("type", EXPLAINER_CHATBOOK_TYPE)
                fallback.setdefault("format", EXPLAINER_CHATBOOK_FORMAT)
                return fallback
        if "structured" in payload:
            fallback = copy.deepcopy(payload)
            fallback["type"] = EXPLAINER_CHATBOOK_TYPE
            fallback.setdefault("format", EXPLAINER_CHATBOOK_FORMAT)
            return fallback

    raise ValueError("Payload is not an Explainer session")


def _session_payload(session: ExplainerSession) -> dict[str, Any]:
    return {
        "id": session.id,
        "title": session.title,
        "status": session.status,
        "settings": {
            "mode": session.mode,
            "outputIntent": session.output_intent,
            "grounding": session.grounding,
            "depthPreset": session.depth_preset,
        },
        "rootNodeIds": list(session.root_node_ids),
        "createdAt": session.created_at,
        "updatedAt": session.updated_at,
        "archivedAt": session.archived_at,
    }


def _selected_source_payload(source: ExplainerSelectedSource) -> dict[str, Any]:
    return {
        "sourceId": source.source_id,
        "sourceType": source.source_type,
        "title": source.title,
        "addedAt": source.added_at,
        "snapshotVersion": source.snapshot_version,
        "metadata": _sanitize_metadata(source.metadata),
    }


def _node_payload(node: ExplainerNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "parentId": node.parent_id,
        "ordinal": node.ordinal,
        "title": node.title,
        "body": node.body,
        "kind": node.kind,
        "intent": node.intent,
        "status": node.status,
        "evidenceState": node.evidence_state,
        "outsideKnowledgeUsed": node.outside_knowledge_used,
        "questionOptions": copy.deepcopy(node.question_options),
        "selectedOptionId": node.selected_option_id,
        "selectedCustomAnswer": node.selected_custom_answer,
        "generationMetadata": _sanitize_metadata(node.generation_metadata),
        "childNodeIds": list(node.child_node_ids),
        "citations": [_citation_payload(citation) for citation in node.citations],
        "createdAt": node.created_at,
        "updatedAt": node.updated_at,
    }


def _citation_payload(citation: ExplainerCitation, *, node_id: str | None = None) -> dict[str, Any]:
    payload = {
        "id": citation.id,
        "sourceId": citation.source_id,
        "sourceType": citation.source_type,
        "title": citation.title,
        "excerpt": citation.excerpt,
        "locationLabel": citation.location_label,
        "startOffset": citation.start_offset,
        "endOffset": citation.end_offset,
        "url": citation.url,
        "snapshotHash": citation.snapshot_hash,
    }
    if node_id is not None:
        payload["nodeId"] = node_id
    return payload


def _ordered_nodes(session: ExplainerSession) -> list[ExplainerNode]:
    ordered: list[ExplainerNode] = []
    seen: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in seen or node_id not in session.nodes:
            return
        seen.add(node_id)
        node = session.nodes[node_id]
        ordered.append(node)
        children = [
            session.nodes[child_id]
            for child_id in node.child_node_ids
            if child_id in session.nodes
        ]
        for child in sorted(children, key=lambda item: (item.ordinal, item.created_at, item.id)):
            visit(child.id)

    roots = [
        session.nodes[root_id]
        for root_id in session.root_node_ids
        if root_id in session.nodes
    ]
    for root in sorted(roots, key=lambda item: (item.ordinal, item.created_at, item.id)):
        visit(root.id)

    remaining = [
        node for node_id, node in session.nodes.items()
        if node_id not in seen
    ]
    ordered.extend(sorted(remaining, key=lambda item: (item.parent_id or "", item.ordinal, item.created_at, item.id)))
    return ordered


def _render_markdown(*, session: ExplainerSession, ordered_nodes: list[ExplainerNode]) -> str:
    lines = [
        f"# {session.title}",
        "",
        f"- Mode: {session.mode}",
        f"- Output intent: {session.output_intent}",
        f"- Grounding: {session.grounding}",
        f"- Depth: {session.depth_preset}",
        "",
    ]
    if session.selected_sources:
        lines.extend(["## Selected Sources", ""])
        for source in session.selected_sources:
            lines.append(f"- {source.title} ({source.source_type}:{source.source_id})")
        lines.append("")

    depths = _node_depths(ordered_nodes)
    for node in ordered_nodes:
        heading_level = min(6, 2 + depths.get(node.id, 0))
        lines.extend([
            f"{'#' * heading_level} {node.title}",
            "",
        ])
        if node.body:
            lines.extend([node.body, ""])
        if node.question_options:
            lines.append("Question options:")
            for option in node.question_options:
                if isinstance(option, dict):
                    label = option.get("label") or option.get("text") or option.get("id")
                    option_id = option.get("id")
                    selected = " (selected)" if option_id and option_id == node.selected_option_id else ""
                    lines.append(f"- {label}{selected}")
            if node.selected_custom_answer:
                lines.append(f"- Custom answer: {node.selected_custom_answer}")
            lines.append("")
        if node.citations:
            lines.append("Citations:")
            for citation in node.citations:
                location = f", {citation.location_label}" if citation.location_label else ""
                lines.append(f"- {citation.title}{location}: {citation.excerpt}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _node_depths(nodes: list[ExplainerNode]) -> dict[str, int]:
    by_id = {node.id: node for node in nodes}
    depths: dict[str, int] = {}

    def depth(node: ExplainerNode) -> int:
        if node.id in depths:
            return depths[node.id]
        if not node.parent_id or node.parent_id not in by_id:
            depths[node.id] = 0
            return 0
        depths[node.id] = depth(by_id[node.parent_id]) + 1
        return depths[node.id]

    for node in nodes:
        depth(node)
    return depths


def _sanitize_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, nested in value.items():
            key_text = str(key)
            normalized = key_text.lower().replace("-", "_")
            if any(part in normalized for part in _SENSITIVE_METADATA_KEY_PARTS):
                continue
            sanitized[key_text] = _sanitize_metadata(nested)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_metadata(item) for item in value]
    return copy.deepcopy(value)


def _restore_selected_source(source: dict[str, Any]) -> dict[str, Any]:
    metadata = _sanitize_metadata(source.get("metadata"))
    if not isinstance(metadata, dict):
        metadata = {}
    source_id = _text(source.get("sourceId") or source.get("source_id"), "")
    source_type = _text(source.get("sourceType") or source.get("source_type"), "")
    metadata.update(
        {
            "originalSourceId": source_id,
            "originalSourceType": source_type,
            "resolutionStatus": "unresolved",
        }
    )
    return {
        "source_id": source_id or "unknown",
        "source_type": source_type or "unknown",
        "title": _text(source.get("title"), "Untitled source"),
        "added_at": _text(source.get("addedAt") or source.get("added_at"), _utcnow_iso()),
        "snapshot_version": source.get("snapshotVersion") or source.get("snapshot_version"),
        "metadata": metadata,
    }


def _restore_citations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    citations: list[dict[str, Any]] = []
    for citation in value:
        if not isinstance(citation, dict):
            continue
        citations.append(
            {
                "id": "",
                "source_id": _text(citation.get("sourceId") or citation.get("source_id"), "unknown"),
                "source_type": _text(citation.get("sourceType") or citation.get("source_type"), "unknown"),
                "title": _text(citation.get("title"), "Untitled source"),
                "excerpt": _text(citation.get("excerpt"), ""),
                "location_label": citation.get("locationLabel") or citation.get("location_label"),
                "start_offset": _first_present(citation, "startOffset", "start_offset"),
                "end_offset": _first_present(citation, "endOffset", "end_offset"),
                "url": citation.get("url"),
                "snapshot_hash": citation.get("snapshotHash") or citation.get("snapshot_hash"),
            }
        )
    return citations


def _apply_node_payload(
    *,
    repo: ExplainerRepository,
    session_id: str,
    owner_user_id: str,
    node_id: str,
    node_payload: dict[str, Any],
) -> None:
    generation_metadata = _sanitize_metadata(node_payload.get("generationMetadata"))
    if not isinstance(generation_metadata, dict):
        generation_metadata = {}
    generation_metadata["import"] = {
        "originalNodeId": _text(node_payload.get("id"), ""),
        "sourceFormat": EXPLAINER_CHATBOOK_FORMAT,
    }
    updated = repo.update_node(
        session_id,
        node_id,
        owner_user_id=owner_user_id,
        title=_text(node_payload.get("title"), "Untitled"),
        body=node_payload.get("body") if isinstance(node_payload.get("body"), str) else None,
        kind=_enum_value(node_payload.get("kind"), ExplainerNodeKind, ExplainerNodeKind.EXPLANATION.value),
        intent=_enum_value(node_payload.get("intent"), ExplainerOutputIntent, ExplainerOutputIntent.EXPLAIN.value),
        status=_enum_value(node_payload.get("status"), ExplainerNodeStatus, ExplainerNodeStatus.IDLE.value),
        evidence_state=_enum_value(
            node_payload.get("evidenceState"),
            ExplainerEvidenceState,
            ExplainerEvidenceState.UNCITED.value,
        ),
        outside_knowledge_used=bool(node_payload.get("outsideKnowledgeUsed")),
        selected_option_id=node_payload.get("selectedOptionId"),
        selected_custom_answer=node_payload.get("selectedCustomAnswer"),
        question_options=copy.deepcopy(node_payload.get("questionOptions")),
        generation_metadata=generation_metadata,
        citations=_restore_citations(node_payload.get("citations")),
    )
    if updated is None:
        raise ValueError("Failed to update restored Explainer node")


def _first_root_payload(nodes_payload: list[Any]) -> dict[str, Any]:
    dict_nodes = [node for node in nodes_payload if isinstance(node, dict)]
    for node in dict_nodes:
        if node.get("parentId") in (None, ""):
            return node
    return dict_nodes[0]


def _enum_value(value: Any, enum_cls: type, default: str) -> str:
    text = str(value or "").strip()
    allowed = {item.value for item in enum_cls}
    return text if text in allowed else default


def _text(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value
    return default


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
