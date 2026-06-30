"""Portable trace serialization for persona dialogue-tree diagnostics."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from tldw_Server_API.app.core.Persona.dialogue_tree import DialogueTreeNode, DialogueTreeResult
from tldw_Server_API.app.core.Persona.dialogue_tree_context import redact_sensitive_payload


_TRACE_RAW_OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "body",
        "content",
        "headers",
        "output",
        "raw_output",
        "raw_response",
        "raw_result",
        "response",
        "result",
        "signature",
    }
)
_REDACTED_PLACEHOLDER = "[REDACTED]"


def serialize_dialogue_tree_trace(
    tree_result: DialogueTreeResult,
    *,
    root: Mapping[str, Any] | None = None,
    prune_diagnostics: Mapping[str, list[Any]] | None = None,
    score_diagnostics: Mapping[str, list[Any]] | None = None,
    trajectory_scores: list[Any] | None = None,
    selected_node_id: str | None = None,
    fallback_node_id: str | None = None,
    decision_label: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sorted_nodes = sorted(tree_result.nodes, key=lambda node: _node_sort_key(node.node_id))
    node_payloads = [
        _serialize_node(
            node,
            prune_diagnostics=(prune_diagnostics or {}).get(node.node_id, []),
            score_diagnostics=(score_diagnostics or {}).get(node.node_id, []),
        )
        for node in sorted_nodes
    ]

    children_by_parent = {
        parent: sorted(child_ids, key=_node_sort_key)
        for parent, child_ids in sorted(tree_result.children_by_parent.items(), key=lambda item: _node_sort_key(item[0]))
    }

    trace = {
        "root": _to_portable_value(dict(root or {})),
        "nodes": node_payloads,
        "edges": [_serialize_edge(node) for node in sorted_nodes if node.parent_node_id is not None],
        "children_by_parent": children_by_parent,
        "max_depth_seen": tree_result.max_depth_seen,
        "trajectory_scores": [_to_portable_value(score) for score in trajectory_scores or []],
        "decision": {
            "selected_node_id": selected_node_id,
            "fallback_node_id": fallback_node_id,
            "decision_label": decision_label,
        },
        "metadata": _portable_mapping(metadata or {}),
    }
    return redact_sensitive_payload(_redact_trace_raw_output_fields(trace))


def _redact_trace_raw_output_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, sub_value in value.items():
            normalized_key = str(key).casefold()
            if normalized_key in _TRACE_RAW_OUTPUT_KEYS:
                redacted[str(key)] = _REDACTED_PLACEHOLDER
            else:
                redacted[str(key)] = _redact_trace_raw_output_fields(sub_value)
        return redacted

    if isinstance(value, list):
        return [_redact_trace_raw_output_fields(item) for item in value]

    if isinstance(value, tuple):
        return [_redact_trace_raw_output_fields(item) for item in value]

    return value


def _serialize_edge(node: DialogueTreeNode) -> dict[str, Any]:
    candidate = _to_portable_value(node.candidate)
    if not isinstance(candidate, dict):
        candidate = {}
    return {
        "parent_node_id": node.parent_node_id,
        "node_id": node.node_id,
        "action_type": candidate.get("action_type"),
        "candidate_text": candidate.get("text", ""),
        "tool_plan": candidate.get("tool_plan"),
        "metadata": candidate.get("metadata", {}),
    }


def _serialize_node(
    node: DialogueTreeNode,
    *,
    prune_diagnostics: list[Any],
    score_diagnostics: list[Any],
) -> dict[str, Any]:
    return {
        "node_id": node.node_id,
        "parent_node_id": node.parent_node_id,
        "depth": node.depth,
        "candidate": _serialize_candidate(node.candidate),
        "payload": _to_portable_value(node.payload),
        "prune_diagnostics": [_to_portable_value(item) for item in prune_diagnostics],
        "score_diagnostics": [_to_portable_value(item) for item in score_diagnostics],
    }


def _serialize_candidate(candidate: Any) -> Any:
    if candidate is None:
        return None
    candidate_dict = _to_portable_value(candidate)
    if isinstance(candidate_dict, dict):
        return {
            "action_type": candidate_dict.get("action_type"),
            "text": candidate_dict.get("text", ""),
            "tool_plan": candidate_dict.get("tool_plan"),
            "metadata": candidate_dict.get("metadata", {}),
        }
    return candidate_dict


def _to_portable_value(value: Any) -> Any:
    if is_dataclass(value):
        return _to_portable_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_portable_value(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [_to_portable_value(item) for item in value]
    if isinstance(value, tuple):
        return [_to_portable_value(item) for item in value]
    if isinstance(value, set):
        return sorted(
            (_to_portable_value(item) for item in value),
            key=_portable_sort_key,
        )
    return value


def _portable_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    portable = _to_portable_value(value)
    if isinstance(portable, Mapping):
        return {str(key): sub_value for key, sub_value in portable.items()}
    return {}


def _portable_sort_key(value: Any) -> tuple[int, str]:
    if isinstance(value, bool):
        return (0, str(int(value)))
    if isinstance(value, int):
        return (1, f"{value:020d}")
    if isinstance(value, float):
        return (2, repr(value))
    if isinstance(value, str):
        return (3, value)
    if value is None:
        return (4, "")
    return (5, repr(value))


def _node_sort_key(node_id: str) -> tuple[tuple[int, Any], ...]:
    if node_id == "root":
        return ()
    segments = node_id.split(".")
    if segments and segments[0] == "root":
        segments = segments[1:]

    key_parts: list[tuple[int, Any]] = []
    for segment in segments:
        if segment.isdigit():
            key_parts.append((0, int(segment)))
        else:
            key_parts.append((1, segment))
    return tuple(key_parts)


__all__ = ["serialize_dialogue_tree_trace"]
