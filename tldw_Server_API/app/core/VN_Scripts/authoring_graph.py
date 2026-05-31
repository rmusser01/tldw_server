"""Pure VN script authoring graph builder."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any
from urllib.parse import quote, unquote

SCHEMA_VERSION = "vn_script_authoring_graph.v1"
GRAPH_SEMANTICS_VERSION = "vn_script_authoring_graph_edges.v1"
PROGRAM_SCHEMA_VERSION = "vn_script_program.v1"
MAX_SUPPLIED_DRAFT_BYTES = 1_048_576
MAX_LABELS = 500
MAX_OPS = 5000
MAX_EDGES = 10000
MAX_SUMMARY_LENGTH = 240

_DYNAMIC_FLOW_OPS = {"random", "return"}


def encoded_label_id(label: str) -> str:
    """Return a deterministic graph node ID for a label."""
    return "label:" + _quote_label(label)


def operation_id(label: str, index: int) -> str:
    """Return a deterministic graph node ID for an operation."""
    return f"op:{_quote_label(label)}:{index}"


def bracket_label_path(label: str) -> str:
    """Return the bracket-notation JSON path for a label."""
    escaped = label.replace("\\", "\\\\").replace("'", "\\'")
    return f"$.labels['{escaped}']"


def content_hash_for_program(
    program: Mapping[str, Any],
    *,
    graph_semantics_version: str = GRAPH_SEMANTICS_VERSION,
    program_schema_version: str = PROGRAM_SCHEMA_VERSION,
) -> str:
    """Return the canonical authoring graph content hash for a source program."""
    payload = {
        "graph_semantics_version": graph_semantics_version,
        "program": program,
        "program_schema_version": program_schema_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_script_authoring_graph(
    program: Mapping[str, Any],
    *,
    source: str = "stored_draft",
    script_id: int | None = None,
    base_revision: int | None = None,
    version_id: int | None = None,
    validation_diagnostics: Mapping[str, Any] | None = None,
    validation_context_source: str = "current_draft_context",
    limits: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a static authoring graph from a parsed VN script program."""
    active_limits = _active_limits(limits)
    diagnostics: dict[str, list[dict[str, Any]]] = {"errors": [], "warnings": []}
    truncated = False
    labels = program.get("labels")
    entry_label = program.get("entry_label") if isinstance(program.get("entry_label"), str) else None
    label_items = _ordered_label_items(labels, entry_label) if isinstance(labels, Mapping) else []

    if not isinstance(labels, Mapping) or not labels:
        _append_diag(
            diagnostics["errors"],
            "graph_labels_missing",
            "error",
            "Script must define at least one label.",
            "$.labels",
        )

    if len(label_items) > active_limits["max_labels"]:
        label_items = label_items[: active_limits["max_labels"]]
        truncated = True
        _append_diag(
            diagnostics["warnings"],
            "graph_node_limit_exceeded",
            "warning",
            "Graph label limit was reached; output is partial.",
            "$.labels",
            {"limit": active_limits["max_labels"]},
        )

    label_names = {label for label, _ in label_items}
    all_label_names = {str(label) for label in labels} if isinstance(labels, Mapping) else set()
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    outgoing_counts: dict[str, int] = {label: 0 for label, _ in label_items}
    incoming_counts: dict[str, int] = {label: 0 for label, _ in label_items}
    op_count_seen = 0

    for label, raw_ops in label_items:
        label_path = bracket_label_path(label)
        nodes.append(
            {
                "id": encoded_label_id(label),
                "type": "label",
                "label": label,
                "source_path": label_path,
                "reachable": False,
                "terminal": "unknown",
                "summary": _clip_summary(_label_node_summary(raw_ops)),
            }
        )
        if not isinstance(raw_ops, list):
            _append_diag(
                diagnostics["errors"],
                "graph_label_body_invalid",
                "error",
                "Label body must be a list of opcodes.",
                label_path,
                {"label": label},
            )
            continue

        for index, opcode in enumerate(raw_ops):
            if op_count_seen >= active_limits["max_ops"]:
                truncated = True
                _append_diag(
                    diagnostics["warnings"],
                    "graph_node_limit_exceeded",
                    "warning",
                    "Graph operation limit was reached; output is partial.",
                    f"{label_path}[{index}]",
                    {"limit": active_limits["max_ops"]},
                )
                break
            op_count_seen += 1
            op_path = f"{label_path}[{index}]"
            op_name = _op_name(opcode) if isinstance(opcode, Mapping) else ""
            if not isinstance(opcode, Mapping):
                _append_diag(
                    diagnostics["errors"],
                    "graph_opcode_invalid",
                    "error",
                    "Opcode must be an object.",
                    op_path,
                    {"label": label, "op_index": index},
                )
                nodes.append(_operation_node(label, index, "invalid", op_path, "Invalid opcode."))
                continue

            nodes.append(_operation_node(label, index, op_name or "unknown", op_path, _operation_summary(opcode)))
            if opcode.get("if") is not None or op_name in _DYNAMIC_FLOW_OPS:
                _append_diag(
                    diagnostics["warnings"],
                    "graph_unsupported_dynamic_flow",
                    "warning",
                    "Operation uses dynamic or conditional flow; terminal state is unknown.",
                    op_path,
                    {"label": label, "op_index": index, "op": op_name},
                )

            extracted = _extract_static_edges(label, index, opcode, label_names, all_label_names, diagnostics)
            for edge in extracted:
                if len(edges) >= active_limits["max_edges"]:
                    truncated = True
                    _append_diag(
                        diagnostics["warnings"],
                        "graph_edge_limit_exceeded",
                        "warning",
                        "Graph edge limit was reached; output is partial.",
                        edge["source_path"],
                        {"limit": active_limits["max_edges"]},
                    )
                    break
                edges.append(edge)
                outgoing_counts[label] = outgoing_counts.get(label, 0) + 1
                target_label = edge["target_label"]
                if edge["target_id"] is not None:
                    if target_label in incoming_counts:
                        incoming_counts[target_label] += 1

    reachable = _reachable_labels(entry_label, edges) if entry_label else set()
    for label in sorted(label_names - reachable):
        if label != entry_label:
            _append_diag(
                diagnostics["warnings"],
                "graph_label_unreachable",
                "warning",
                "Label is never reached from the entry label.",
                bracket_label_path(label),
                {"label": label},
            )

    terminal_by_label = {
        label: _terminal_state(raw_ops, outgoing_counts.get(label, 0))
        for label, raw_ops in label_items
    }
    _append_fallthrough_limitations(label_items, outgoing_counts, terminal_by_label, diagnostics)
    reachable_by_label = {label: label in reachable for label, _ in label_items}
    for node in nodes:
        label = str(node["label"])
        if node["type"] == "label":
            node["reachable"] = reachable_by_label.get(label, False)
            node["terminal"] = terminal_by_label.get(label, "unknown")

    outline_labels = [
        _outline_label(
            label,
            raw_ops,
            incoming_counts.get(label, 0),
            outgoing_counts.get(label, 0),
            reachable_by_label.get(label, False),
            terminal_by_label.get(label, "unknown"),
        )
        for label, raw_ops in label_items
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "graph_semantics_version": GRAPH_SEMANTICS_VERSION,
        "program_schema_version": PROGRAM_SCHEMA_VERSION,
        "source": source,
        "script_id": script_id,
        "base_revision": base_revision,
        "version_id": version_id,
        "content_hash": content_hash_for_program(program),
        "validation_context_source": validation_context_source,
        "truncated": truncated,
        "limits": {
            "max_labels": active_limits["max_labels"],
            "max_ops": active_limits["max_ops"],
            "max_edges": active_limits["max_edges"],
            "max_supplied_draft_bytes": active_limits["max_supplied_draft_bytes"],
        },
        "outline": {"entry_label": entry_label, "labels": outline_labels},
        "graph": {"nodes": nodes, "edges": edges},
        "diagnostics": diagnostics,
        "validation_diagnostics": _validation_diagnostics(validation_diagnostics),
    }


def _active_limits(overrides: Mapping[str, int] | None) -> dict[str, int]:
    limits = {
        "max_labels": MAX_LABELS,
        "max_ops": MAX_OPS,
        "max_edges": MAX_EDGES,
        "max_supplied_draft_bytes": MAX_SUPPLIED_DRAFT_BYTES,
    }
    if overrides:
        for key in ("max_labels", "max_ops", "max_edges", "max_supplied_draft_bytes"):
            if key in overrides:
                limits[key] = max(0, int(overrides[key]))
    return limits


def _ordered_label_items(labels: Mapping[str, Any], entry_label: str | None) -> list[tuple[str, Any]]:
    items = [(str(label), raw_ops) for label, raw_ops in labels.items()]
    if not entry_label:
        return items
    entry_items = [(label, raw_ops) for label, raw_ops in items if label == entry_label]
    return entry_items + [(label, raw_ops) for label, raw_ops in items if label != entry_label]


def _operation_node(label: str, index: int, op_name: str, source_path: str, summary: str) -> dict[str, Any]:
    return {
        "id": operation_id(label, index),
        "type": "operation",
        "label": label,
        "op_index": index,
        "op": op_name,
        "source_path": source_path,
        "summary": _clip_summary(summary),
    }


def _outline_label(
    label: str,
    raw_ops: Any,
    incoming_edge_count: int,
    outgoing_edge_count: int,
    reachable: bool,
    terminal: str,
) -> dict[str, Any]:
    op_count = len(raw_ops) if isinstance(raw_ops, list) else 0
    return {
        "id": encoded_label_id(label),
        "label": label,
        "source_path": bracket_label_path(label),
        "op_count": op_count,
        "incoming_edge_count": incoming_edge_count,
        "outgoing_edge_count": outgoing_edge_count,
        "reachable": reachable,
        "terminal": terminal,
        "summary": _clip_summary(_outline_summary(op_count, outgoing_edge_count)),
    }


def _label_node_summary(raw_ops: Any) -> str:
    if not isinstance(raw_ops, list):
        return "Invalid label body."
    return _outline_summary(len(raw_ops), 0)


def _outline_summary(op_count: int, outgoing_edge_count: int) -> str:
    op_word = "operation" if op_count == 1 else "operations"
    if outgoing_edge_count:
        edge_word = "edge" if outgoing_edge_count == 1 else "edges"
        return f"{op_count} {op_word} and {outgoing_edge_count} outgoing {edge_word}."
    return f"{op_count} {op_word}."


def _operation_summary(opcode: Mapping[str, Any]) -> str:
    op = _op_name(opcode)
    if op == "generate":
        output_schema = (
            opcode.get("output_schema")
            if isinstance(opcode.get("output_schema"), str)
            else "narrative_dialogue"
        )
        profile_key = (
            opcode.get("profile_key")
            if isinstance(opcode.get("profile_key"), str)
            else "default"
        )
        return f"Generate {output_schema} using profile {profile_key}."
    if op == "choice":
        choices = opcode.get("choices")
        count = len(choices) if isinstance(choices, list) else 0
        choice_word = "choice" if count == 1 else "choices"
        return f"Authored choice with {count} {choice_word}."
    if op == "jump":
        target = opcode.get("target")
        return f"Jump to {target}." if isinstance(target, str) else "Jump with invalid target."
    if op == "end":
        return "End script."
    if op == "narrate":
        return "Narration."
    if op == "say":
        return "Dialogue."
    return f"{op} operation."


def _extract_static_edges(
    label: str,
    index: int,
    opcode: Mapping[str, Any],
    emitted_label_names: set[str],
    all_label_names: set[str],
    diagnostics: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    op = opcode.get("op")
    if op == "jump":
        return [
            _edge(
                label,
                index,
                "jump",
                opcode.get("target"),
                f"{bracket_label_path(label)}[{index}].target",
                emitted_label_names,
                all_label_names,
                diagnostics,
                missing_code="graph_target_missing",
            )
        ]
    if op == "choice":
        choices = opcode.get("choices")
        if not isinstance(choices, list):
            _append_diag(
                diagnostics["errors"],
                "graph_choice_options_invalid",
                "error",
                "Choice opcode requires a list of options.",
                f"{bracket_label_path(label)}[{index}].choices",
                {"label": label, "op_index": index},
            )
            return []
        edges: list[dict[str, Any]] = []
        for choice_index, choice in enumerate(choices):
            choice_path = f"{bracket_label_path(label)}[{index}].choices[{choice_index}]"
            if not isinstance(choice, Mapping):
                _append_diag(
                    diagnostics["errors"],
                    "graph_choice_options_invalid",
                    "error",
                    "Choice option must be an object.",
                    choice_path,
                    {"label": label, "op_index": index, "choice_index": choice_index},
                )
                continue
            edges.append(
                _edge(
                    label,
                    index,
                    "choice",
                    choice.get("target"),
                    f"{choice_path}.target",
                    emitted_label_names,
                    all_label_names,
                    diagnostics,
                    missing_code="graph_target_missing",
                    discriminator=f"choice:{choice_index}",
                    metadata={"choice_index": choice_index},
                )
            )
        return edges
    if op == "generate":
        edges = []
        if "on_generated_choice" in opcode or opcode.get("output_schema") == "choice_set":
            edges.append(
                _edge(
                    label,
                    index,
                    "generated_choice_handler",
                    opcode.get("on_generated_choice"),
                    f"{bracket_label_path(label)}[{index}].on_generated_choice",
                    emitted_label_names,
                    all_label_names,
                    diagnostics,
                    missing_code="graph_generated_choice_handler_missing",
                    metadata={
                        "output_schema": opcode.get("output_schema")
                        if isinstance(opcode.get("output_schema"), str)
                        else None
                    },
                )
            )
        if "on_cancel" in opcode:
            edges.append(
                _edge(
                    label,
                    index,
                    "generation_cancel",
                    opcode.get("on_cancel"),
                    f"{bracket_label_path(label)}[{index}].on_cancel",
                    emitted_label_names,
                    all_label_names,
                    diagnostics,
                    missing_code="graph_cancel_target_missing",
                )
            )
        return edges
    return []


def _edge(
    label: str,
    index: int,
    edge_type: str,
    raw_target: Any,
    source_path: str,
    emitted_label_names: set[str],
    all_label_names: set[str],
    diagnostics: dict[str, list[dict[str, Any]]],
    *,
    missing_code: str,
    discriminator: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source_id = operation_id(label, index)
    target_label = raw_target if isinstance(raw_target, str) else ""
    target_id = encoded_label_id(target_label) if target_label in emitted_label_names else None
    missing_key = "missing:" + _quote_label(target_label or "invalid")
    edge_id_parts = ["edge", source_id, edge_type]
    if discriminator is not None:
        edge_id_parts.append(discriminator)
    edge_id_parts.append(target_id or missing_key)
    edge = {
        "id": ":".join(edge_id_parts),
        "type": edge_type,
        "source_id": source_id,
        "target_id": target_id,
        "source_path": source_path,
        "target_label": target_label,
    }
    compact_metadata = {key: value for key, value in dict(metadata or {}).items() if value is not None}
    if compact_metadata:
        edge["metadata"] = compact_metadata
    if target_id is None and target_label in all_label_names:
        edge["omitted_target"] = True
        _append_diag(
            diagnostics["warnings"],
            "graph_target_omitted",
            "warning",
            "Target label was omitted because the graph node limit was reached.",
            source_path,
            {"target_label": target_label, "edge_type": edge_type},
        )
    elif target_id is None:
        edge["missing_target"] = True
        _append_diag(
            diagnostics["errors"],
            missing_code,
            "error",
            "Target label was not found.",
            source_path,
            {"target_label": target_label, "edge_type": edge_type},
        )
    return edge


def _validation_diagnostics(validation_diagnostics: Mapping[str, Any] | None) -> dict[str, Any]:
    if validation_diagnostics is None:
        return {"valid": False, "errors": [], "warnings": []}
    return {
        "valid": bool(validation_diagnostics.get("valid", False)),
        "errors": list(validation_diagnostics.get("errors") or []),
        "warnings": list(validation_diagnostics.get("warnings") or []),
    }


def _reachable_labels(entry_label: str, edges: list[dict[str, Any]]) -> set[str]:
    """Return labels reachable through emitted static graph edges."""
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        if edge.get("target_id") is None:
            continue
        source_label = _edge_source_label(edge)
        target_label = edge.get("target_label")
        if isinstance(source_label, str) and isinstance(target_label, str):
            adjacency.setdefault(source_label, []).append(target_label)

    reachable: set[str] = set()
    stack = [entry_label]
    while stack:
        label = stack.pop()
        if label in reachable:
            continue
        reachable.add(label)
        stack.extend(reversed(adjacency.get(label, [])))
    return reachable


def _edge_source_label(edge: Mapping[str, Any]) -> str | None:
    """Extract the encoded source label from an emitted operation edge ID."""
    source_id = edge.get("source_id")
    if not isinstance(source_id, str):
        return None
    parts = source_id.split(":")
    if len(parts) < 3 or parts[0] != "op":
        return None
    return unquote(parts[1])


def _append_fallthrough_limitations(
    label_items: list[tuple[str, Any]],
    outgoing_counts: Mapping[str, int],
    terminal_by_label: Mapping[str, str],
    diagnostics: dict[str, list[dict[str, Any]]],
) -> None:
    """Append warnings for labels whose possible fallthrough is intentionally static-only."""
    for index, (label, raw_ops) in enumerate(label_items[:-1]):
        if not isinstance(raw_ops, list):
            continue
        if outgoing_counts.get(label, 0) > 0 or terminal_by_label.get(label) != "unknown":
            continue
        next_label = label_items[index + 1][0]
        _append_diag(
            diagnostics["warnings"],
            "graph_fallthrough_not_inferred",
            "warning",
            "Static graph reachability does not infer implicit fallthrough to the next label.",
            bracket_label_path(label),
            {"label": label, "next_label": next_label},
        )


def _terminal_state(raw_ops: Any, outgoing_edge_count: int) -> str:
    if not isinstance(raw_ops, list):
        return "unknown"
    malformed = False
    dynamic_or_conditional = False
    last_op = ""
    for opcode in raw_ops:
        if not isinstance(opcode, Mapping):
            malformed = True
            continue
        last_op = _op_name(opcode)
        if opcode.get("if") is not None or last_op in _DYNAMIC_FLOW_OPS:
            dynamic_or_conditional = True
    if malformed or dynamic_or_conditional:
        return "unknown"
    if last_op == "end" and outgoing_edge_count == 0:
        return "terminal"
    if outgoing_edge_count > 0:
        return "continues"
    return "unknown"


def _append_diag(
    target: list[dict[str, Any]],
    code: str,
    severity: str,
    message: str,
    path: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    target.append(
        {
            "code": code,
            "severity": severity,
            "message": message,
            "path": path,
            "details": dict(details or {}),
        }
    )


def _clip_summary(summary: str) -> str:
    if len(summary) <= MAX_SUMMARY_LENGTH:
        return summary
    return summary[: MAX_SUMMARY_LENGTH - 3].rstrip() + "..."


def _quote_label(label: str) -> str:
    return quote(label, safe="").replace(".", "%2E")


def _op_name(opcode: Mapping[str, Any]) -> str:
    op = opcode.get("op")
    return op if isinstance(op, str) and op else "unknown"
