"""Shared Tree-sitter helpers for JVM-family CodeGraph extractors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.CodeGraph.models import (
    CodeGraphEdge,
    CodeGraphNode,
    CodeGraphUnresolvedRef,
    make_edge_id,
    make_node_id,
)


@dataclass(frozen=True)
class JvmCallSite:
    """Deferred same-file call reference captured inside a callable body."""

    source_node_id: str
    reference_name: str
    line: int
    column: int


def node_text(source: bytes, node: Any | None) -> str:
    """Return a node's UTF-8 source slice, or an empty string for missing nodes."""
    if node is None:
        return ""
    return source[node.start_byte : node.end_byte].decode("utf-8")


def declaration_payload_text(source: bytes, node: Any, keyword: str) -> str:
    """Return declaration text after a leading keyword while preserving syntax."""
    text = node_text(source, node).strip()
    if text.startswith(keyword):
        text = text[len(keyword) :].strip()
    if text.endswith(";"):
        text = text[:-1].strip()
    return " ".join(text.split())


def named_descendants_of_type(node: Any, *node_types: str) -> tuple[Any, ...]:
    """Return named descendants matching any requested Tree-sitter node type."""
    matches: list[Any] = []
    requested = set(node_types)
    stack = list(reversed(node.named_children))
    while stack:
        current = stack.pop()
        if current.type in requested:
            matches.append(current)
        stack.extend(reversed(current.named_children))
    return tuple(matches)


def first_named_child_of_type(node: Any, *node_types: str) -> Any | None:
    """Return the first direct named child with a requested Tree-sitter node type."""
    requested = set(node_types)
    for child in node.named_children:
        if child.type in requested:
            return child
    return None


def module_qualified_name(file_path: str) -> str:
    """Return a dotted module identity from a workspace-relative path."""
    path = Path(file_path)
    return ".".join((*path.with_suffix("").parts,))


def qualified_name(*parts: str) -> str:
    """Join qualified-name parts while dropping empty segments."""
    return ".".join(part for part in parts if part)


def make_jvm_node(
    *,
    workspace_key: str,
    language_id: str,
    file_path: str,
    kind: str,
    name: str,
    qualified_name_value: str,
    node: Any,
    start_line: int | None = None,
    end_line: int | None = None,
    signature: str | None = None,
    visibility: str | None = None,
    flags: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
) -> CodeGraphNode:
    """Create a stable CodeGraph node for a JVM Tree-sitter node."""
    resolved_start = start_line or line(node)
    identity_key = (
        f"{workspace_key}:{language_id}:{file_path}:{kind}:{qualified_name_value}:{resolved_start}"
    )
    return CodeGraphNode(
        id=make_node_id(workspace_key, language_id, file_path, kind, qualified_name_value, resolved_start),
        identity_key=identity_key,
        kind=kind,
        name=name,
        qualified_name=qualified_name_value,
        file_path=file_path,
        language=language_id,
        start_line=resolved_start,
        end_line=end_line or end_line_number(node),
        start_column=column(node),
        end_column=end_column(node),
        signature=signature,
        visibility=visibility,
        flags=flags,
        metadata=dict(metadata or {}),
    )


def remember_callable(callable_by_name: dict[str, CodeGraphNode | None], node: CodeGraphNode) -> None:
    """Register a callable by simple and qualified names unless ambiguous."""
    for key in {node.name, node.qualified_name}:
        existing = callable_by_name.get(key)
        if existing is None and key in callable_by_name:
            continue
        if existing is not None and existing.id != node.id:
            callable_by_name[key] = None
            continue
        callable_by_name[key] = node


def resolve_call_sites(
    *,
    call_sites: tuple[JvmCallSite, ...],
    callable_by_name: dict[str, CodeGraphNode | None],
    file_path: str,
    language_id: str,
    provenance: str,
) -> tuple[tuple[CodeGraphEdge, ...], tuple[CodeGraphUnresolvedRef, ...]]:
    """Resolve deferred same-file calls into edges or unresolved references."""
    edges: list[CodeGraphEdge] = []
    unresolved_refs: list[CodeGraphUnresolvedRef] = []
    for call in call_sites:
        target = callable_by_name.get(call.reference_name)
        if target is None:
            unresolved_refs.append(
                CodeGraphUnresolvedRef(
                    from_node_id=call.source_node_id,
                    reference_name=call.reference_name,
                    reference_kind="call",
                    file_path=file_path,
                    line=call.line,
                    column=call.column,
                    language=language_id,
                )
            )
            continue
        edges.append(
            CodeGraphEdge(
                id=make_edge_id(call.source_node_id, "calls", target.id, file_path, call.line, call.column),
                source=call.source_node_id,
                target=target.id,
                kind="calls",
                file_path=file_path,
                line=call.line,
                column=call.column,
                provenance=provenance,
            )
        )
    return tuple(edges), tuple(unresolved_refs)


def line(node: Any) -> int:
    """Return a Tree-sitter start line using 1-based coordinates."""
    return int(node.start_point.row) + 1


def column(node: Any) -> int:
    """Return a Tree-sitter start column using 1-based coordinates."""
    return int(node.start_point.column) + 1


def end_line_number(node: Any) -> int:
    """Return a Tree-sitter end line using 1-based coordinates."""
    return int(node.end_point.row) + 1


def end_column(node: Any) -> int:
    """Return a Tree-sitter end column using 1-based coordinates."""
    return int(node.end_point.column) + 1


__all__ = [
    "JvmCallSite",
    "column",
    "declaration_payload_text",
    "end_column",
    "end_line_number",
    "first_named_child_of_type",
    "line",
    "make_jvm_node",
    "module_qualified_name",
    "named_descendants_of_type",
    "node_text",
    "qualified_name",
    "remember_callable",
    "resolve_call_sites",
]
