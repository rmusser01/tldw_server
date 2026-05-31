"""Shared value objects and stable ID helpers for native CodeGraph indexes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def stable_hash_id(prefix: str, identity: str) -> str:
    """Return a deterministic prefixed identifier for a logical graph identity."""
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def make_node_id(
    workspace_key: str,
    language: str,
    file_path: str,
    kind: str,
    qualified_name: str,
    start_line: int,
) -> str:
    """Build the stable node identifier used across repeated workspace indexes."""
    identity = json.dumps(
        [
            workspace_key,
            language,
            file_path,
            kind,
            qualified_name,
            int(start_line),
        ],
        separators=(",", ":"),
    )
    return stable_hash_id("node", identity)


def make_edge_id(
    source_node_id: str,
    edge_kind: str,
    target_or_ref: str,
    file_path: str,
    line: int,
    column: int,
) -> str:
    """Build a stable edge identifier from source, relation, target, and location."""
    identity = json.dumps(
        [
            source_node_id,
            edge_kind,
            target_or_ref,
            file_path,
            int(line),
            int(column),
        ],
        separators=(",", ":"),
    )
    return stable_hash_id("edge", identity)


@dataclass(frozen=True)
class LanguageInfo:
    """Language support metadata exposed through CodeGraph status and validation."""

    language_id: str
    display_name: str
    extensions: tuple[str, ...]
    stage: str
    dependency_missing: tuple[str, ...] = ()
    symbol_extraction: bool = False


@dataclass(frozen=True)
class WorkspaceResolution:
    """Resolved workspace root and database location for one MCP request context."""

    workspace_root: Path
    workspace_key: str
    index_db_path: Path
    workspace_id: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class IndexedFile:
    """Persisted file inventory row for a workspace-relative source file."""

    path: str
    language: str
    size: int
    content_hash: str
    modified_at: float
    indexed_at: str | None = None
    status: str = "indexed"
    node_count: int = 0
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodeGraphNode:
    """Symbol, import, or module node extracted from an indexed source file."""

    id: str
    identity_key: str
    kind: str
    name: str
    qualified_name: str
    file_path: str
    language: str
    start_line: int | None = None
    end_line: int | None = None
    start_column: int | None = None
    end_column: int | None = None
    signature: str | None = None
    docstring: str | None = None
    visibility: str | None = None
    flags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def codegraph_node_to_dict(node: CodeGraphNode) -> dict[str, Any]:
    """Serialize a CodeGraphNode for API and repository relationship payloads."""
    return {
        "id": node.id,
        "kind": node.kind,
        "name": node.name,
        "qualified_name": node.qualified_name,
        "file_path": node.file_path,
        "language": node.language,
        "start_line": node.start_line,
        "end_line": node.end_line,
        "start_column": node.start_column,
        "end_column": node.end_column,
        "signature": node.signature,
        "docstring": node.docstring,
        "visibility": node.visibility,
        "flags": list(node.flags),
        "metadata": dict(node.metadata),
    }


@dataclass(frozen=True)
class CodeGraphEdge:
    """Directed graph relationship between two indexed CodeGraph nodes."""

    id: str
    source: str
    target: str | None
    kind: str
    file_path: str
    line: int | None = None
    column: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: str | None = None


@dataclass(frozen=True)
class CodeGraphUnresolvedRef:
    """Reference that could not be resolved to a concrete node during extraction."""

    from_node_id: str
    reference_name: str
    reference_kind: str
    file_path: str
    line: int | None = None
    column: int | None = None
    candidates: tuple[str, ...] = ()
    language: str | None = None


@dataclass(frozen=True)
class StoredCodeGraphReference:
    """Persisted reference row with optional resolved-edge state."""

    id: int
    from_node_id: str
    reference_name: str
    reference_kind: str
    file_path: str
    line: int | None = None
    column: int | None = None
    candidates: tuple[str, ...] = ()
    language: str | None = None
    resolved_target: str | None = None
    resolved_edge: str | None = None
    resolution_kind: str | None = None
    resolved_at: str | None = None


@dataclass(frozen=True)
class ExtractionResult:
    """Extractor output for one file, including graph rows and parse errors."""

    nodes: tuple[CodeGraphNode, ...] = ()
    edges: tuple[CodeGraphEdge, ...] = ()
    unresolved_refs: tuple[CodeGraphUnresolvedRef, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class IndexRunSummary:
    """Stored summary for a completed or running CodeGraph index operation."""

    run_id: str
    workspace_key: str
    mode: str
    status: str
    counters: dict[str, int] = field(default_factory=dict)
    error_summary: tuple[str, ...] = ()
    started_at: str | None = None
    finished_at: str | None = None


@dataclass(frozen=True)
class CodeGraphStatus:
    """Read-only status payload describing dependencies, languages, and index state."""

    dependency_available: bool
    languages: tuple[LanguageInfo, ...]
    workspace_key: str
    index_present: bool
    counts: dict[str, int]
    last_index_run: IndexRunSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
