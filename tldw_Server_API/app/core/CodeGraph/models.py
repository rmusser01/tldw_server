from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def stable_hash_id(prefix: str, identity: str) -> str:
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
    identity = ":".join(
        (
            workspace_key,
            language,
            file_path,
            kind,
            qualified_name,
            str(int(start_line)),
        )
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
    identity = ":".join(
        (
            source_node_id,
            edge_kind,
            target_or_ref,
            file_path,
            str(int(line)),
            str(int(column)),
        )
    )
    return stable_hash_id("edge", identity)


@dataclass(frozen=True)
class LanguageInfo:
    language_id: str
    display_name: str
    extensions: tuple[str, ...]
    stage: str
    dependency_missing: tuple[str, ...] = ()
    symbol_extraction: bool = False


@dataclass(frozen=True)
class WorkspaceResolution:
    workspace_root: Path
    workspace_key: str
    index_db_path: Path
    workspace_id: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class IndexedFile:
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
class IndexRunSummary:
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
    dependency_available: bool
    languages: tuple[LanguageInfo, ...]
    workspace_key: str
    index_present: bool
    counts: dict[str, int]
    last_index_run: IndexRunSummary | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
