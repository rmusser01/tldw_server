"""Immutable contracts for shared Workspace clone snapshots and results."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

MAX_IDENTIFIER_LENGTH = 255
MAX_NAME_LENGTH = 255
MAX_WARNING_CODE_LENGTH = 128
MAX_WARNING_COUNT = 1_000_000_000
MAX_WARNINGS = 8

_READINESS_VALUES = frozenset({"ready", "unavailable"})
_VECTOR_READINESS_VALUES = frozenset({"ready", "needs_indexing", "not_configured"})
_OUTCOMES = frozenset({"complete", "partial"})


def _validate_ascii(value: Any, field_name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty ASCII string")
    if len(value) > maximum:
        raise ValueError(f"{field_name} must be at most {maximum} characters")
    if not value.isascii() or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value):
        raise ValueError(f"{field_name} must contain only printable ASCII characters")
    return value


def _validate_identifier(value: Any, field_name: str) -> str:
    return _validate_ascii(value, field_name, MAX_IDENTIFIER_LENGTH)


def _normalize_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("name must be a non-empty string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("name must be non-empty after normalization")
    if len(normalized) > MAX_NAME_LENGTH:
        raise ValueError(f"name must be at most {MAX_NAME_LENGTH} characters")
    return normalized


def _freeze(value: Any) -> Any:
    """Deep-copy a row and replace mutable containers with immutable equivalents."""
    if isinstance(value, Mapping):
        return MappingProxyType(
            {deepcopy(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return deepcopy(value)


def _freeze_row(row: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise TypeError(f"{field_name} rows must be mappings")
    frozen = _freeze(row)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by _freeze
        raise TypeError(f"{field_name} rows must be mappings")
    return frozen


def _freeze_rows(rows: Iterable[Mapping[str, Any]], field_name: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(_freeze_row(row, field_name) for row in rows)


@dataclass(frozen=True, slots=True)
class WorkspaceCloneRequest:
    """Validated identity and normalized target name for one clone operation."""

    source_workspace_id: str
    target_workspace_id: str
    operation_id: str
    request_fingerprint: str
    name: str

    def __post_init__(self) -> None:
        for field_name in (
            "source_workspace_id",
            "target_workspace_id",
            "operation_id",
            "request_fingerprint",
        ):
            _validate_identifier(getattr(self, field_name), field_name)
        object.__setattr__(self, "name", _normalize_name(self.name))


@dataclass(frozen=True, slots=True)
class WorkspaceCloneSnapshot:
    """Immutable Workspace metadata and rows captured from one source snapshot."""

    workspace: Mapping[str, Any]
    memberships: tuple[Mapping[str, Any], ...]
    sources: tuple[Mapping[str, Any], ...]
    notes: tuple[Mapping[str, Any], ...]
    artifacts: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", _freeze_row(self.workspace, "workspace"))
        object.__setattr__(self, "memberships", _freeze_rows(self.memberships, "membership"))
        object.__setattr__(self, "sources", _freeze_rows(self.sources, "source"))
        object.__setattr__(self, "notes", _freeze_rows(self.notes, "note"))
        object.__setattr__(self, "artifacts", _freeze_rows(self.artifacts, "artifact"))

    @classmethod
    def from_rows(
        cls,
        workspace: Mapping[str, Any],
        sources: Iterable[Mapping[str, Any]],
        notes: Iterable[Mapping[str, Any]],
        artifacts: Iterable[Mapping[str, Any]],
        memberships: Iterable[Mapping[str, Any]] = (),
    ) -> WorkspaceCloneSnapshot:
        return cls(
            workspace=workspace,
            memberships=memberships,
            sources=sources,
            notes=notes,
            artifacts=artifacts,
        )


@dataclass(frozen=True, slots=True)
class MediaCloneSnapshot:
    """Immutable media metadata with its chunk and transcript rows."""

    media: Mapping[str, Any]
    chunks: tuple[Mapping[str, Any], ...]
    transcripts: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "media", _freeze_row(self.media, "media"))
        object.__setattr__(self, "chunks", _freeze_rows(self.chunks, "chunk"))
        object.__setattr__(self, "transcripts", _freeze_rows(self.transcripts, "transcript"))

    @classmethod
    def from_rows(
        cls,
        media: Mapping[str, Any],
        chunks: Iterable[Mapping[str, Any]],
        transcripts: Iterable[Mapping[str, Any]],
    ) -> MediaCloneSnapshot:
        return cls(
            media=media,
            chunks=chunks,
            transcripts=transcripts,
        )


def _validate_count(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    if value > MAX_WARNING_COUNT:
        raise ValueError(f"{field_name} exceeds the supported bound")
    return value


@dataclass(frozen=True, slots=True)
class CloneCopyCounts:
    """Truthful attempted/copied/failed counts for each copied item class."""

    sources_attempted: int = 0
    sources_copied: int = 0
    sources_failed: int = 0
    notes_attempted: int = 0
    notes_copied: int = 0
    notes_failed: int = 0
    artifacts_attempted: int = 0
    artifacts_copied: int = 0
    artifacts_failed: int = 0
    media_attempted: int = 0
    media_copied: int = 0
    media_failed: int = 0
    operation_owned_media_count: int = 0

    def __post_init__(self) -> None:
        count_fields = (
            "sources_attempted",
            "sources_copied",
            "sources_failed",
            "notes_attempted",
            "notes_copied",
            "notes_failed",
            "artifacts_attempted",
            "artifacts_copied",
            "artifacts_failed",
            "media_attempted",
            "media_copied",
            "media_failed",
            "operation_owned_media_count",
        )
        for field_name in count_fields:
            _validate_count(getattr(self, field_name), field_name)
        for item_name in ("sources", "notes", "artifacts", "media"):
            attempted = getattr(self, f"{item_name}_attempted")
            copied = getattr(self, f"{item_name}_copied")
            failed = getattr(self, f"{item_name}_failed")
            if copied + failed > attempted:
                raise ValueError(
                    f"{item_name}_copied plus {item_name}_failed cannot exceed {item_name}_attempted"
                )

    @classmethod
    def empty(cls) -> CloneCopyCounts:
        return cls()


@dataclass(frozen=True, slots=True)
class CloneRetrievalReadiness:
    """Independent readiness states for text, citation, and vector retrieval."""

    text_search: str
    citations: str
    vector_search: str

    def __post_init__(self) -> None:
        if self.text_search not in _READINESS_VALUES:
            raise ValueError("text_search must be ready or unavailable")
        if self.citations not in _READINESS_VALUES:
            raise ValueError("citations must be ready or unavailable")
        if self.vector_search not in _VECTOR_READINESS_VALUES:
            raise ValueError(
                "vector_search must be ready, needs_indexing, or not_configured"
            )


@dataclass(frozen=True, slots=True)
class CloneWarning:
    """A bounded, stable diagnostic that contains no source data or exception text."""

    code: str
    count: int

    def __post_init__(self) -> None:
        _validate_ascii(self.code, "warning code", MAX_WARNING_CODE_LENGTH)
        _validate_count(self.count, "warning count")


@dataclass(frozen=True, slots=True)
class WorkspaceCloneResult:
    """Safe terminal facts about one clone operation."""

    workspace_id: str
    name: str
    outcome: str
    publication_confirmed: bool
    counts: CloneCopyCounts
    readiness: CloneRetrievalReadiness
    warnings: tuple[CloneWarning, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _validate_identifier(self.workspace_id, "workspace_id")
        object.__setattr__(self, "name", _normalize_name(self.name))
        if self.outcome not in _OUTCOMES:
            raise ValueError("outcome must be complete or partial")
        if not isinstance(self.publication_confirmed, bool):
            raise ValueError("publication_confirmed must be a boolean")
        if self.outcome == "complete" and not self.publication_confirmed:
            raise ValueError("complete results require publication_confirmed")
        if not isinstance(self.counts, CloneCopyCounts):
            raise TypeError("counts must be CloneCopyCounts")
        if not isinstance(self.readiness, CloneRetrievalReadiness):
            raise TypeError("readiness must be CloneRetrievalReadiness")
        normalized_warnings = tuple(self.warnings)
        if len(normalized_warnings) > MAX_WARNINGS:
            raise ValueError("warnings may contain at most 8 entries")
        if any(not isinstance(warning, CloneWarning) for warning in normalized_warnings):
            raise TypeError("warnings must contain only CloneWarning values")
        object.__setattr__(self, "warnings", normalized_warnings)


@dataclass(frozen=True, slots=True)
class CloneCancelled(Exception):
    """Controlled cooperative-cancellation failure with no unsafe detail."""

    cleanup_state: str = "unknown"
    code: str = field(init=False, default="clone_cancelled")

    def __post_init__(self) -> None:
        _validate_ascii(self.cleanup_state, "cleanup_state", 16)
        if self.cleanup_state not in {"complete", "pending", "unknown"}:
            raise ValueError("cleanup_state must be complete, pending, or unknown")
        Exception.__init__(self, self.code)


@dataclass(frozen=True, slots=True)
class CloneSnapshotUnavailable(Exception):
    """Controlled failure when a coherent source snapshot cannot be established."""

    cleanup_state: str = "unknown"
    code: str = field(init=False, default="source_snapshot_unavailable")

    def __post_init__(self) -> None:
        _validate_ascii(self.cleanup_state, "cleanup_state", 16)
        if self.cleanup_state not in {"complete", "pending", "unknown"}:
            raise ValueError("cleanup_state must be complete, pending, or unknown")
        Exception.__init__(self, self.code)
