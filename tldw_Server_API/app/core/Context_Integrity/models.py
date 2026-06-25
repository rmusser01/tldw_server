"""Shared models for context integrity verification."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal

ContextAssetSource = Literal["skill_file", "prompt_file", "db_prompt"]
ContextAssetState = Literal[
    "trusted",
    "changed_approved_executable",
    "changed_approved_non_executable",
    "new_unapproved",
    "missing_required",
    "missing_optional",
    "signature_invalid",
    "manifest_rollback_detected",
    "verification_error",
    "degraded_integrity",
    "quarantined",
]


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


@dataclass(frozen=True, slots=True)
class CanonicalDigest:
    """Canonical digest plus optional JSON payload used to produce it."""

    digest: str
    canonical_json: str = ""

    def __str__(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class ContextAssetDescriptor:
    """One prompt-bearing asset discovered by an inventory adapter."""

    asset_id: str
    source_type: ContextAssetSource
    digest: str
    display_name: str
    executable: bool = False
    required: bool = False
    owner_scope: str = "system"
    path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ContextIntegrityFinding:
    """Verification finding for one asset or source scope."""

    asset_id: str
    state: ContextAssetState
    severity: Literal["info", "warning", "error"]
    summary: str
    remediation: str
    source_type: ContextAssetSource | Literal["manifest"]
    current_digest: str | None = None
    approved_digest: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", _freeze_mapping(self.details))


@dataclass(frozen=True, slots=True)
class ContextIntegrityBootState:
    """Current-process context integrity verification result."""

    mode: Literal["audit_only", "enforce", "hardened"]
    degraded: bool
    manifest_sequence: int | None
    manifest_digest: str | None
    approved_digests_by_asset_id: Mapping[str, str] = field(default_factory=dict)
    findings: tuple[ContextIntegrityFinding, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "approved_digests_by_asset_id",
            _freeze_mapping(self.approved_digests_by_asset_id),
        )
