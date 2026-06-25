"""Shared models for context integrity verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    metadata: dict[str, Any] = field(default_factory=dict)


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
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True, slots=True)
class ContextIntegrityBootState:
    """Current-process context integrity verification result."""

    mode: Literal["audit_only", "enforce", "hardened"]
    degraded: bool
    manifest_sequence: int | None
    manifest_digest: str | None
    approved_digests_by_asset_id: dict[str, str] = field(default_factory=dict)
    findings: tuple[ContextIntegrityFinding, ...] = ()
