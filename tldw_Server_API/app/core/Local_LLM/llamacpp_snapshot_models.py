"""Strict persisted types for managed llama.cpp slot snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class Fingerprint(BaseModel):
    """Content-derived runtime identity required before restore."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_sha256: str = Field(pattern=_SHA256_PATTERN)
    executable_sha256: str = Field(pattern=_SHA256_PATTERN)
    projector_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    effective_options_sha256: str = Field(pattern=_SHA256_PATTERN)
    adapters_sha256: str = Field(pattern=_SHA256_PATTERN)
    format_version: Literal[1] = 1


class SnapshotRequest(BaseModel):
    """Path-free request shared by snapshot operation boundaries."""

    model_config = ConfigDict(extra="forbid", strict=True)

    slot_id: int = Field(ge=0)
    expected_launch_generation: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=512)
    replace_confirmed: bool = False


class SnapshotMetadata(BaseModel):
    """Published metadata for one immutable snapshot binary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(pattern=_ID_PATTERN)
    snapshot_id: str = Field(pattern=_ID_PATTERN)
    source_slot: int = Field(ge=0)
    created_at: datetime
    commit_sequence: int = Field(ge=1)
    byte_count: int = Field(gt=0)
    token_count: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    fingerprint: Fingerprint
    actor_id: str = Field(min_length=1, max_length=256)
    format_version: Literal[1] = 1

    @field_validator("created_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        """Reject naive or non-UTC timestamps in durable manifests."""
        if value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("created_at must be UTC")
        return value


class OperationReceipt(BaseModel):
    """Durable, path-free evidence for a single snapshot mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: str = Field(pattern=_ID_PATTERN)
    operation_id: str = Field(pattern=_ID_PATTERN)
    launch_generation: str = Field(min_length=1, max_length=128)
    request_digest: str = Field(pattern=_SHA256_PATTERN)
    kind: Literal["save", "restore"]
    state: Literal[
        "validating",
        "saving",
        "verifying",
        "restoring",
        "complete",
        "failed",
        "outcome_unknown",
    ]
    dispatched: bool = False
    actor_id: str = Field(default="unknown", min_length=1, max_length=256)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    warning_code: str | None = None
    snapshot_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    token_count: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]{0,127}$")
    format_version: Literal[1] = 1

    @property
    def recovery_action(self) -> Literal["none", "retry_manually", "stop_runtime"]:
        """Describe operator recovery without adding a field to durable receipts."""
        if self.state == "outcome_unknown":
            return "stop_runtime"
        if self.state == "failed":
            return "retry_manually"
        return "none"
