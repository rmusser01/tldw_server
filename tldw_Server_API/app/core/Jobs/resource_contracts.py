"""Backend-neutral immutable contracts for exclusive Jobs resources."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

if TYPE_CHECKING:
    from .operations.contracts import AdmissionResult

RESOURCE_REQUEST_RETENTION_SECONDS = 86_400
RESOURCE_REQUEST_MAX_ALIASES = 128
RESOURCE_METADATA_FIELDS = (
    "resource",
    "resource_authority_uuid",
    "resource_generation",
    "resource_settings_json",
    "resource_settings_fingerprint",
)


def _identity(value: object, name: str) -> None:
    """Preserve generic Jobs owner strings while rejecting blank identities."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")


def _uuid(value: object, name: str) -> None:
    """Require a canonical opaque UUID, without imposing a UUID version."""
    try:
        valid = isinstance(value, str) and str(UUID(value)) == value
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be a canonical UUID")


def _json_value(value: object) -> object:
    """Validate JSON types and normalize negative zero at every nesting level."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return 0.0 if value == 0 else value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_value(item) for key, item in value.items()}
    raise ValueError("settings must contain only finite JSON values and string keys")


def canonical_settings_json(settings: dict[str, object]) -> str:
    """Encode semantic settings as sorted compact UTF-8 JSON (never NaN)."""
    if not isinstance(settings, dict):
        raise ValueError("settings must be an object")
    try:
        encoded = json.dumps(
            _json_value(settings), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        encoded.encode("utf-8")
        return encoded
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("settings must be a finite JSON object") from exc


def settings_fingerprint(settings: dict[str, object]) -> str:
    """Fingerprint the canonical encoding; reuse must also compare that encoding."""
    return hashlib.sha256(canonical_settings_json(settings).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResourceScope:
    """Exclusive resource identity, deliberately independent of queue and type."""

    domain: str
    owner_user_id: str
    resource: str

    def __post_init__(self) -> None:
        """Validate generic Jobs identities without Claims-specific coercions."""
        for name in ("domain", "owner_user_id", "resource"):
            _identity(getattr(self, name), name)


@dataclass(frozen=True)
class ResourceAdmissionRequest:
    """Semantic admission inputs; raw request keys must never be persisted."""

    scope: ResourceScope
    canonical_settings: dict[str, object]
    request_key: str | None = None

    def __post_init__(self) -> None:
        """Validate keys and copy nested caller-owned settings defensively."""
        if not isinstance(self.scope, ResourceScope):
            raise ValueError("scope must be a ResourceScope")
        canonical_settings_json(self.canonical_settings)
        key = self.request_key
        if key is not None and (
            not isinstance(key, str)
            or not 1 <= len(key) <= 128
            or key != key.strip()
            or any(not 32 <= ord(char) <= 126 for char in key)
        ):
            raise ValueError("request_key must be 1..128 printable ASCII characters without surrounding whitespace")
        object.__setattr__(self, "canonical_settings", copy.deepcopy(self.canonical_settings))


@dataclass(frozen=True)
class ResourceBinding:
    """Immutable authority and logical generation allocated by Jobs."""

    scope: ResourceScope
    authority_uuid: str
    generation: int

    def __post_init__(self) -> None:
        """Reject invalid identity and counter inputs before persistence."""
        if not isinstance(self.scope, ResourceScope):
            raise ValueError("scope must be a ResourceScope")
        _uuid(self.authority_uuid, "authority_uuid")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("generation must be a positive integer")


@dataclass(frozen=True)
class ResourceAdmission:
    """Shared admission outcome, not a second domain lifecycle record."""

    kind: Literal["created", "active_reused", "request_replayed", "active_conflict", "request_conflict", "rejected"]
    job: dict[str, Any] | None
    rejection: AdmissionResult | None

    def __post_init__(self) -> None:
        """Keep outcomes closed and isolate backend-owned mutable Job facts."""
        if self.kind not in (
            "created",
            "active_reused",
            "request_replayed",
            "active_conflict",
            "request_conflict",
            "rejected",
        ):
            raise ValueError("invalid resource admission kind")
        if self.kind in ("created", "active_reused", "request_replayed") and self.job is None:
            raise ValueError("successful admission requires a Job")
        if (self.kind == "rejected") != (self.rejection is not None):
            raise ValueError("only rejected admission carries a rejection")
        object.__setattr__(self, "job", copy.deepcopy(self.job))
        object.__setattr__(self, "rejection", copy.deepcopy(self.rejection))


@dataclass(frozen=True)
class ResourceAttempt:
    """Exact acquired Job/lease identity carried through finalization."""

    job_id: int
    job_uuid: str
    worker_id: str
    lease_id: str
    binding: ResourceBinding

    def __post_init__(self) -> None:
        """Reject incomplete or ambiguous attempt identity."""
        if type(self.job_id) is not int or self.job_id < 1:
            raise ValueError("job_id must be a positive integer")
        _uuid(self.job_uuid, "job_uuid")
        _identity(self.worker_id, "worker_id")
        _identity(self.lease_id, "lease_id")
        if not isinstance(self.binding, ResourceBinding):
            raise ValueError("binding must be a ResourceBinding")
