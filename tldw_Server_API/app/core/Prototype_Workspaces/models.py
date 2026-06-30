"""Typed models and shared constants for prototype workspace runtime orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PrototypeActorType(str, Enum):
    OWNER = "owner"
    INTERNAL_COLLABORATOR = "internal_collaborator"
    EXTERNAL_COLLABORATOR = "external_collaborator"


class PrototypeJobType(str, Enum):
    BRANCH_SESSION_BOOTSTRAP = "branch_session_bootstrap"
    PREVIEW_BOOT = "preview_boot"
    SNAPSHOT_SAVE = "snapshot_save"
    PUBLISH_VALIDATE_AND_PROMOTE = "publish_validate_and_promote"


class PrototypePreviewScope(str, Enum):
    CANONICAL = "canonical"
    SESSION = "session"


class PrototypePreviewStatus(str, Enum):
    UNINITIALIZED = "uninitialized"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    REVOKED = "revoked"


class PrototypeRuntimeStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"
    REVOKED = "revoked"


@dataclass(slots=True)
class PrototypePreviewHandleRecord:
    handle_id: str
    preview_scope: PrototypePreviewScope
    scope_id: str
    prototype_workspace_id: str
    prototype_session_id: str | None
    actor_key: str
    target_ref: str
    runtime_policy_profile: str
    metadata: dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: str | None = None
    revoked_at: str | None = None


@dataclass(slots=True)
class PrototypePreviewGrant:
    preview_handle: str
    preview_url: str
    expires_at: str
    token: str


@dataclass(slots=True)
class PrototypeJobRequest:
    job_type: PrototypeJobType
    idempotency_key: str
    payload: dict[str, Any]
    owner_user_id: str | None


@dataclass(slots=True)
class PrototypePromotionResult:
    status: str
    failure_code: str | None = None
    prototype_workspace_id: str | None = None
    candidate_snapshot_id: str | None = None
    canonical_snapshot_id: str | None = None
    preview_handle: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "failure_code": self.failure_code,
            "prototype_workspace_id": self.prototype_workspace_id,
            "candidate_snapshot_id": self.candidate_snapshot_id,
            "canonical_snapshot_id": self.canonical_snapshot_id,
            "preview_handle": self.preview_handle,
            "details": dict(self.details),
        }


def actor_key_for_session(
    *,
    actor_type: str,
    actor_user_id: int | None = None,
    actor_shared_actor_id: str | None = None,
) -> str:
    normalized = PrototypeActorType(str(actor_type).strip().lower())
    if normalized == PrototypeActorType.EXTERNAL_COLLABORATOR:
        if not actor_shared_actor_id:
            raise ValueError("external collaborator actor key requires actor_shared_actor_id")
        return f"shared_actor:{actor_shared_actor_id}"
    if actor_user_id is None:
        raise ValueError("internal actor key requires actor_user_id")
    return f"user:{int(actor_user_id)}"


def preview_scope_id(
    *,
    preview_scope: str,
    prototype_workspace_id: str,
    prototype_session_id: str | None = None,
) -> str:
    normalized = PrototypePreviewScope(str(preview_scope).strip().lower())
    if normalized == PrototypePreviewScope.CANONICAL:
        return f"canonical:{prototype_workspace_id}"
    if not prototype_session_id:
        raise ValueError("session preview scope requires prototype_session_id")
    return f"session:{prototype_session_id}"
