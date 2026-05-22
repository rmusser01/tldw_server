"""Pydantic schemas for prototype workspace collaboration endpoints."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


PrototypeErrorCategory = Literal[
    "invalid_or_unavailable_link",
    "password_required",
    "invalid_password",
    "workspace_unavailable",
    "inactive_session",
    "bootstrap_failed",
    "preview_unavailable",
    "stale_promotion",
    "promotion_conflict",
    "promotion_validation_failed",
    "unauthorized",
    "missing",
    "conflict",
    "invalid_request",
]

PrototypeFrontendState = Literal[
    "link_unavailable",
    "password_required",
    "password_rejected",
    "workspace_unavailable",
    "session_inactive",
    "setup_failed",
    "preview_unavailable",
    "promotion_stale",
    "promotion_conflict",
    "promotion_failed",
    "unauthorized",
    "missing",
    "conflict",
    "invalid_request",
]


class PrototypeErrorDetail(BaseModel):
    """Machine-readable prototype workspace error detail used in HTTP responses."""

    category: PrototypeErrorCategory
    message: str = Field(..., min_length=1)
    frontend_state: PrototypeFrontendState
    retryable: bool = False


class PrototypeErrorResponse(BaseModel):
    """FastAPI error envelope for prototype workspace contract errors."""

    detail: PrototypeErrorDetail


def prototype_error_detail(
    *,
    category: PrototypeErrorCategory,
    message: str,
    frontend_state: PrototypeFrontendState,
    retryable: bool = False,
) -> dict[str, Any]:
    """Build the stable prototype error detail payload used by HTTPException."""
    return PrototypeErrorDetail(
        category=category,
        message=message,
        frontend_state=frontend_state,
        retryable=retryable,
    ).model_dump()


class PrototypeWorkspaceCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    creation_source: str = Field(..., min_length=1, max_length=64)
    description: str | None = Field(None, max_length=2000)
    prompt: str | None = None
    preview_policy: dict[str, Any] = Field(default_factory=dict)
    share_policy: dict[str, Any] = Field(default_factory=dict)
    runtime_policy: dict[str, Any] = Field(default_factory=dict)
    designated_promoter_ids: list[int] = Field(default_factory=list)


class PrototypeWorkspaceResponse(BaseModel):
    id: str
    owner_user_id: int
    title: str
    description: str | None = None
    creation_source: str
    canonical_snapshot_id: str | None = None
    last_known_good_snapshot_id: str | None = None
    canonical_preview_status: str | None = None
    publish_validation_status: str | None = None
    preview_policy: dict[str, Any] = Field(default_factory=dict)
    share_policy: dict[str, Any] = Field(default_factory=dict)
    runtime_policy: dict[str, Any] = Field(default_factory=dict)
    designated_promoter_ids: list[int] = Field(default_factory=list)
    created_at: str
    updated_at: str
    archived_at: str | None = None
    is_archived: bool = False


class PrototypeWorkspaceSessionSummaryResponse(BaseModel):
    id: str
    prototype_workspace_id: str
    base_snapshot_id: str
    actor_user_id: int | None = None
    actor_shared_actor_id: str | None = None
    actor_type: str
    share_link_id: int | None = None
    acp_session_id: str | None = None
    sandbox_session_id: str | None = None
    sandbox_run_id: str | None = None
    runtime_status: str
    preview_handle: str | None = None
    preview_status: str
    last_saved_snapshot_id: str | None = None
    last_activity_at: str | None = None
    expires_at: str | None = None
    revoked_at: str | None = None
    created_at: str
    updated_at: str
    is_revoked: bool = False


class PrototypeWorkspaceSnapshotSummaryResponse(BaseModel):
    snapshot_id: str
    prototype_workspace_id: str
    parent_snapshot_id: str | None = None
    created_from_session_id: str | None = None
    author_user_id: int | None = None
    author_shared_actor_id: str | None = None
    storage_ref: str | None = None
    diff_summary: dict[str, Any] = Field(default_factory=dict)
    prompt_summary: str | None = None
    preview_health: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    is_canonical: bool = False
    is_last_known_good: bool = False


class PrototypePromotionRequestSummaryResponse(BaseModel):
    """Owner-facing summary of a collaborator promotion request."""

    id: str
    prototype_workspace_id: str
    prototype_session_id: str
    candidate_snapshot_id: str
    requested_by_user_id: int | None = None
    requested_by_shared_actor_id: str | None = None
    status: str
    reviewed_by_user_id: int | None = None
    review_notes: str | None = None
    created_at: str
    updated_at: str


class PrototypeWorkspaceDetailResponse(PrototypeWorkspaceResponse):
    viewer_role: Literal["owner", "internal_collaborator"] = "owner"
    sessions: list[PrototypeWorkspaceSessionSummaryResponse] = Field(default_factory=list)
    snapshots: list[PrototypeWorkspaceSnapshotSummaryResponse] = Field(default_factory=list)
    promotion_requests: list[PrototypePromotionRequestSummaryResponse] = Field(default_factory=list)


class PrototypeWorkspaceSessionCreateRequest(BaseModel):
    request_nonce: str | None = Field(None, min_length=1, max_length=255)


class PrototypeCollaboratorSessionCreateRequest(BaseModel):
    session_token: str = Field(..., min_length=1)
    request_nonce: str | None = Field(None, min_length=1, max_length=255)


class PrototypeSessionJobResponse(BaseModel):
    job_id: str
    job_type: Literal["branch_session_bootstrap"] = "branch_session_bootstrap"
    status: str = "pending"
    message: str = "Prototype branch session job created"
    prototype_workspace_id: str
    prototype_session_id: str
    actor_type: str
    shared_actor_id: str | None = None
    idempotency_key: str | None = None


class PrototypePromotionCreateRequest(BaseModel):
    prototype_workspace_id: str = Field(..., min_length=1)
    prototype_session_id: str = Field(..., min_length=1)
    candidate_snapshot_id: str = Field(..., min_length=1)
    session_token: str = Field(..., min_length=1)
    request_reason: str | None = Field(None, max_length=2000)


class PrototypePromotionRequestResponse(BaseModel):
    id: str
    prototype_workspace_id: str
    prototype_session_id: str
    candidate_snapshot_id: str
    requested_by_user_id: int | None = None
    requested_by_shared_actor_id: str | None = None
    status: str
    reviewed_by_user_id: int | None = None
    review_notes: str | None = None
    created_at: str
    updated_at: str


class PrototypePromotionReviewRequest(BaseModel):
    decision: Literal["approve", "reject"]
    review_notes: str | None = Field(None, max_length=2000)
    review_baseline_snapshot_id: str | None = Field(None, min_length=1)


class PrototypePromotionReviewResponse(BaseModel):
    status: str
    failure_code: str | None = None
    prototype_workspace_id: str
    candidate_snapshot_id: str
    canonical_snapshot_id: str | None = None
    preview_handle: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class PrototypePreviewRenewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PrototypePreviewGrantResponse(BaseModel):
    preview_handle: str
    preview_scope: str
    prototype_workspace_id: str
    prototype_session_id: str | None = None
    snapshot_id: str | None = None
    preview_url: str
    expires_at: str
    token: str
    runtime_policy_profile: str
