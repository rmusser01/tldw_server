"""Pydantic schemas for prototype workspace collaboration endpoints."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    pass


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
