"""Pydantic schemas for workspace CRUD."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class WorkspaceUpsertRequest(BaseModel):
    name: str
    archived: bool = False
    study_materials_policy: Literal["general", "workspace"] = "general"


class WorkspacePatchRequest(BaseModel):
    name: str | None = None
    archived: bool | None = None
    study_materials_policy: Literal["general", "workspace"] | None = None
    banner_title: str | None = None
    banner_subtitle: str | None = None
    banner_color: str | None = None
    audio_provider: str | None = None
    audio_model: str | None = None
    audio_voice: str | None = None
    audio_speed: float | None = None
    version: int = Field(..., description="Current version for optimistic locking")


class WorkspaceResponse(BaseModel):
    id: str
    name: str | None = None
    archived: bool = False
    study_materials_policy: Literal["general", "workspace"] = "general"
    deleted: bool = False
    banner_title: str | None = None
    banner_subtitle: str | None = None
    banner_color: str | None = None
    audio_provider: str | None = None
    audio_model: str | None = None
    audio_voice: str | None = None
    audio_speed: float | None = None
    created_at: str
    last_modified: str
    version: int


class WorkspaceListResponse(BaseModel):
    items: list[WorkspaceResponse]
    total: int


# --- Source schemas ---

class WorkspaceSourceCreateRequest(BaseModel):
    id: str
    media_id: int
    title: str
    source_type: str
    url: str | None = None
    position: int = 0
    selected: bool = True


class WorkspaceSourceUpdateRequest(BaseModel):
    title: str | None = None
    source_type: str | None = None
    url: str | None = None
    position: int | None = None
    selected: bool | None = None
    version: int = Field(..., description="Current version for optimistic locking")


class WorkspaceSourceResponse(BaseModel):
    id: str
    workspace_id: str
    media_id: int
    title: str
    source_type: str
    url: str | None = None
    position: int = 0
    selected: bool = True
    added_at: str
    version: int


class WorkspaceSourceSelectionRequest(BaseModel):
    selected_ids: list[str]


class WorkspaceSourceReorderRequest(BaseModel):
    ordered_ids: list[str]


class StatusResponse(BaseModel):
    ok: bool = True


# --- Artifact schemas ---

WorkspaceArtifactReviewState = Literal[
    "draft",
    "reviewing",
    "accepted",
    "needs_revision",
    "rejected",
    "exported",
    "assigned",
    "archived",
]


class WorkspaceArtifactCreateRequest(BaseModel):
    id: str
    artifact_type: str
    title: str
    status: str = "pending"
    content: str | None = None
    content_type: str = "text/markdown"
    preview_text: str | None = None
    summary: str | None = None
    review_state: WorkspaceArtifactReviewState = "draft"
    owner_scope: str = "user"
    owner_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    source_collection_id: str | None = None
    root_artifact_id: str | None = None
    artifact_version_id: str | None = None
    previous_version_id: str | None = None
    producer_metadata: dict[str, Any] = Field(default_factory=dict)
    source_lineage: dict[str, Any] = Field(default_factory=dict)
    review_metadata: dict[str, Any] = Field(default_factory=dict)
    version_metadata: dict[str, Any] = Field(default_factory=dict)
    export_refs: list[dict[str, Any]] = Field(default_factory=list)
    redaction: dict[str, Any] = Field(default_factory=lambda: {"support_safe": True, "redacted": False})
    schema_version: int = 1


class WorkspaceArtifactUpdateRequest(BaseModel):
    title: str | None = None
    status: str | None = None
    content: str | None = None
    content_type: str | None = None
    preview_text: str | None = None
    summary: str | None = None
    review_state: WorkspaceArtifactReviewState | None = None
    owner_scope: str | None = None
    owner_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    source_collection_id: str | None = None
    root_artifact_id: str | None = None
    artifact_version_id: str | None = None
    previous_version_id: str | None = None
    producer_metadata: dict[str, Any] | None = None
    source_lineage: dict[str, Any] | None = None
    review_metadata: dict[str, Any] | None = None
    version_metadata: dict[str, Any] | None = None
    export_refs: list[dict[str, Any]] | None = None
    redaction: dict[str, Any] | None = None
    schema_version: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    completed_at: str | None = None
    version: int = Field(..., description="Current version for optimistic locking")


class WorkspaceArtifactResponse(BaseModel):
    id: str
    workspace_id: str
    artifact_type: str
    title: str
    status: str = "pending"
    content: str | None = None
    content_type: str = "text/markdown"
    preview_text: str | None = None
    summary: str | None = None
    review_state: WorkspaceArtifactReviewState = "draft"
    owner_scope: str = "user"
    owner_id: str | None = None
    project_id: str | None = None
    task_id: str | None = None
    source_collection_id: str | None = None
    root_artifact_id: str
    artifact_version_id: str
    previous_version_id: str | None = None
    producer_metadata: dict[str, Any] = Field(default_factory=dict)
    source_lineage: dict[str, Any] = Field(default_factory=dict)
    review_metadata: dict[str, Any] = Field(default_factory=dict)
    version_metadata: dict[str, Any] = Field(default_factory=dict)
    export_refs: list[dict[str, Any]] = Field(default_factory=list)
    redaction: dict[str, Any] = Field(default_factory=lambda: {"support_safe": True, "redacted": False})
    schema_version: int = 1
    total_tokens: int | None = None
    total_cost_usd: float | None = None
    created_at: str
    completed_at: str | None = None
    version: int


# --- Note schemas ---

class WorkspaceNoteCreateRequest(BaseModel):
    title: str = ""
    content: str = ""
    keywords: list[str] = Field(default_factory=list)


class WorkspaceNoteUpdateRequest(BaseModel):
    title: str | None = None
    content: str | None = None
    keywords_json: str | None = None
    version: int = Field(..., description="Current version for optimistic locking")


class WorkspaceNoteResponse(BaseModel):
    id: int
    workspace_id: str
    title: str
    content: str
    keywords_json: str = "[]"
    created_at: str
    last_modified: str
    version: int
