"""Pydantic schemas for workspace CRUD."""
from __future__ import annotations

from typing import Literal

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


WorkspaceSourceLifecycleState = Literal[
    "queued",
    "ingesting",
    "extracting",
    "chunking",
    "indexing",
    "queryable",
    "partially_queryable",
    "failed",
    "retrying",
    "missing_media",
    "blocked_by_permissions",
]

WorkspaceCapabilityServiceState = Literal[
    "available",
    "private",
    "not_configured",
    "unknown",
    "blocked",
    "degraded",
]


class WorkspaceSourceReadiness(BaseModel):
    metadata_ready: bool = False
    text_extracted: bool = False
    fts_ready: bool = False
    vector_ready: bool = False
    citation_ready: bool = False
    summary_ready: bool = False
    tool_accessible: bool = False


class WorkspaceSourceJobStatus(BaseModel):
    id: int | None = None
    uuid: str | None = None
    status: str | None = None
    job_type: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None
    error_message: str | None = None


class WorkspaceSourceStatusResponse(BaseModel):
    id: str
    workspace_id: str
    media_id: int | None = None
    title: str
    source_type: str
    url: str | None = None
    selected: bool = True
    state: WorkspaceSourceLifecycleState
    status_reason: str
    readiness: WorkspaceSourceReadiness
    progress_percent: float | None = None
    progress_message: str | None = None
    job: WorkspaceSourceJobStatus | None = None
    updated_at: str = ""


class WorkspaceSourceStatusSummary(BaseModel):
    total: int = 0
    selected: int = 0
    queryable: int = 0
    partially_queryable: int = 0
    processing: int = 0
    failed: int = 0
    missing: int = 0


class WorkspaceSourceStatusListResponse(BaseModel):
    workspace_id: str
    sources: list[WorkspaceSourceStatusResponse]
    summary: WorkspaceSourceStatusSummary


class WorkspaceCapabilityService(BaseModel):
    state: WorkspaceCapabilityServiceState
    reason_code: str | None = None
    management_surface: str | None = None


class WorkspaceAllowedAction(BaseModel):
    allowed: bool
    reason_code: str | None = None


class WorkspaceCapabilitiesResponse(BaseModel):
    workspace_id: str
    workspace_kind: Literal["research_workspace"]
    access_level: Literal["owner", "editor", "viewer"] = "owner"
    source_summary: WorkspaceSourceStatusSummary
    workspace_services: dict[str, WorkspaceCapabilityService]
    allowed_actions: dict[str, WorkspaceAllowedAction]


class StatusResponse(BaseModel):
    ok: bool = True


# --- Artifact schemas ---

class WorkspaceArtifactCreateRequest(BaseModel):
    id: str
    artifact_type: str
    title: str
    status: str = "pending"
    content: str | None = None


class WorkspaceArtifactUpdateRequest(BaseModel):
    title: str | None = None
    status: str | None = None
    content: str | None = None
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
